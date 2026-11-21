"""score 명령: 생성 실행 폴더의 답을 RAGAS의 세 지표로 채점해 그 폴더에 쓴다.

    uv run --env-file .env python -m rag.score runs/generate/naive/<시각>

채점 모델과 임베딩은 그 폴더의 config 사본에서, 질문·정답·유형·근거 문서는 `EVAL_FILE`에서 읽는다.
"""

import argparse
import asyncio
import json
import math
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from rag.config import Config, load_config
from rag.data import read_questions
from rag.metrics import retrieval_hit, summarize_answers
from rag.serve import check_model

METRICS = ["faithfulness", "response_relevancy", "factual_correctness"]
# 동시에 채점하는 문항 수. 생성 단계와 같은 값을 쓴다.
CONCURRENCY = 8
# 채점 모델의 출력 한도(토큰). RAGAS 기본값 1024로는 긴 답의 주장 목록이 잘려 점수가 빈다(실측 20문항 중 2).
# 입력은 문맥 몇 개와 답이라 judge 서버의 요청 한도(32768)에 넉넉히 들어간다.
JUDGE_MAX_TOKENS = 4096

# 문항 하나의 `{"question", "answer", "contexts", "reference"}`를 받아 점수 하나를 내는 채점기.
Scorer = Callable[[dict[str, Any]], Awaitable[float]]


def score(
    run_dir: Path,
    eval_file: Path,
    scorers: dict[str, Scorer],
    servers: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    """생성 실행 폴더의 답을 채점해 `scores.jsonl`과 `metrics.json`을 그 폴더에 쓴다.

    채점에 실패한 지표는 그 문항만 빈 점수(None)로 두고 계속한다.

    Args:
        run_dir: `generate`의 실행 폴더. `config.yaml`과 `results.jsonl`을 읽는다.
        eval_file: 평가 JSONL 파일. 그 폴더의 config에 적힌 `eval.fields`로 읽는다.
        scorers: 지표 이름마다 채점기.
        servers: 시작할 때 대조한 서버 모델. `metrics.json`에 함께 적는다.

    Returns:
        `metrics.json`에 쓴 집계.

    Raises:
        ValueError: config의 `eval.fields`에 정답(`answer`)이 없는 경우.
        FileExistsError: 이미 채점한 폴더인 경우. 앞선 점수를 덮어쓰지 않는다.
        RuntimeError: 어떤 지표가 모든 문항에서 빈 경우. 서버에 닿지 못했을 가능성이 커서 쓰지 않는다.
    """
    config = load_config(run_dir / "config.yaml")
    if config.eval.fields.answer is None:
        raise ValueError(f"{run_dir}의 config에 eval.fields.answer가 없어 정답을 읽을 수 없다")
    if (run_dir / "scores.jsonl").exists():
        raise FileExistsError(f"{run_dir}는 이미 채점했다")
    questions = {q["id"]: q for q in read_questions(eval_file, config.eval.fields)}
    with (run_dir / "results.jsonl").open(encoding="utf-8") as f:
        results = [json.loads(line) for line in f]

    rows = asyncio.run(_score_all(results, questions, scorers))
    all_empty = [name for name in scorers if all(row["scores"][name] is None for row in rows)]
    if all_empty:
        raise RuntimeError(
            f"모든 문항에서 {', '.join(all_empty)} 채점이 실패했다. 채점·임베딩 서버를 확인한다"
        )
    with (run_dir / "scores.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    metrics = {**summarize_answers(rows, list(scorers)), "servers": servers}
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


async def _score_all(results, questions, scorers) -> list[dict[str, Any]]:
    gate = asyncio.Semaphore(CONCURRENCY)

    async def one(result: dict[str, Any]) -> dict[str, Any]:
        q = questions[result["id"]]
        inputs = {
            "question": q["question"],
            "answer": result["answer"],
            "contexts": result["contexts"],
            "reference": q["answer"],
        }
        scores = {}
        async with gate:
            for name, scorer in scorers.items():
                try:
                    value = float(await scorer(inputs))
                    scores[name] = None if math.isnan(value) else value
                # 채점 모델이 출력 형식을 못 맞추면 RAGAS와 그 아래 instructor가 여러 형식의 예외를 낸다.
                # 그 형식을 고르려면 간접 의존성인 instructor를 직접 불러와야 해서 넓게 잡는다. 서버 접속
                # 실패까지 빈 점수가 되는 일은 `score`가 "모든 문항이 빈 지표"로 잡는다.
                except Exception as e:  # noqa: BLE001
                    print(f"{result['id']} {name}: 빈 점수 ({type(e).__name__}: {e})")
                    scores[name] = None
        hit = retrieval_hit(result["doc_ids"], q["doc_ids"])
        return {"id": result["id"], "type": q["type"], "hit": hit, "scores": scores}

    return await asyncio.gather(*(one(result) for result in results))


def ragas_scorers(config: Config) -> dict[str, Scorer]:
    """config의 채점 서버와 임베딩 서버로 RAGAS의 세 지표 채점기를 만든다."""
    # RAGAS를 불러오는 데 몇 초 걸려, 채점할 때만 불러온다.
    from openai import AsyncOpenAI
    from ragas.embeddings.base import embedding_factory
    from ragas.llms import llm_factory
    from ragas.metrics.collections import AnswerRelevancy, FactualCorrectness, Faithfulness

    # vLLM은 키를 검사하지 않지만 OpenAI 클라이언트는 빈 값을 거부한다.
    llm = llm_factory(
        config.judge.model,
        client=AsyncOpenAI(base_url=config.judge.base_url, api_key="EMPTY"),
        temperature=0,
        max_tokens=JUDGE_MAX_TOKENS,
    )
    embeddings = embedding_factory(
        "openai",
        model=config.embedding.model,
        client=AsyncOpenAI(base_url=config.embedding.base_url, api_key="EMPTY"),
        interface="modern",
    )
    faithfulness = Faithfulness(llm=llm)
    relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)
    factual = FactualCorrectness(llm=llm, mode="f1")

    async def faithfulness_score(x: dict[str, Any]) -> float:
        result = await faithfulness.ascore(
            user_input=x["question"], response=x["answer"], retrieved_contexts=x["contexts"]
        )
        return result.value

    async def relevancy_score(x: dict[str, Any]) -> float:
        return (await relevancy.ascore(user_input=x["question"], response=x["answer"])).value

    async def factual_score(x: dict[str, Any]) -> float:
        return (await factual.ascore(response=x["answer"], reference=x["reference"])).value

    return dict(zip(METRICS, [faithfulness_score, relevancy_score, factual_score], strict=True))


def main(argv: list[str] | None = None) -> None:
    """명령줄 인자를 읽고, 그 실행 폴더의 config에 적힌 서버로 채점한다.

    Args:
        argv: 명령줄 인자. None이면 `sys.argv`를 쓴다.
    """
    parser = argparse.ArgumentParser(prog="python -m rag.score", description=__doc__)
    parser.add_argument("run_dir", type=Path, help="generate의 실행 폴더")
    args = parser.parse_args(argv)

    config = load_config(args.run_dir / "config.yaml")
    j, e = config.judge, config.embedding
    servers = {
        "judge": check_model("judge", j.base_url, j.model, j.revision),
        "embed": check_model("embed", e.base_url, e.model, e.revision),
    }
    eval_file = Path(os.environ["EVAL_FILE"])
    metrics = score(args.run_dir, eval_file, ragas_scorers(config), servers)
    print(json.dumps({"mean": metrics["mean"], "empty": metrics["empty"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
