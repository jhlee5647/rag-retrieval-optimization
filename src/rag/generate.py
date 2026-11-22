"""generate 명령: 저장된 검색 결과로 문항마다 답을 만들어 실행 폴더에 쓴다.

uv run --env-file .env python -m rag.generate configs/naive.yaml --contexts runs/retrieve/naive/<시각>
"""

import argparse
import json
import platform
import shutil
import sys
import time
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from rag.config import Config, load_config
from rag.pipeline import Resources, build_graph
from rag.runs import create_run_dir, git_commit
from rag.serve import check_model

# 검색 실행에 쓰인 config와 같아야 하는 항목. 다르면 생성 실행의 config 사본이 실제 검색 조건과 어긋난다.
RETRIEVAL_PARTS = ("index", "chunking", "embedding", "eval", "retrieve", "rerank")
# 문맥 말고 프롬프트에 들어가는 것(틀, 질문, 채팅 템플릿)의 여유분. 청크 크기를 임베딩 토크나이저로 잰 데서
# 오는 차이도 여기에 포함한다 (docs/design.md의 "실행 전 검사").
PROMPT_MARGIN_TOKENS = 1024
# 동시에 처리하는 문항 수. 설계에서 설정으로 두지 않기로 했다.
CONCURRENCY = 8


def generate(
    config_path: Path,
    config: Config,
    *,
    llm: BaseChatModel,
    contexts_dir: Path,
    runs_root: Path,
    command: list[str],
    servers: dict[str, dict[str, str]] | None = None,
) -> Path:
    """검색 실행 폴더의 문맥으로 생성 그래프를 돌려 답과 실행 기록을 쓴다.

    실행 폴더에는 `results.jsonl`, config와 프롬프트 틀의 사본, `run.json`이 생긴다. 생성 모델을
    부르기 전에 검색 조건이 같은지, 문맥이 생성 모델의 토큰 한도에 들어가는지 확인한다.

    Args:
        config_path: config 파일. 사본과 실행 폴더 이름에 쓴다.
        config: `config_path`를 읽어 검사한 config. `generation`과 `generate`가 있어야 한다.
        llm: 생성 모델.
        contexts_dir: `retrieve`의 실행 폴더. `config.yaml`과 `contexts.jsonl`을 읽는다.
        runs_root: 실행 폴더들의 최상위 폴더. 보통 `runs`.
        command: 실행한 명령줄. `run.json`에 그대로 적는다.
        servers: 시작할 때 대조한 서버 모델. `run.json`에 그대로 적는다.

    Returns:
        `<runs_root>/generate/<config 이름>/<실행 시각>/`.

    Raises:
        ValueError: 검색 조건이 다르거나, 문맥이 생성 모델의 토큰 한도를 넘는 경우.
    """
    start = time.monotonic()
    commit = git_commit()
    used = load_config(contexts_dir / "config.yaml")
    differ = [part for part in RETRIEVAL_PARTS if getattr(config, part) != getattr(used, part)]
    if differ:
        raise ValueError(
            f"config의 검색 조건이 {contexts_dir}의 검색에 쓰인 것과 다르다: {', '.join(differ)}"
        )
    with (contexts_dir / "contexts.jsonl").open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    # 문맥 수는 노드 인자가 아니라 실제 검색 결과에서 센다. 검색 노드가 바뀌어도 같은 식으로 잰다.
    most = max(len(row["contexts"]) for row in rows)
    need = most * config.chunking.size + config.generation.max_tokens + PROMPT_MARGIN_TOKENS
    if need > config.generation.max_model_len:
        raise ValueError(
            f"문맥 {most}개 × 청크 {config.chunking.size} + 답 {config.generation.max_tokens} "
            f"+ 여유 {PROMPT_MARGIN_TOKENS} = {need}토큰이 "
            f"max_model_len {config.generation.max_model_len}을 넘는다"
        )

    graph = build_graph(config.generate, Resources(config=config, llm=llm))
    states = graph.batch(
        [{k: row[k] for k in ("question", "contexts", "doc_ids")} for row in rows],
        config={"max_concurrency": CONCURRENCY},
    )

    run_dir = create_run_dir(runs_root / "generate", config_path)
    # 노드가 읽은 프롬프트 틀을 남긴다. 틀을 고치면 같은 config라도 답이 달라지기 때문이다.
    for step in config.generate:
        for params in step.values():
            if "prompt" in params:
                shutil.copyfile(params["prompt"], run_dir / Path(params["prompt"]).name)
    with (run_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for row, state in zip(rows, states, strict=True):
            result = {"id": row["id"], **{k: state[k] for k in ("answer", "contexts", "doc_ids")}}
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    record = {
        "command": command,
        "git_commit": commit,
        # 여러 장비의 실행 폴더를 합친 뒤에도 어느 장비의 실행인지 가린다.
        "host": platform.node(),
        # 문서 수, 평가 파일 해시, 색인 이름은 이 검색 실행 폴더의 run.json에 있다.
        "contexts": str(contexts_dir),
        "servers": servers,
        "elapsed_seconds": round(time.monotonic() - start, 1),
    }
    (run_dir / "run.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return run_dir


def main(argv: list[str] | None = None) -> None:
    """명령줄 인자를 읽고, config의 생성 서버에 연결해 `generate`를 실행한다.

    Args:
        argv: 명령줄 인자. None이면 `sys.argv`를 쓴다.
    """
    parser = argparse.ArgumentParser(prog="python -m rag.generate", description=__doc__)
    parser.add_argument("config", type=Path, help="config 파일 경로")
    parser.add_argument("--contexts", type=Path, required=True, help="retrieve의 실행 폴더")
    if argv is None:
        argv = sys.argv[1:]
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if config.generation is None:
        parser.error(f"{args.config}에 generation과 generate가 없다")
    g = config.generation
    servers = {"gen": check_model("gen", g.base_url, g.model, g.revision)}
    run_dir = generate(
        args.config,
        config,
        llm=ChatOpenAI(
            base_url=g.base_url,
            model=g.model,
            # vLLM은 키를 검사하지 않지만 OpenAI 클라이언트는 빈 값을 거부한다.
            api_key="EMPTY",
            temperature=g.temperature,
            max_tokens=g.max_tokens,
        ),
        contexts_dir=args.contexts,
        runs_root=Path("runs"),
        command=["python", "-m", "rag.generate", *argv],
        servers=servers,
    )
    print(run_dir)


if __name__ == "__main__":
    main()
