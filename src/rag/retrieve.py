"""retrieve 명령: 평가 문항마다 검색하고 검색 지표를 계산해 실행 폴더에 쓴다.

    uv run --env-file .env python -m rag.retrieve configs/naive.yaml

문서 폴더와 평가 파일은 환경변수 `DOCS_DIR`과 `EVAL_FILE`에서 읽는다.
"""

import argparse
import json
import os
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from rag.config import Config, load_config
from rag.data import count_docs, read_questions
from rag.index import index_name, require_index
from rag.metrics import question_scores, summarize
from rag.pipeline import Resources, build_graph
from rag.runs import create_run_dir

# 문서 전체에 대한 정확 검색은 클라이언트의 기본 timeout 5초를 넘는다 (장비에서 측정).
QDRANT_TIMEOUT_SECONDS = 120
# 동시에 처리하는 문항 수. 설계에서 설정으로 두지 않기로 했다.
CONCURRENCY = 8


def retrieve(
    config_path: Path,
    config: Config,
    *,
    client: QdrantClient,
    embeddings: Embeddings,
    docs_dir: Path,
    eval_file: Path,
    runs_root: Path,
) -> Path:
    """평가 문항마다 검색 그래프를 돌려 `contexts.jsonl`, 검색 지표, config 사본을 쓴다.

    검색 지표는 문항별 점수 `scores.jsonl`과 평균 `metrics.json`이다. 근거 문서가 지정되지 않은 문항은
    대조할 정답이 없어 지표에서 빼지만, 생성 단계가 그 문맥도 쓰므로 검색은 한다.

    검색을 시작하기 전에 노드 목록과 색인을 모두 확인한다. 실행 폴더는 검색이 끝난 뒤에 만들어,
    실패한 실행이 빈 폴더를 남기지 않게 한다.

    Args:
        config_path: config 파일. 사본과 실행 폴더 이름에 쓴다.
        config: `config_path`를 읽어 검사한 config.
        client: Qdrant 클라이언트.
        embeddings: 질문을 임베딩할 모델.
        docs_dir: 문서 폴더. 색인 이름을 계산하는 데 파일 수만 쓴다.
        eval_file: 평가 JSONL 파일.
        runs_root: 실행 폴더들의 최상위 폴더. 보통 `runs`.

    Returns:
        `<runs_root>/retrieve/<config 이름>/<실행 시각>/`.

    Raises:
        LookupError: 색인 조건에 맞는 색인이 Qdrant에 없는 경우.
        ValueError: 등록되지 않은 노드 이름이 있는 경우.
    """
    doc_count = count_docs(docs_dir)
    name = index_name(config, doc_count)
    print(f"문서 {doc_count}개, 색인 {name}")
    # 색인 검사를 맡긴 require_index와 겹치므로, 저장소를 만들 때 컬렉션을 검사하지 않는다.
    # 켜 두면 임베딩 서버도 한 번 더 부른다.
    store = QdrantVectorStore(
        client=client,
        collection_name=name,
        embedding=embeddings,
        validate_collection_config=False,
    )
    graph = build_graph(config.retrieve, Resources(config=config, vectorstore=store))
    require_index(client, name)

    questions = read_questions(eval_file, config.eval.fields)
    states = graph.batch(
        [{"question": q["question"]} for q in questions],
        config={"max_concurrency": CONCURRENCY},
    )

    run_dir = create_run_dir(runs_root / "retrieve", config_path)
    with (run_dir / "contexts.jsonl").open("w", encoding="utf-8") as f:
        for q, state in zip(questions, states, strict=True):
            f.write(json.dumps({"id": q["id"], **state}, ensure_ascii=False) + "\n")

    ks = config.eval.k
    scored = [
        {
            "id": q["id"],
            "type": q["type"],
            "scores": question_scores(s["doc_ids"], q["doc_ids"], ks),
        }
        for q, s in zip(questions, states, strict=True)
        if q["doc_ids"]
    ]
    with (run_dir / "scores.jsonl").open("w", encoding="utf-8") as f:
        for row in scored:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    metrics = summarize(scored, skipped=len(questions) - len(scored))
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return run_dir


def main(argv: list[str] | None = None) -> None:
    """명령줄 인자를 읽고, config의 서버에 연결해 `retrieve`를 실행한다.

    Args:
        argv: 명령줄 인자. None이면 `sys.argv`를 쓴다.
    """
    parser = argparse.ArgumentParser(prog="python -m rag.retrieve", description=__doc__)
    parser.add_argument("config", type=Path, help="config 파일 경로")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    run_dir = retrieve(
        args.config,
        config,
        client=QdrantClient(url=config.index.url, timeout=QDRANT_TIMEOUT_SECONDS),
        embeddings=OpenAIEmbeddings(
            base_url=config.embedding.base_url,
            model=config.embedding.model,
            # vLLM은 키를 검사하지 않지만 OpenAI 클라이언트는 빈 값을 거부한다.
            api_key="EMPTY",
            # 켜 두면 tiktoken으로 토큰화한 정수 배열을 보내, vLLM의 모델과 맞지 않는다.
            check_embedding_ctx_length=False,
        ),
        docs_dir=Path(os.environ["DOCS_DIR"]),
        eval_file=Path(os.environ["EVAL_FILE"]),
        runs_root=Path("runs"),
    )
    print(run_dir)


if __name__ == "__main__":
    main()
