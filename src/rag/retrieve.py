"""retrieve 명령: 평가 문항마다 검색하고 검색 지표를 계산해 실행 폴더에 쓴다.

    uv run --env-file .env python -m rag.retrieve configs/naive.yaml

문서 폴더와 평가 파일은 환경변수 `DOCS_DIR`과 `EVAL_FILE`에서 읽는다.
"""

import argparse
import json
import os
import platform
import sys
import time
from collections.abc import Callable
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from rag.config import Config, load_config
from rag.data import count_docs, read_questions
from rag.index import build_index, has_index, index_name, token_counter
from rag.metrics import question_scores, summarize
from rag.nodes.rerank import make_reranker
from rag.pipeline import Resources, build_graph
from rag.runs import create_run_dir, file_sha256, git_commit
from rag.serve import check_model

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
    command: list[str],
    token_len: Callable[[str], int],
    limit: int | None = None,
    servers: dict[str, dict[str, str]] | None = None,
    reranker: Callable[[str, list[str]], list[float]] | None = None,
    llm: BaseChatModel | None = None,
) -> Path:
    """평가 문항마다 검색 그래프를 돌려 `contexts.jsonl`, 검색 지표, config 사본을 쓴다.

    무엇으로 돌렸는지는 `run.json`에 남긴다. 검색 지표는 문항별 점수 `scores.jsonl`과 평균
    `metrics.json`이다. 근거 문서가 지정되지 않은 문항은 대조할 정답이 없어 지표에서 빼지만, 생성
    단계가 그 문맥도 쓰므로 검색은 한다.

    검색을 시작하기 전에 노드 목록을 확인하고, 색인 조건에 맞는 색인이 없으면 새로 만든다. 실행 폴더는
    검색이 끝난 뒤에 만들어, 실패한 실행이 빈 폴더를 남기지 않게 한다.

    Args:
        config_path: config 파일. 사본과 실행 폴더 이름에 쓴다.
        config: `config_path`를 읽어 검사한 config.
        client: Qdrant 클라이언트.
        embeddings: 질문을 임베딩할 모델.
        docs_dir: 문서 폴더. 색인 이름은 파일 수로 계산하고, 색인이 없을 때만 문서를 읽는다.
        eval_file: 평가 JSONL 파일.
        runs_root: 실행 폴더들의 최상위 폴더. 보통 `runs`.
        command: 실행한 명령줄. `run.json`에 그대로 적는다.
        token_len: 색인을 만들 때 청크 길이를 세는 함수.
        limit: 평가 파일의 앞 몇 문항만 돌릴지. None이면 전부 돌린다.
        servers: 시작할 때 대조한 서버 모델. `run.json`에 그대로 적는다.
        reranker: 질문과 청크들로 청크마다 점수를 내는 함수. config에 rerank 노드가 있을 때 넘긴다.
        llm: 생성 모델. config에 expand나 select 노드가 있을 때 넘긴다.

    Returns:
        `<runs_root>/retrieve/<config 이름>/<실행 시각>/`.

    Raises:
        RuntimeError: 색인을 만들다 점 수가 맞지 않은 경우 (`build_index`).
        ValueError: 등록되지 않은 노드 이름이 있는 경우.
    """
    start = time.monotonic()
    commit = git_commit()
    doc_count = count_docs(docs_dir)
    name = index_name(config, doc_count)
    print(f"문서 {doc_count}개, 색인 {name}")
    # 색인 확인과 만들기를 따로 하므로, 저장소를 만들 때 컬렉션을 검사하지 않는다.
    # 켜 두면 임베딩 서버도 한 번 더 부른다.
    store = QdrantVectorStore(
        client=client,
        collection_name=name,
        embedding=embeddings,
        validate_collection_config=False,
    )
    graph = build_graph(
        config.retrieve,
        Resources(config=config, vectorstore=store, reranker=reranker, llm=llm),
    )
    if not has_index(client, name):
        print(
            f"색인 {name}이 없어 문서 {doc_count}개로 새로 만든다. 문서 전체는 하루 가까이 걸린다"
        )
        build_index(client, name, config, docs_dir, embeddings, token_len)

    questions = read_questions(eval_file, config.eval.fields)[:limit]
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
    record = {
        "command": command,
        "git_commit": commit,
        # 여러 장비의 실행 폴더를 합친 뒤에도 어느 장비의 실행인지 가린다.
        "host": platform.node(),
        "doc_count": doc_count,
        # 평가 파일은 일부만 돌려도 파일 전체로 해시한다. 어떤 파일의 앞 몇 문항인지 가리기 위해서다.
        "eval_file_sha256": file_sha256(eval_file),
        "index": name,
        "limit": limit,
        "servers": servers,
        "elapsed_seconds": round(time.monotonic() - start, 1),
    }
    (run_dir / "run.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return run_dir


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"1 이상이어야 한다: {text}")
    return value


def main(argv: list[str] | None = None) -> None:
    """명령줄 인자를 읽고, config의 서버에 연결해 `retrieve`를 실행한다.

    Args:
        argv: 명령줄 인자. None이면 `sys.argv`를 쓴다.
    """
    parser = argparse.ArgumentParser(prog="python -m rag.retrieve", description=__doc__)
    parser.add_argument("config", type=Path, help="config 파일 경로")
    parser.add_argument(
        "--limit", type=_positive_int, help="평가 파일의 앞 N문항만 돌린다. 빼면 전부 돌린다"
    )
    if argv is None:
        argv = sys.argv[1:]
    args = parser.parse_args(argv)

    config = load_config(args.config)
    # expand와 select 노드는 검색 단계에서 생성 모델을 부른다.
    uses_llm = any(name in step for step in config.retrieve for name in ("expand", "select"))
    if uses_llm and config.generation is None:
        parser.error(f"{args.config}에 generation이 없다. expand와 select 노드는 생성 모델을 쓴다")
    e = config.embedding
    # 오래 걸리는 일을 시작하기 전에, 떠 있는 서버가 config의 모델인지 확인한다.
    servers = {"embed": check_model("embed", e.base_url, e.model, e.revision)}
    reranker = None
    if config.rerank is not None:
        r = config.rerank
        servers["rerank"] = check_model("rerank", r.base_url, r.model, r.revision)
        reranker = make_reranker(r)
    llm = None
    if uses_llm:
        g = config.generation
        servers["gen"] = check_model("gen", g.base_url, g.model, g.revision)
        # 답의 토큰 한도는 노드가 정한다. generation.max_tokens는 답을 만드는 생성 단계의 값이다.
        llm = ChatOpenAI(
            base_url=g.base_url, model=g.model, api_key="EMPTY", temperature=g.temperature
        )
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
        command=["python", "-m", "rag.retrieve", *argv],
        token_len=token_counter(config.embedding.model, config.embedding.revision),
        limit=args.limit,
        servers=servers,
        reranker=reranker,
        llm=llm,
    )
    print(run_dir)


if __name__ == "__main__":
    main()
