import hashlib
import json
import re

import pytest
from conftest import add_index
from langchain_core.messages import AIMessage

from rag.config import load_config
from rag.data import count_docs
from rag.index import has_index, index_name
from rag.retrieve import main, retrieve

CONFIG = """
index: {url: http://127.0.0.1:6333, exact: true}
chunking: {size: 512, overlap: 64}
embedding: {base_url: http://127.0.0.1:8003/v1, model: m, revision: r, max_input_tokens: 8192}
eval: {fields: {id: id, question: question, doc_ids: gold, type: type}, k: [1]}
retrieve:
  - search: {mode: dense, top_k: 1}
"""


@pytest.fixture
def setup(tmp_path):
    """문서 폴더 2개 문서, 평가 파일 2문항, config 파일을 만든다."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "doc-a.txt").write_text("alpha")
    (docs / "doc-b.txt").write_text("beta")
    eval_file = tmp_path / "eval.jsonl"
    eval_file.write_text(
        json.dumps({"id": "q1", "question": "beta?", "gold": ["doc-b"], "type": "basic"})
        + "\n"
        + json.dumps({"id": "q2", "question": "alpha?", "gold": ["doc-b"], "type": "basic"})
        + "\n"
    )
    config_path = tmp_path / "naive.yaml"
    config_path.write_text(CONFIG)
    return config_path, docs, eval_file


def run(setup, qdrant, embeddings, tmp_path, limit=None, servers=None):
    config_path, docs, eval_file = setup
    return retrieve(
        config_path,
        load_config(config_path),
        client=qdrant,
        embeddings=embeddings,
        docs_dir=docs,
        eval_file=eval_file,
        runs_root=tmp_path / "runs",
        command=["python", "-m", "rag.retrieve", str(config_path)],
        # 문서가 짧아 글자 수로 세도 청크 하나씩이다
        token_len=len,
        limit=limit,
        servers=servers,
    )


def test_writes_contexts_and_config_copy(setup, qdrant, embeddings, tmp_path):
    """문항마다 검색해 contexts.jsonl과 config 사본을 runs/retrieve/<config 이름>/<시각>/에 쓴다."""
    # Arrange
    config_path, docs, _ = setup
    name = index_name(load_config(config_path), count_docs(docs))
    add_index(qdrant, name, [("doc-a", "alpha"), ("doc-b", "beta")])

    # Act
    run_dir = run(setup, qdrant, embeddings, tmp_path)

    # Assert
    assert run_dir.parent == tmp_path / "runs" / "retrieve" / "naive"
    assert len(run_dir.name) == len("YYYYMMDD-HHMMSS")
    assert (run_dir / "config.yaml").read_text() == CONFIG
    rows = [json.loads(line) for line in (run_dir / "contexts.jsonl").read_text().splitlines()]
    assert rows == [
        {"id": "q1", "question": "beta?", "contexts": ["beta"], "doc_ids": ["doc-b"]},
        {"id": "q2", "question": "alpha?", "contexts": ["alpha"], "doc_ids": ["doc-a"]},
    ]


def test_builds_missing_index_then_searches(setup, qdrant, embeddings, tmp_path):
    """색인 조건에 맞는 색인이 없으면 멈추지 않고 문서 폴더로 새로 만든 뒤 검색한다."""
    # Arrange: 문서 수가 다른 조건의 색인만 있다
    config_path, docs, _ = setup
    other = index_name(load_config(config_path), count_docs(docs) + 1)
    add_index(qdrant, other, [("doc-a", "alpha")])

    # Act
    run_dir = run(setup, qdrant, embeddings, tmp_path)

    # Assert
    name = index_name(load_config(config_path), count_docs(docs))
    assert has_index(qdrant, name)
    rows = [json.loads(line) for line in (run_dir / "contexts.jsonl").read_text().splitlines()]
    assert [r["doc_ids"] for r in rows] == [["doc-b"], ["doc-a"]]


def test_rejects_unregistered_node_before_index_check(setup, qdrant, embeddings, tmp_path):
    """노드 이름 오타는 색인 확인보다 먼저, 검색 없이 거부한다."""
    # Arrange
    config_path, _, _ = setup
    config_path.write_text(CONFIG.replace("- search:", "- serach:"))

    # Act / Assert
    with pytest.raises(ValueError, match="serach"):
        run(setup, qdrant, embeddings, tmp_path)
    assert embeddings.queries == []


def test_writes_retrieval_scores_and_metrics(setup, qdrant, embeddings, tmp_path):
    """문항별 점수와 평균을 쓰고, 근거 문서가 없는 문항은 검색만 하고 지표에서 뺀다."""
    # Arrange: q1은 근거 문서를 맞히고, q2는 놓치고, q3는 근거 문서가 지정되지 않았다
    config_path, docs, eval_file = setup
    rows = [
        {"id": "q1", "question": "beta?", "gold": ["doc-b"], "type": "basic"},
        {"id": "q2", "question": "alpha?", "gold": ["doc-b"], "type": "basic"},
        {"id": "q3", "question": "gamma?", "gold": [], "type": "unanswerable"},
    ]
    eval_file.write_text("".join(json.dumps(r) + "\n" for r in rows))
    name = index_name(load_config(config_path), count_docs(docs))
    add_index(qdrant, name, [("doc-a", "alpha"), ("doc-b", "beta")])

    # Act
    run_dir = run(setup, qdrant, embeddings, tmp_path)

    # Assert
    assert len((run_dir / "contexts.jsonl").read_text().splitlines()) == 3
    scores = [json.loads(line) for line in (run_dir / "scores.jsonl").read_text().splitlines()]
    assert [(s["id"], s["scores"]["recall@1"]) for s in scores] == [("q1", 1.0), ("q2", 0.0)]
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert (metrics["n_scored"], metrics["n_skipped"]) == (2, 1)
    assert metrics["mean"]["recall@1"] == 0.5
    assert list(metrics["by_type"]) == ["basic"]


def test_writes_run_record(setup, qdrant, embeddings, tmp_path):
    """run.json에 명령, git 커밋, 문서 수, 평가 파일 해시, 색인 이름, 문항 제한, 소요 시간을 남긴다."""
    # Arrange
    config_path, docs, eval_file = setup
    name = index_name(load_config(config_path), count_docs(docs))
    add_index(qdrant, name, [("doc-a", "alpha"), ("doc-b", "beta")])

    # Act
    run_dir = run(setup, qdrant, embeddings, tmp_path)

    # Assert
    record = json.loads((run_dir / "run.json").read_text())
    assert record["command"] == ["python", "-m", "rag.retrieve", str(config_path)]
    assert re.fullmatch(r"[0-9a-f]{40}", record["git_commit"])
    assert record["doc_count"] == 2
    assert record["eval_file_sha256"] == hashlib.sha256(eval_file.read_bytes()).hexdigest()
    assert record["index"] == name
    assert record["limit"] is None
    assert record["elapsed_seconds"] >= 0


def test_limit_runs_only_first_questions(setup, qdrant, embeddings, tmp_path):
    """limit을 주면 평가 파일의 앞 N문항만 검색하고 지표를 내며, 그 값을 run.json에 적는다."""
    # Arrange
    config_path, docs, _ = setup
    name = index_name(load_config(config_path), count_docs(docs))
    add_index(qdrant, name, [("doc-a", "alpha"), ("doc-b", "beta")])

    # Act
    run_dir = run(setup, qdrant, embeddings, tmp_path, limit=1)

    # Assert
    contexts = (run_dir / "contexts.jsonl").read_text().splitlines()
    assert [json.loads(line)["id"] for line in contexts] == ["q1"]
    assert json.loads((run_dir / "metrics.json").read_text())["n_scored"] == 1
    assert json.loads((run_dir / "run.json").read_text())["limit"] == 1
    assert embeddings.queries and len(embeddings.queries) == 1


@pytest.mark.parametrize("value", ["0", "-3"])
def test_rejects_limit_below_one(value, capsys):
    """--limit이 1보다 작으면 config를 읽거나 서버에 연결하기 전에 거부한다."""
    with pytest.raises(SystemExit):
        main(["configs/does-not-exist.yaml", "--limit", value])
    assert "1 이상" in capsys.readouterr().err


def test_records_checked_servers(setup, qdrant, embeddings, tmp_path):
    """시작할 때 대조한 서버 모델을 run.json에 남긴다."""
    # Arrange
    config_path, docs, _ = setup
    name = index_name(load_config(config_path), count_docs(docs))
    add_index(qdrant, name, [("doc-a", "alpha"), ("doc-b", "beta")])
    checked = {"embed": {"container": "rag-embed", "image": "i", "model": "m", "revision": "r"}}

    # Act
    run_dir = run(setup, qdrant, embeddings, tmp_path, servers=checked)

    # Assert
    assert json.loads((run_dir / "run.json").read_text())["servers"] == checked


def test_records_host_name(setup, qdrant, embeddings, tmp_path, monkeypatch):
    """여러 장비의 실행 폴더를 합친 뒤에도 어느 장비의 실행인지 가리도록, run.json에 호스트 이름을 남긴다."""
    # Arrange
    config_path, docs, _ = setup
    name = index_name(load_config(config_path), count_docs(docs))
    add_index(qdrant, name, [("doc-a", "alpha"), ("doc-b", "beta")])
    monkeypatch.setattr("platform.node", lambda: "node-b")

    # Act
    run_dir = run(setup, qdrant, embeddings, tmp_path)

    # Assert
    assert json.loads((run_dir / "run.json").read_text())["host"] == "node-b"


def test_reranks_search_results_with_given_reranker(setup, qdrant, embeddings, tmp_path):
    """config에 rerank 노드가 있으면, 넘겨받은 리랭커로 검색 결과를 다시 줄 세워 쓴다."""
    # Arrange: 검색은 두 청크를 모두 가져오고, 리랭커는 beta가 든 청크에 낮은 점수를 준다
    config_path, docs, eval_file = setup
    config_path.write_text(
        CONFIG.replace("top_k: 1}", "top_k: 2}\n  - rerank: {top_k: 1}")
        + "rerank: {base_url: http://127.0.0.1:8004/v1, model: m, revision: r}\n"
    )
    config = load_config(config_path)
    add_index(qdrant, index_name(config, count_docs(docs)), [("doc-a", "alpha"), ("doc-b", "beta")])

    # Act
    run_dir = retrieve(
        config_path,
        config,
        client=qdrant,
        embeddings=embeddings,
        docs_dir=docs,
        eval_file=eval_file,
        runs_root=tmp_path / "runs",
        command=[],
        token_len=len,
        reranker=lambda question, contexts: [-float("beta" in text) for text in contexts],
    )

    # Assert: 첫 문항 "beta?"의 검색 1위는 doc-b지만 리랭커가 doc-a를 고른다
    rows = [json.loads(line) for line in (run_dir / "contexts.jsonl").read_text().splitlines()]
    assert [row["doc_ids"] for row in rows] == [["doc-a"], ["doc-a"]]


class SecondPickLLM:
    """언제나 둘째 청크를 고르는 가짜 생성 모델."""

    def bind(self, **kwargs):
        return self

    def invoke(self, prompt):
        return AIMessage(content="[2]")


def test_selects_search_results_with_given_llm(setup, qdrant, embeddings, tmp_path):
    """config에 select 노드가 있으면, 넘겨받은 생성 모델이 고른 청크를 쓴다."""
    # Arrange: 검색은 두 청크를 모두 가져오고, 생성 모델은 둘째를 고른다
    config_path, docs, eval_file = setup
    prompt = tmp_path / "select.txt"
    prompt.write_text("{question} {documents} {top_k}")
    config_path.write_text(
        CONFIG.replace("top_k: 1}", f"top_k: 2}}\n  - select: {{prompt: {prompt}, top_k: 1}}")
    )
    config = load_config(config_path)
    add_index(qdrant, index_name(config, count_docs(docs)), [("doc-a", "alpha"), ("doc-b", "beta")])

    # Act
    run_dir = retrieve(
        config_path,
        config,
        client=qdrant,
        embeddings=embeddings,
        docs_dir=docs,
        eval_file=eval_file,
        runs_root=tmp_path / "runs",
        command=[],
        token_len=len,
        llm=SecondPickLLM(),
    )

    # Assert: 검색 1위는 문항마다 다르지만("beta?"는 doc-b, "alpha?"는 doc-a) 둘째가 넘어간다
    rows = [json.loads(line) for line in (run_dir / "contexts.jsonl").read_text().splitlines()]
    assert [row["doc_ids"] for row in rows] == [["doc-a"], ["doc-b"]]


def test_main_rejects_select_node_without_generation(setup, tmp_path, capsys):
    """select 노드가 있는데 generation이 없는 config는 서버에 접속하기 전에 거부한다."""
    # Arrange
    config_path, _, _ = setup
    config_path.write_text(CONFIG + "  - select: {prompt: prompts/select.txt, top_k: 1}\n")

    # Act / Assert
    with pytest.raises(SystemExit):
        main([str(config_path)])
    assert "generation이 없다" in capsys.readouterr().err
