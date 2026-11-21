import json

import pytest
from conftest import add_index

from rag.config import load_config
from rag.data import count_docs
from rag.index import index_name
from rag.retrieve import retrieve

CONFIG = """
index: {url: http://127.0.0.1:6333, exact: true}
chunking: {size: 512, overlap: 64}
embedding: {base_url: http://127.0.0.1:8003/v1, model: m, revision: r}
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


def run(setup, qdrant, embeddings, tmp_path):
    config_path, docs, eval_file = setup
    return retrieve(
        config_path,
        load_config(config_path),
        client=qdrant,
        embeddings=embeddings,
        docs_dir=docs,
        eval_file=eval_file,
        runs_root=tmp_path / "runs",
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


def test_stops_without_run_dir_when_index_missing(setup, qdrant, embeddings, tmp_path):
    """색인 조건에 맞는 색인이 없으면 검색 전에 멈추고 실행 폴더를 남기지 않는다."""
    # Arrange: 문서 수가 다른 조건의 색인만 있다
    config_path, docs, _ = setup
    other = index_name(load_config(config_path), count_docs(docs) + 1)
    add_index(qdrant, other, [("doc-a", "alpha")])

    # Act / Assert
    with pytest.raises(LookupError):
        run(setup, qdrant, embeddings, tmp_path)
    assert embeddings.queries == []
    assert not (tmp_path / "runs").exists()


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
