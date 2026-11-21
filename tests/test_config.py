import pytest
from pydantic import ValidationError

from rag.config import load_config


def write(tmp_path, text):
    path = tmp_path / "c.yaml"
    path.write_text(text, encoding="utf-8")
    return path


VALID = """
index: {url: http://127.0.0.1:6333}
chunking: {size: 512, overlap: 64}
embedding: {base_url: http://127.0.0.1:8003/v1, model: m, revision: r}
retrieve:
  - search: {mode: dense, top_k: 5}
"""


def test_repo_naive_config_loads():
    """저장소의 configs/naive.yaml이 검사를 통과한다."""
    # Arrange
    from conftest import REPO_ROOT

    # Act
    config = load_config(REPO_ROOT / "configs" / "naive.yaml")

    # Assert
    assert config.index.exact is True
    assert config.retrieve == [{"search": {"mode": "dense", "top_k": 5}}]


def test_query_instruction_defaults_to_model_card_sentence(tmp_path):
    """query_instruction을 빼면 모델 카드의 예시 문장을 쓴다."""
    # Act
    config = load_config(write(tmp_path, VALID))

    # Assert
    assert config.embedding.query_instruction == (
        "Given a web search query, retrieve relevant passages that answer the query"
    )
    assert config.index.exact is False


def test_rejects_misspelled_key(tmp_path):
    """오타로 적은 항목(overlap → overlp)을 조용히 무시하지 않고 거부한다."""
    # Arrange
    path = write(tmp_path, VALID.replace("overlap", "overlp"))

    # Act
    with pytest.raises(ValidationError) as exc:
        load_config(path)

    # Assert: overlap이 빠진 오류만이 아니라, overlp 자체를 정의되지 않은 항목으로 거부한다
    errors = {(e["type"], e["loc"]) for e in exc.value.errors()}
    assert ("extra_forbidden", ("chunking", "overlp")) in errors


def test_rejects_unknown_top_level_key(tmp_path):
    """정의되지 않은 최상위 항목을 거부한다."""
    # Arrange
    path = write(tmp_path, VALID + "eval: {k: [1, 5]}\n")

    # Act
    with pytest.raises(ValidationError) as exc:
        load_config(path)

    # Assert
    assert [(e["type"], e["loc"]) for e in exc.value.errors()] == [("extra_forbidden", ("eval",))]


def test_rejects_step_with_two_nodes(tmp_path):
    """retrieve 항목 하나에 노드를 둘 적으면 순서가 모호하므로 거부한다."""
    # Arrange
    path = write(tmp_path, VALID.replace("- search: {mode: dense, top_k: 5}", "- {a: {}, b: {}}"))

    # Act / Assert
    with pytest.raises(ValidationError, match="노드 하나만"):
        load_config(path)


def test_rejects_empty_node_list(tmp_path):
    """노드가 하나도 없는 검색 단계는 거부한다."""
    # Arrange
    path = write(
        tmp_path, VALID.replace("retrieve:\n  - search: {mode: dense, top_k: 5}", "retrieve: []")
    )

    # Act / Assert
    with pytest.raises(ValidationError, match="retrieve"):
        load_config(path)
