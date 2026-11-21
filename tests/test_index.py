import pytest
from conftest import REPO_ROOT, add_index, make_config

from rag.config import load_config
from rag.index import index_name, require_index


def test_index_name_of_existing_index_is_fixed():
    """장비의 기존 색인 조건(naive.yaml, 문서 511,956개)의 이름이 바뀌지 않는다.

    장비의 컬렉션 data-d89aa11508031fc4에 이 이름을 별칭으로 붙인다. 계산식이 바뀌어 이 테스트가
    깨지면 별칭을 못 찾아 문서 전체의 색인을 새로 만들기 시작하므로, 별칭도 다시 붙여야 한다.
    """
    # Arrange
    config = load_config(REPO_ROOT / "configs" / "naive.yaml")

    # Act
    name = index_name(config, doc_count=511_956)

    # Assert
    assert name == "index-1fd89b43b5786220"


@pytest.mark.parametrize(
    "overrides, doc_count",
    [
        ({"chunking": {"size": 256, "overlap": 64}}, 10),
        ({"chunking": {"size": 512, "overlap": 32}}, 10),
        (
            {
                "embedding": {
                    "base_url": "http://127.0.0.1:8003/v1",
                    "model": "other",
                    "revision": "abc",
                }
            },
            10,
        ),
        (
            {
                "embedding": {
                    "base_url": "http://127.0.0.1:8003/v1",
                    "model": "test/model",
                    "revision": "def",
                }
            },
            10,
        ),
        ({}, 11),
    ],
    ids=["size", "overlap", "model", "revision", "doc_count"],
)
def test_index_name_changes_with_each_condition(overrides, doc_count):
    """색인 조건이 하나라도 다르면 다른 색인 이름이 나온다."""
    # Act / Assert
    assert index_name(make_config(**overrides), doc_count) != index_name(make_config(), 10)


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "embedding": {
                "base_url": "http://127.0.0.1:9999/v1",
                "model": "test/model",
                "revision": "abc",
                "query_instruction": "x",
            }
        },
        {"index": {"url": "http://localhost:6333", "exact": True}},
        {"retrieve": [{"search": {"mode": "dense", "top_k": 9}}]},
    ],
    ids=["embedding-server-and-instruction", "index-url-and-exact", "retrieve-nodes"],
)
def test_index_name_ignores_non_index_settings(overrides):
    """문서 청크에 영향이 없는 설정(서버 주소, 질문 지시문, 검색 노드)은 색인을 바꾸지 않는다."""
    # Act / Assert
    assert index_name(make_config(**overrides), 10) == index_name(make_config(), 10)


def test_require_index_passes_when_alias_exists(qdrant):
    """색인 이름의 별칭이 있으면 통과한다."""
    # Arrange
    add_index(qdrant, "index-0123456789abcdef", [("d1", "alpha")])

    # Act / Assert: 예외가 나지 않는다
    require_index(qdrant, "index-0123456789abcdef")


def test_require_index_stops_when_alias_missing(qdrant):
    """별칭이 없으면 그 이름을 대며 멈춘다. 다른 이름의 별칭은 대신 쓰지 않는다."""
    # Arrange
    add_index(qdrant, "index-ffffffffffffffff", [("d1", "alpha")])

    # Act / Assert
    with pytest.raises(LookupError, match="index-0123456789abcdef"):
        require_index(qdrant, "index-0123456789abcdef")


def test_require_index_ignores_collection_without_alias(qdrant):
    """같은 이름의 컬렉션이 있어도 별칭이 아니면 완성된 색인으로 보지 않는다."""
    # Arrange
    from qdrant_client import models

    qdrant.create_collection(
        "index-0123456789abcdef",
        vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE),
    )

    # Act / Assert
    with pytest.raises(LookupError):
        require_index(qdrant, "index-0123456789abcdef")
