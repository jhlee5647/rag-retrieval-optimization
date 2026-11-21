import pytest
from conftest import add_index, make_config
from langchain_qdrant import QdrantVectorStore
from pydantic import ValidationError

from rag.nodes.search import search
from rag.pipeline import Resources

CHUNKS = [
    ("doc-a", "alpha alpha"),
    ("doc-b", "beta"),
    ("doc-a", "alpha beta"),
    ("doc-c", "gamma"),
]


@pytest.fixture
def resources(qdrant, embeddings):
    add_index(qdrant, "index-test", CHUNKS)
    store = QdrantVectorStore(
        client=qdrant,
        collection_name="index-test",
        embedding=embeddings,
        validate_collection_config=False,
    )
    return Resources(config=make_config(), vectorstore=store)


def test_returns_top_k_nearest_chunks_with_doc_ids(resources):
    """가장 가까운 청크 top_k개를 가까운 순으로 가져오고, 청크마다 문서 ID를 붙인다."""
    # Arrange
    run = search({"mode": "dense", "top_k": 2}, resources)

    # Act
    result = run({"question": "about alpha"})

    # Assert
    assert result == {"contexts": ["alpha alpha", "alpha beta"], "doc_ids": ["doc-a", "doc-a"]}


def test_embeds_question_with_instruction(resources, embeddings):
    """질문은 `Instruct: <지시문>\\nQuery:<질문>` 형식으로 임베딩한다."""
    # Arrange
    run = search({"mode": "dense", "top_k": 1}, resources)

    # Act
    run({"question": "about gamma"})

    # Assert
    instruction = resources.config.embedding.query_instruction
    assert embeddings.queries == [f"Instruct: {instruction}\nQuery:about gamma"]


@pytest.mark.parametrize("exact", [True, False])
def test_passes_exact_setting_to_qdrant(qdrant, embeddings, monkeypatch, exact):
    """config의 index.exact를 Qdrant 검색 인자로 넘긴다.

    메모리 안의 Qdrant는 이 인자를 무시하고 늘 정확 검색을 하므로, 넘긴 인자를 직접 본다.
    """
    # Arrange
    add_index(qdrant, "index-test", CHUNKS)
    store = QdrantVectorStore(
        client=qdrant,
        collection_name="index-test",
        embedding=embeddings,
        validate_collection_config=False,
    )
    config = make_config(index={"url": "http://127.0.0.1:6333", "exact": exact})
    sent = []
    query_points = qdrant.query_points

    def spy(*args, **kwargs):
        sent.append(kwargs["search_params"])
        return query_points(*args, **kwargs)

    monkeypatch.setattr(qdrant, "query_points", spy)
    run = search({"mode": "dense", "top_k": 1}, Resources(config=config, vectorstore=store))

    # Act
    run({"question": "alpha"})

    # Assert
    assert [p.exact for p in sent] == [exact]


@pytest.mark.parametrize(
    "params, expected",
    [
        ({"mode": "dense", "top_k": 5, "topk": 5}, ("extra_forbidden", ("topk",))),
        ({"mode": "hybrid", "top_k": 5}, ("literal_error", ("mode",))),
        ({"mode": "dense", "top_k": 0}, ("greater_than", ("top_k",))),
    ],
    ids=["misspelled-key", "unsupported-mode", "non-positive-top_k"],
)
def test_rejects_invalid_params_before_running(resources, embeddings, params, expected):
    """인자 오타, 지원하지 않는 mode, 0 이하의 top_k는 노드를 만들 때 검색 없이 거부한다."""
    # Act
    with pytest.raises(ValidationError) as exc:
        search(params, resources)

    # Assert
    assert [(e["type"], e["loc"]) for e in exc.value.errors()] == [expected]
    assert embeddings.queries == []
