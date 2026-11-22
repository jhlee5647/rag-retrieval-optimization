import pytest
from conftest import add_index, make_config
from langchain_qdrant import QdrantVectorStore
from pydantic import ValidationError
from qdrant_client import models

from rag.index import build_index
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
        ({"mode": "splade", "top_k": 5}, ("literal_error", ("mode",))),
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


# 가짜 임베딩은 alpha·beta·gamma만 세므로 zulu는 BM25 검색에만 보인다.
# 질문 "alpha zulu"에 임베딩 검색은 a, b, c, d 순으로, BM25 검색은 c, b, a 순으로 가져온다(d는 겹치는 단어가 없다).
DOCS = {
    "doc-a": "alpha alpha",
    "doc-b": "alpha alpha beta zulu",
    "doc-c": "alpha beta zulu zulu",
    "doc-d": "beta beta",
}


@pytest.fixture
def bm25_resources(qdrant, embeddings, tmp_path):
    """`build_index`로 만든 색인. BM25 벡터가 실제 색인 형식대로 들어 있고, 문서 하나가 청크 하나다."""
    for name, text in DOCS.items():
        (tmp_path / f"{name}.txt").write_text(text, encoding="utf-8")
    config = make_config(index={"url": "http://127.0.0.1:6333", "exact": True})
    build_index(qdrant, "index-test", config, tmp_path, embeddings, lambda text: len(text.split()))
    store = QdrantVectorStore(
        client=qdrant,
        collection_name="index-test",
        embedding=embeddings,
        validate_collection_config=False,
    )
    return Resources(config=config, vectorstore=store)


def test_bm25_ranks_chunks_by_question_words_without_embedding(bm25_resources, embeddings):
    """BM25 검색은 질문의 단어가 많이 든 청크부터 가져오고, 임베딩 모델을 부르지 않는다."""
    # Arrange
    run = search({"mode": "bm25", "top_k": 3}, bm25_resources)

    # Act
    result = run({"question": "Zulu?"})

    # Assert: zulu가 두 번 든 c가 먼저고, zulu가 없는 a와 d는 나오지 않는다
    assert result == {
        "contexts": ["alpha beta zulu zulu", "alpha alpha beta zulu"],
        "doc_ids": ["doc-c", "doc-b"],
    }
    assert embeddings.queries == []


def test_hybrid_puts_chunk_ranked_high_by_both_searches_first(bm25_resources):
    """혼합 검색은 두 순위를 합친다. 두 검색에서 모두 2위인 청크가 각 검색의 1위를 앞선다."""
    # Arrange
    question = {"question": "alpha zulu"}
    dense = search({"mode": "dense", "top_k": 1}, bm25_resources)
    bm25 = search({"mode": "bm25", "top_k": 1}, bm25_resources)
    hybrid = search({"mode": "hybrid", "top_k": 1, "candidates": 2}, bm25_resources)

    # Act / Assert
    assert dense(question)["doc_ids"] == ["doc-a"]
    assert bm25(question)["doc_ids"] == ["doc-c"]
    assert hybrid(question) == {"contexts": ["alpha alpha beta zulu"], "doc_ids": ["doc-b"]}


def test_hybrid_fuses_only_candidates_of_each_search(bm25_resources):
    """두 검색이 각각 candidates개만 내놓는다. 어느 검색의 후보에도 못 든 청크는 top_k가 남아도 안 나온다."""
    # Arrange
    run = search({"mode": "hybrid", "top_k": 4, "candidates": 2}, bm25_resources)

    # Act
    doc_ids = run({"question": "alpha zulu"})["doc_ids"]

    # Assert: 후보는 임베딩 검색의 a, b와 BM25 검색의 c, b다. a와 c는 한 검색의 1위로 점수가 같다
    assert doc_ids[0] == "doc-b"
    assert sorted(doc_ids[1:]) == ["doc-a", "doc-c"]


@pytest.mark.parametrize("exact", [True, False])
def test_hybrid_sends_candidates_exact_setting_and_fusion_constant(
    qdrant, bm25_resources, monkeypatch, exact
):
    """혼합 검색은 후보 수, 임베딩 검색의 정확 검색 여부, 순위 융합 상수 60을 Qdrant에 넘긴다.

    메모리 안의 Qdrant는 정확 검색 인자를 무시하고, 상수가 달라도 이 색인에서는 순서가 같아 넘긴 인자를 직접 본다.
    """
    # Arrange
    config = make_config(index={"url": "http://127.0.0.1:6333", "exact": exact})
    resources = Resources(config=config, vectorstore=bm25_resources.vectorstore)
    sent = []
    query_points = qdrant.query_points

    def spy(*args, **kwargs):
        sent.append(kwargs)
        return query_points(*args, **kwargs)

    monkeypatch.setattr(qdrant, "query_points", spy)
    run = search({"mode": "hybrid", "top_k": 1, "candidates": 3}, resources)

    # Act
    run({"question": "alpha zulu"})

    # Assert
    (kwargs,) = sent
    dense, bm25 = kwargs["prefetch"]
    assert (dense.limit, dense.params.exact) == (3, exact)
    assert (bm25.limit, bm25.using, bm25.params) == (3, "bm25", None)
    assert kwargs["query"] == models.RrfQuery(rrf=models.Rrf(k=60))


@pytest.mark.parametrize(
    "params, expected",
    [
        ({"mode": "hybrid", "top_k": 5}, ("value_error", ())),
        ({"mode": "dense", "top_k": 5, "candidates": 50}, ("value_error", ())),
        ({"mode": "bm25", "top_k": 5, "candidates": 50}, ("value_error", ())),
        ({"mode": "hybrid", "top_k": 5, "candidates": 0}, ("greater_than", ("candidates",))),
    ],
    ids=["hybrid-without", "dense-with", "bm25-with", "non-positive"],
)
def test_rejects_candidates_that_do_not_fit_mode(resources, embeddings, params, expected):
    """candidates는 hybrid에서 꼭 적고 다른 mode에서는 적지 않는다. 어기면 검색 없이 거부한다."""
    # Act
    with pytest.raises(ValidationError) as exc:
        search(params, resources)

    # Assert
    assert [(e["type"], e["loc"]) for e in exc.value.errors()] == [expected]
    assert embeddings.queries == []
