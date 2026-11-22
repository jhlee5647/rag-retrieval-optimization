import dataclasses
import json

import httpx
import pytest
from conftest import make_config
from langchain_qdrant import QdrantVectorStore
from pydantic import ValidationError
from qdrant_client import models

from rag.config import Rerank
from rag.index import point_id
from rag.nodes.rerank import make_reranker, rerank
from rag.pipeline import Resources

STATE = {
    "question": "which one?",
    "contexts": ["first", "second", "third"],
    "doc_ids": ["doc-a", "doc-b", "doc-c"],
}


def resources_scoring(scores, calls=None):
    """정해 둔 점수를 돌려주는 가짜 리랭커를 넣은 자원. 받은 인자는 `calls`에 남긴다."""

    def reranker(question, contexts):
        if calls is not None:
            calls.append((question, contexts))
        return scores

    return Resources(config=make_config(), reranker=reranker)


def test_keeps_top_k_chunks_in_reranker_score_order():
    """리랭커 점수가 높은 순으로 top_k개만 남기고, 문서 ID도 청크를 따라 옮긴다."""
    # Arrange
    calls = []
    run = rerank({"top_k": 2}, resources_scoring([0.1, 0.9, 0.5], calls))

    # Act
    result = run(STATE)

    # Assert
    assert result == {"contexts": ["second", "third"], "doc_ids": ["doc-b", "doc-c"]}
    assert calls == [("which one?", ["first", "second", "third"])]


def test_equal_scores_keep_search_order():
    """점수가 같은 청크는 앞 노드가 준 순서대로 남는다."""
    # Arrange
    run = rerank({"top_k": 2}, resources_scoring([0.5, 0.9, 0.5]))

    # Act
    result = run(STATE)

    # Assert
    assert result["doc_ids"] == ["doc-b", "doc-a"]


def test_top_k_above_chunk_count_keeps_all_chunks():
    """top_k가 받은 청크 수보다 크면 전부를 점수 순으로 돌려준다."""
    # Arrange
    run = rerank({"top_k": 5}, resources_scoring([0.1, 0.9, 0.5]))

    # Act
    result = run(STATE)

    # Assert
    assert result["doc_ids"] == ["doc-b", "doc-c", "doc-a"]


@pytest.mark.parametrize(
    "params, expected",
    [
        ({"top_k": 0}, ("greater_than", ("top_k",))),
        ({"top_k": 5, "topk": 5}, ("extra_forbidden", ("topk",))),
        ({}, ("missing", ("top_k",))),
    ],
    ids=["non-positive-top_k", "misspelled-key", "missing-top_k"],
)
def test_rejects_invalid_params_before_running(params, expected):
    """0 이하의 top_k, 인자 오타, 빠진 top_k는 노드를 만들 때 리랭커를 부르지 않고 거부한다."""
    # Arrange
    calls = []

    # Act
    with pytest.raises(ValidationError) as exc:
        rerank(params, resources_scoring([], calls))

    # Assert
    assert [(e["type"], e["loc"]) for e in exc.value.errors()] == [expected]
    assert calls == []


def test_dedup_keeps_only_best_chunk_of_each_document():
    """dedup을 켜면 한 문서에서 점수가 가장 높은 청크만 남기고, 빈 자리는 다음 문서의 청크로 채운다."""
    # Arrange: 점수 1위와 2위가 같은 문서의 청크다
    state = {**STATE, "doc_ids": ["doc-a", "doc-a", "doc-b"]}
    run = rerank({"top_k": 2, "dedup": True}, resources_scoring([0.9, 0.8, 0.5]))

    # Act
    result = run(state)

    # Assert
    assert result == {"contexts": ["first", "third"], "doc_ids": ["doc-a", "doc-b"]}


def test_title_adds_document_first_line_to_reranker_input_only(qdrant, embeddings):
    """title을 켜면 리랭커에는 문서의 첫 줄을 붙인 청크를 넘기고, 돌려주는 청크는 원래 글 그대로다."""
    # Arrange: 한 문서의 첫 청크와 둘째 청크. 첫 청크만 제목 줄로 시작한다
    head, tail = "Quarterly Plan\nalpha goals", "beta budget details"
    qdrant.create_collection(
        "data-test", vectors_config=models.VectorParams(size=1, distance=models.Distance.COSINE)
    )
    qdrant.upsert(
        "data-test",
        points=[
            models.PointStruct(
                id=point_id("doc-a", chunk),
                vector=[1.0],
                payload={"page_content": text, "metadata": {"doc_id": "doc-a", "chunk": chunk}},
            )
            for chunk, text in enumerate([head, tail])
        ],
    )
    store = QdrantVectorStore(
        client=qdrant,
        collection_name="data-test",
        embedding=embeddings,
        validate_collection_config=False,
    )
    calls = []
    resources = dataclasses.replace(resources_scoring([0.1, 0.9], calls), vectorstore=store)
    run = rerank({"top_k": 2, "title": True}, resources)

    # Act
    result = run(
        {"question": "which one?", "contexts": [head, tail], "doc_ids": ["doc-a", "doc-a"]}
    )

    # Assert: 첫 청크는 이미 제목 줄로 시작하므로 그대로 넘긴다
    assert calls == [("which one?", [head, "Quarterly Plan\n\nbeta budget details"])]
    assert result == {"contexts": [tail, head], "doc_ids": ["doc-a", "doc-a"]}


SECTION = Rerank(base_url="http://127.0.0.1:8004/v1", model="test/reranker", revision="abc")


def test_make_reranker_posts_pairs_and_returns_scores_in_chunk_order():
    """서버의 `/v1/rerank`에 모델, 질문, 청크들을 보내고, 점수 순으로 온 결과를 청크 순서로 되돌린다."""
    # Arrange: 실제 서버처럼 점수가 높은 순으로 응답한다
    requests = []

    def server(request):
        requests.append((str(request.url), json.loads(request.content)))
        results = [
            {"index": 1, "relevance_score": 0.9},
            {"index": 2, "relevance_score": 0.5},
            {"index": 0, "relevance_score": 0.1},
        ]
        return httpx.Response(200, json={"results": results})

    score = make_reranker(SECTION, transport=httpx.MockTransport(server))

    # Act
    scores = score("which one?", ["first", "second", "third"])

    # Assert
    assert scores == [0.1, 0.9, 0.5]
    assert requests == [
        (
            "http://127.0.0.1:8004/v1/rerank",
            {
                "model": "test/reranker",
                "query": "which one?",
                "documents": ["first", "second", "third"],
            },
        )
    ]


def test_make_reranker_raises_when_server_rejects_request():
    """서버가 요청을 거부하면(예: 입력이 길이 한도를 넘음) 점수를 지어내지 않고 오류를 낸다."""
    # Arrange
    transport = httpx.MockTransport(lambda request: httpx.Response(400, json={"error": "long"}))
    score = make_reranker(SECTION, transport=transport)

    # Act / Assert
    with pytest.raises(httpx.HTTPStatusError):
        score("which one?", ["first"])
