"""GPU와 서버 없이 도는 테스트 환경: 가짜 임베딩과 메모리 안의 Qdrant."""

from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient, models

from rag.config import Config

REPO_ROOT = Path(__file__).resolve().parent.parent
KEYWORDS = ("alpha", "beta", "gamma")


class KeywordEmbeddings(Embeddings):
    """키워드 등장 횟수를 벡터로 쓰는 결정적인 가짜 임베딩.

    같은 키워드가 든 질문과 청크가 가장 가까워져, 검색 결과를 예상할 수 있다.
    받은 질문은 `queries`에 남겨 질문 형식을 확인할 수 있게 한다.
    """

    def __init__(self) -> None:
        self.queries: list[str] = []

    def _vector(self, text: str) -> list[float]:
        # 키워드가 없어도 영벡터가 되지 않게 작은 값을 더한다. Cosine 거리는 영벡터에서 정의되지 않는다.
        return [text.lower().count(word) + 0.01 for word in KEYWORDS]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return self._vector(text)


def make_config(**overrides) -> Config:
    """테스트용 config를 만든다. 최상위 항목을 통째로 바꿀 수 있다."""
    raw = {
        "index": {"url": "http://127.0.0.1:6333"},
        "chunking": {"size": 512, "overlap": 64},
        "embedding": {
            "base_url": "http://127.0.0.1:8003/v1",
            "model": "test/model",
            "revision": "abc",
        },
        "eval": {
            "fields": {"id": "id", "question": "question", "doc_ids": "gold", "type": "type"},
            "k": [1, 2],
        },
        "retrieve": [{"search": {"mode": "dense", "top_k": 2}}],
    }
    raw.update(overrides)
    return Config.model_validate(raw)


def add_index(client: QdrantClient, alias: str, chunks: list[tuple[str, str]]) -> None:
    """청크 `(문서 ID, 내용)`들을 담은 컬렉션을 만들고 `alias` 별칭을 붙인다.

    점 ID와 payload 형태는 실제 색인 형식(docs/design.md의 "색인 형식")을 따른다.
    """
    collection = "data-test"
    client.create_collection(
        collection,
        vectors_config=models.VectorParams(size=len(KEYWORDS), distance=models.Distance.COSINE),
    )
    vectors = KeywordEmbeddings().embed_documents([text for _, text in chunks])
    client.upsert(
        collection,
        points=[
            models.PointStruct(
                id=i,
                vector=vector,
                payload={"page_content": text, "metadata": {"doc_id": doc_id, "chunk": 0}},
            )
            for i, ((doc_id, text), vector) in enumerate(zip(chunks, vectors, strict=True))
        ],
    )
    client.update_collection_aliases(
        change_aliases_operations=[
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(collection_name=collection, alias_name=alias)
            )
        ]
    )


@pytest.fixture
def qdrant() -> QdrantClient:
    """테스트마다 비어 있는 메모리 안의 Qdrant."""
    return QdrantClient(":memory:")


@pytest.fixture
def embeddings() -> KeywordEmbeddings:
    return KeywordEmbeddings()
