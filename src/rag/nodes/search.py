"""search 노드: 색인에서 청크를 가져온다. 임베딩 검색, BM25 검색, 혼합 검색, 여러 검색어로 찾는 검색이 있다."""

from typing import Any, Literal, Self

from pydantic import PositiveInt, model_validator
from qdrant_client import models

from rag.config import Strict
from rag.index import SPARSE, bm25_query
from rag.pipeline import Resources, State, node

# 혼합 검색에서 두 검색의 순위를 합치는 식(RRF)의 상수. 한 검색에서 n위인 청크가 1 / (RRF_K + n - 1)점을
# 받고, 두 검색의 점수를 더한다. Qdrant의 기본값 2는 1위에 점수가 몰려, RRF를 제안한 논문의 값을 쓴다.
RRF_K = 60


class SearchParams(Strict):
    """config에 적는 search 노드의 인자."""

    mode: Literal["dense", "bm25", "hybrid", "multi"]
    # multi에서는 적지 않는다. multi는 청크 수가 아니라 문서 수(docs)로 자른다.
    top_k: PositiveInt | None = None
    # hybrid와 multi에서만 적는다. 임베딩 검색과 BM25 검색이 한 번에 각각 가져오는 후보 수다.
    candidates: PositiveInt | None = None
    # multi에서만 적는다. 순위를 합친 뒤 남길 문서 수다.
    docs: PositiveInt | None = None

    @model_validator(mode="after")
    def _fields_fit_mode(self) -> Self:
        multi = self.mode == "multi"
        if (self.mode == "hybrid" or multi) != (self.candidates is not None):
            raise ValueError(
                "candidates는 mode가 hybrid나 multi일 때 적고, 다른 mode에서는 적지 않는다"
            )
        if multi != (self.docs is not None) or multi != (self.top_k is None):
            raise ValueError(
                "mode가 multi면 top_k 없이 docs를 적고, 다른 mode에서는 docs 없이 top_k를 적는다"
            )
        return self


def fuse_documents(lists: list[list[str]], limit: int) -> list[str]:
    """여러 검색 목록의 문서 순위를 RRF로 합쳐, 점수가 높은 문서 `limit`개의 ID를 돌려준다.

    목록에서 문서의 순위는 그 문서의 청크가 처음 나온 자리다. 한 문서의 청크가 한 목록에 여러 개
    걸려도 한 번만 세고, 뒤의 문서를 밀어내지 않는다.

    Args:
        lists: 검색 목록들. 목록마다 걸린 청크의 문서 ID를 순위대로 적는다.
        limit: 돌려줄 문서 수.

    Returns:
        합친 점수가 높은 순서의 문서 ID. 점수가 같으면 먼저 나온 문서가 앞이다.
    """
    scores: dict[str, float] = {}
    for doc_ids in lists:
        for rank, doc_id in enumerate(dict.fromkeys(doc_ids)):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1 / (RRF_K + rank)
    return sorted(scores, key=lambda doc_id: -scores[doc_id])[:limit]


@node("search")
def search(params: dict[str, Any], resources: Resources):
    """찾은 청크로 `contexts`와 `doc_ids`를 채우는 노드를 만든다. `multi`가 아니면 청크 `top_k`개다.

    `mode`가 검색 방식을 정한다.

    - `dense`: 질문을 임베딩해 가장 가까운 청크를 가져온다.
    - `bm25`: 질문의 단어가 든 청크를 색인의 BM25 벡터로 찾는다. 임베딩 모델을 부르지 않는다.
    - `hybrid`: 두 검색이 각각 `candidates`개를 가져오고, 두 순위를 합쳐(RRF) 위에서부터 고른다.
    - `multi`: 질문, 그리고 expand 노드가 만든 검색어와 가상의 글마다 두 검색이 각각 `candidates`개를
      가져온다. 모든 목록의 순위를 문서 단위로 합쳐(`fuse_documents`) 상위 `docs`개 문서를 고르고, 그
      문서의 청크 가운데 검색에 걸린 것을 모두 넘긴다. 청크 수는 문항마다 다르다.

    Args:
        params: config에 적은 인자. `SearchParams`로 검사한다.
        resources: 벡터 저장소와 config. config에서 질문 지시문과 정확 검색 여부를 읽는다.

    Returns:
        State를 받아 `contexts`와 `doc_ids`를 돌려주는 함수.

    Raises:
        pydantic.ValidationError: 인자가 정의되지 않았거나 값이 틀린 경우.
    """
    p = SearchParams.model_validate(params)
    instruction = resources.config.embedding.query_instruction
    search_params = models.SearchParams(exact=resources.config.index.exact)
    store = resources.vectorstore

    def run(state: State) -> dict[str, Any]:
        # 지시문은 질문에만 붙인다. 문서 청크는 지시문 없이 임베딩돼 있다.
        query = f"Instruct: {instruction}\nQuery:{state['question']}"
        if p.mode == "dense":
            docs = store.similarity_search(query, k=p.top_k, search_params=search_params)
            return {
                "contexts": [doc.page_content for doc in docs],
                "doc_ids": [doc.metadata["doc_id"] for doc in docs],
            }
        # BM25 검색은 단어 색인을 그대로 읽는 정확한 검색이라 정확 검색 인자를 넘기지 않는다.
        sparse = bm25_query(state["question"])
        if p.mode == "bm25":
            request = {"query": sparse, "using": SPARSE}
        else:
            dense = store.embeddings.embed_query(query)
            request = {
                "prefetch": [
                    models.Prefetch(query=dense, limit=p.candidates, params=search_params),
                    models.Prefetch(query=sparse, using=SPARSE, limit=p.candidates),
                ],
                "query": models.RrfQuery(rrf=models.Rrf(k=RRF_K)),
            }
        points = store.client.query_points(
            store.collection_name, limit=p.top_k, with_payload=True, **request
        ).points
        return {
            "contexts": [point.payload["page_content"] for point in points],
            "doc_ids": [point.payload["metadata"]["doc_id"] for point in points],
        }

    def run_multi(state: State) -> dict[str, Any]:
        question = state["question"]
        # (임베딩할 글, BM25로 찾을 글)의 쌍이다. 검색어는 질문처럼 지시문을 붙여 임베딩한다. 가상의 글은
        # 문서처럼 지시문 없이 임베딩하고, BM25에는 질문의 단어를 잃지 않게 질문과 함께 넣는다.
        pairs = [
            (f"Instruct: {instruction}\nQuery:{query}", query)
            for query in [question, *state["queries"]]
        ] + [(passage, f"{question} {passage}") for passage in state["passages"]]
        lists: list[list[str]] = []
        # 문서마다 걸린 청크를 처음 걸린 순서로 모은다. 같은 청크가 여러 목록에 걸려도 하나다.
        found: dict[str, dict[Any, str]] = {}
        for dense_text, bm25_text in pairs:
            for request in (
                {"query": store.embeddings.embed_query(dense_text), "search_params": search_params},
                {"query": bm25_query(bm25_text), "using": SPARSE},
            ):
                points = store.client.query_points(
                    store.collection_name, limit=p.candidates, with_payload=True, **request
                ).points
                lists.append([point.payload["metadata"]["doc_id"] for point in points])
                for point, doc_id in zip(points, lists[-1], strict=True):
                    found.setdefault(doc_id, {})[point.id] = point.payload["page_content"]
        top = fuse_documents(lists, p.docs)
        return {
            "contexts": [text for doc_id in top for text in found[doc_id].values()],
            "doc_ids": [doc_id for doc_id in top for _ in found[doc_id]],
        }

    return run_multi if p.mode == "multi" else run
