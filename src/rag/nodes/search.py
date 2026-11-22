"""search 노드: 질문으로 색인에서 청크를 가져온다. 임베딩 검색, BM25 검색, 둘을 합친 혼합 검색이 있다."""

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

    mode: Literal["dense", "bm25", "hybrid"]
    top_k: PositiveInt
    # hybrid에서만 적는다. 임베딩 검색과 BM25 검색이 각각 가져오는 후보 수다.
    candidates: PositiveInt | None = None

    @model_validator(mode="after")
    def _candidates_only_for_hybrid(self) -> Self:
        if (self.mode == "hybrid") != (self.candidates is not None):
            raise ValueError("candidates는 mode가 hybrid일 때 적고, 다른 mode에서는 적지 않는다")
        return self


@node("search")
def search(params: dict[str, Any], resources: Resources):
    """청크 `top_k`개로 `contexts`와 `doc_ids`를 채우는 노드를 만든다.

    `mode`가 검색 방식을 정한다.

    - `dense`: 질문을 임베딩해 가장 가까운 청크를 가져온다.
    - `bm25`: 질문의 단어가 든 청크를 색인의 BM25 벡터로 찾는다. 임베딩 모델을 부르지 않는다.
    - `hybrid`: 두 검색이 각각 `candidates`개를 가져오고, 두 순위를 합쳐(RRF) 위에서부터 고른다.

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

    return run
