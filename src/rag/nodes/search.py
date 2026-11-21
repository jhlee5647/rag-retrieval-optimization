"""search 노드: 질문을 임베딩해 색인에서 가장 가까운 청크를 가져온다."""

from typing import Any, Literal

from pydantic import PositiveInt
from qdrant_client import models

from rag.config import Strict
from rag.pipeline import Resources, State, node


class SearchParams(Strict):
    """config에 적는 search 노드의 인자."""

    mode: Literal["dense"]
    top_k: PositiveInt


@node("search")
def search(params: dict[str, Any], resources: Resources):
    """가장 가까운 청크 `top_k`개로 `contexts`와 `doc_ids`를 채우는 노드를 만든다.

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

    def run(state: State) -> dict[str, Any]:
        # 지시문은 질문에만 붙인다. 문서 청크는 지시문 없이 임베딩돼 있다.
        query = f"Instruct: {instruction}\nQuery:{state['question']}"
        docs = resources.vectorstore.similarity_search(
            query, k=p.top_k, search_params=search_params
        )
        return {
            "contexts": [doc.page_content for doc in docs],
            "doc_ids": [doc.metadata["doc_id"] for doc in docs],
        }

    return run
