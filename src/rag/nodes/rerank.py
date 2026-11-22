"""rerank 노드: 앞 노드가 가져온 청크를 리랭커 점수로 다시 줄 세워 위에서부터 남긴다."""

from collections.abc import Callable
from typing import Any

import httpx
from pydantic import PositiveInt

from rag.config import Rerank, Strict
from rag.index import point_id
from rag.pipeline import Resources, State, node

# 청크 100개에 1초가 걸리지 않지만(장비에서 측정), 여러 문항의 요청이 한꺼번에 몰릴 때를 위해 넉넉히 둔다.
TIMEOUT_SECONDS = 120


class RerankParams(Strict):
    """config에 적는 rerank 노드의 인자."""

    top_k: PositiveInt
    # 청크를 리랭커에 넘길 때 그 문서의 첫 줄을 앞에 붙인다. 문서 중간의 청크에는 어느 문서인지 알려 주는
    # 제목 줄이 없기 때문이다.
    title: bool = False
    # 한 문서에서 점수가 가장 높은 청크 하나만 남긴다. 같은 문서의 청크가 여러 자리를 차지하지 않게 한다.
    dedup: bool = False


def make_reranker(
    rerank: Rerank, transport: httpx.BaseTransport | None = None
) -> Callable[[str, list[str]], list[float]]:
    """리랭커 서버(vLLM의 `/v1/rerank`)에서 청크마다 관련도 점수를 받는 함수를 만든다.

    Args:
        rerank: config의 `rerank` 절.
        transport: 테스트에서 가짜 서버를 끼울 때만 넘긴다.

    Returns:
        질문과 청크들을 받아, 청크 순서대로 점수를 돌려주는 함수. 서버가 요청을 거부하면
        `httpx.HTTPStatusError`를 낸다.
    """
    client = httpx.Client(timeout=TIMEOUT_SECONDS, transport=transport)

    def score(question: str, contexts: list[str]) -> list[float]:
        response = client.post(
            f"{rerank.base_url}/rerank",
            json={"model": rerank.model, "query": question, "documents": contexts},
        )
        response.raise_for_status()
        # 서버는 점수가 높은 순으로 돌려준다. 받은 청크의 순서로 되돌려 청크와 짝짓게 한다.
        results = sorted(response.json()["results"], key=lambda result: result["index"])
        return [result["relevance_score"] for result in results]

    return score


@node("rerank")
def rerank(params: dict[str, Any], resources: Resources):
    """앞 노드의 청크 가운데 리랭커 점수가 높은 `top_k`개만 남기는 노드를 만든다.

    순서는 리랭커 점수만으로 정한다. 검색에서의 순위는 점수가 같을 때만 따른다.

    - `title`을 켜면 리랭커에는 "문서의 첫 줄 + 청크"를 넘긴다. 돌려주는 `contexts`는 원래 청크 그대로다.
    - `dedup`을 켜면 한 문서에서 점수가 가장 높은 청크만 남긴 뒤에 `top_k`개를 고른다.

    Args:
        params: config에 적은 인자. `RerankParams`로 검사한다.
        resources: 리랭커. `make_reranker`가 만든 함수다. `title`을 켜면 벡터 저장소에서 문서의
            첫 청크도 읽는다.

    Returns:
        State를 받아 줄어든 `contexts`와 `doc_ids`를 돌려주는 함수.

    Raises:
        pydantic.ValidationError: 인자가 정의되지 않았거나 값이 틀린 경우.
    """
    p = RerankParams.model_validate(params)
    store = resources.vectorstore

    def first_lines(doc_ids: list[str]) -> dict[str, str]:
        """문서마다 첫 청크의 첫 줄을 색인에서 가져온다."""
        ids = {point_id(doc_id, 0): doc_id for doc_id in set(doc_ids)}
        points = store.client.retrieve(store.collection_name, ids=list(ids), with_payload=True)
        return {
            ids[point.id]: point.payload["page_content"].lstrip().split("\n", 1)[0].strip()
            for point in points
        }

    def run(state: State) -> dict[str, Any]:
        contexts, doc_ids = state["contexts"], state["doc_ids"]
        inputs = contexts
        if p.title:
            line = first_lines(doc_ids)
            # 문서의 첫 청크는 이미 그 줄로 시작하므로 다시 붙이지 않는다.
            inputs = [
                text if text.lstrip().startswith(line[doc_id]) else f"{line[doc_id]}\n\n{text}"
                for text, doc_id in zip(contexts, doc_ids, strict=True)
            ]
        scores = resources.reranker(state["question"], inputs)
        # sorted는 순서를 지키는 정렬이라, 점수가 같은 청크는 앞 노드의 순서대로 남는다.
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        if p.dedup:
            best: dict[str, int] = {}
            for i in order:
                best.setdefault(doc_ids[i], i)
            order = list(best.values())
        order = order[: p.top_k]
        return {
            "contexts": [contexts[i] for i in order],
            "doc_ids": [doc_ids[i] for i in order],
        }

    return run
