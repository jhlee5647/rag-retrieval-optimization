"""rerank 노드: 앞 노드가 가져온 청크를 리랭커 점수로 다시 줄 세워 위에서부터 남긴다."""

from collections.abc import Callable
from typing import Any

import httpx
from pydantic import PositiveInt

from rag.config import Rerank, Strict
from rag.pipeline import Resources, State, node

# 청크 100개에 1초가 걸리지 않지만(장비에서 측정), 여러 문항의 요청이 한꺼번에 몰릴 때를 위해 넉넉히 둔다.
TIMEOUT_SECONDS = 120


class RerankParams(Strict):
    """config에 적는 rerank 노드의 인자."""

    top_k: PositiveInt


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

    Args:
        params: config에 적은 인자. `RerankParams`로 검사한다.
        resources: 리랭커. `make_reranker`가 만든 함수다.

    Returns:
        State를 받아 줄어든 `contexts`와 `doc_ids`를 돌려주는 함수.

    Raises:
        pydantic.ValidationError: 인자가 정의되지 않았거나 값이 틀린 경우.
    """
    p = RerankParams.model_validate(params)

    def run(state: State) -> dict[str, Any]:
        scores = resources.reranker(state["question"], state["contexts"])
        # sorted는 순서를 지키는 정렬이라, 점수가 같은 청크는 앞 노드의 순서대로 남는다.
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[: p.top_k]
        return {
            "contexts": [state["contexts"][i] for i in order],
            "doc_ids": [state["doc_ids"][i] for i in order],
        }

    return run
