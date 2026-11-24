"""select 노드: 앞 노드가 추린 청크를 생성 모델이 읽고, 다음 단계에 넘길 것을 고른다."""

import json
from pathlib import Path
from typing import Any

from pydantic import PositiveInt

from rag.config import Strict
from rag.index import point_id
from rag.pipeline import Resources, State, node

# 생성 모델은 고른 번호의 배열만 낸다. 번호 몇 개와 구분자가 들어가고 남는 길이다.
MAX_TOKENS = 80


class SelectParams(Strict):
    """config에 적는 select 노드의 인자."""

    # 저장소 루트 기준 경로. 틀에는 {question}, {documents}, {top_k}가 있어야 한다.
    prompt: Path
    # 다음 단계에 넘길 청크 수이자, 생성 모델이 고를 수 있는 최대 수다.
    top_k: PositiveInt
    # 주면 고른 문서의 청크를 문서마다 앞에서부터 이 수만큼 색인에서 읽어, `top_k`개를 고른 문서의 청크로
    # 먼저 채운다. 답이 고른 문서의 다른 청크에 있을 때 그 청크도 넘어가게 한다.
    doc_chunks: PositiveInt | None = None


@node("select")
def select(params: dict[str, Any], resources: Resources):
    """앞 노드의 청크 가운데 생성 모델이 고른 것을 앞에 둔 `top_k`개만 남기는 노드를 만든다.

    청크마다 `[1]`부터 번호를 붙여 틀의 `{documents}`에 넣고, 고른 번호를 정수 배열로 받는다.
    고른 청크를 생성 모델이 적은 순서대로 앞에 두고, 남는 자리는 앞 노드의 순서대로 채운다.

    청크 하나를 문서 하나로 보여 주므로, 문서마다 청크 하나만 오도록 rerank의 `dedup`과 함께 쓴다.

    `doc_chunks`를 주면 고른 청크 대신 그 문서의 청크들을 넘긴다. 문서마다 앞에서부터 `doc_chunks`개를
    색인에서 읽어 리랭커로 질문과의 관련도를 매긴다. 고른 문서마다 점수가 가장 높은 청크를 하나씩 넣고,
    남는 자리는 고른 문서의 다른 청크를 점수순으로 넣는다. 문서는 생성 모델이 적은 순서대로, 한 문서의
    청크는 문서에 나온 순서대로 놓는다. 그래도 남는 자리는 고르지 않은 청크를 앞 노드의 순서대로 채운다.

    Args:
        params: config에 적은 인자. `SelectParams`로 검사한다.
        resources: 생성 모델. `doc_chunks`를 주면 벡터 저장소와 리랭커도 쓴다.

    Returns:
        State를 받아 줄어든 `contexts`와 `doc_ids`를 돌려주는 함수.

    Raises:
        pydantic.ValidationError: 인자가 정의되지 않았거나 값이 틀린 경우.
        ValueError: 틀에 `{question}`, `{documents}`, `{top_k}` 중 없는 것이 있는 경우.
    """
    p = SelectParams.model_validate(params)
    template = p.prompt.read_text(encoding="utf-8")
    missing = [f for f in ("{question}", "{documents}", "{top_k}") if f not in template]
    if missing:
        raise ValueError(f"프롬프트 틀 {p.prompt}에 {', '.join(missing)}이 없다")
    store = resources.vectorstore

    def document_chunks(question: str, picked: list[str]) -> tuple[list[str], list[str]]:
        """고른 문서들의 청크에서 `top_k`개까지 골라, 넘길 순서대로 글과 문서 ID를 돌려준다."""
        # 청크를 (고른 순서, 문서 안의 청크 번호)로 가리킨다. 문서의 앞 `doc_chunks`개만 보므로, 앞 노드가
        # 넘긴 청크가 그보다 뒤에 있으면 그 청크는 넘어가지 않는다.
        ids = {
            point_id(doc_id, n): (rank, n)
            for rank, doc_id in enumerate(picked)
            for n in range(p.doc_chunks)
        }
        points = store.client.retrieve(store.collection_name, ids=list(ids), with_payload=True)
        text = {ids[point.id]: point.payload["page_content"] for point in points}
        keys = list(text)
        scores = dict(
            zip(keys, resources.reranker(question, [text[key] for key in keys]), strict=True)
        )
        by_score = sorted(keys, key=lambda key: (-scores[key], key))
        best: dict[int, tuple[int, int]] = {}
        for key in by_score:
            best.setdefault(key[0], key)
        # 고른 문서가 하나도 빠지지 않게 문서마다 가장 높은 청크를 먼저 넣고, 나머지를 점수순으로 잇는다.
        take = [*best.values(), *(key for key in by_score if best[key[0]] != key)][: p.top_k]
        # 생성 모델이 읽기 쉽게, 한 문서의 청크를 모아 문서에 나온 순서대로 놓는다.
        take.sort()
        return [text[key] for key in take], [picked[rank] for rank, _ in take]

    def run(state: State) -> dict[str, Any]:
        contexts, doc_ids = state["contexts"], state["doc_ids"]
        documents = "\n\n".join(f"[{i}]\n{text}" for i, text in enumerate(contexts, start=1))
        # 출력을 정수 배열로 제한한다. 제한하지 않으면 고르지 않고 질문에 답하기 시작하는 일이 있다.
        llm = resources.llm.bind(
            max_tokens=MAX_TOKENS,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "picks",
                    "schema": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 1, "maximum": len(contexts)},
                        "maxItems": p.top_k,
                    },
                },
            },
        )
        message = llm.invoke(
            template.format(question=state["question"], documents=documents, top_k=p.top_k)
        )
        # 같은 번호를 두 번 적거나 범위 밖의 번호를 적는 것까지 서버가 막아 준다고 보지 않는다.
        picks = [
            i - 1 for i in dict.fromkeys(json.loads(message.content)) if 1 <= i <= len(contexts)
        ][: p.top_k]
        texts, docs = [contexts[i] for i in picks], [doc_ids[i] for i in picks]
        if p.doc_chunks and picks:
            texts, docs = document_chunks(state["question"], docs)
        fill = [i for i in range(len(contexts)) if i not in picks][: p.top_k - len(texts)]
        return {
            "contexts": texts + [contexts[i] for i in fill],
            "doc_ids": docs + [doc_ids[i] for i in fill],
        }

    return run
