"""select 노드: 앞 노드가 추린 청크를 생성 모델이 읽고, 다음 단계에 넘길 것을 고른다."""

import json
from pathlib import Path
from typing import Any

from pydantic import PositiveInt

from rag.config import Strict
from rag.pipeline import Resources, State, node

# 생성 모델은 고른 번호의 배열만 낸다. 번호 몇 개와 구분자가 들어가고 남는 길이다.
MAX_TOKENS = 80


class SelectParams(Strict):
    """config에 적는 select 노드의 인자."""

    # 저장소 루트 기준 경로. 틀에는 {question}, {documents}, {top_k}가 있어야 한다.
    prompt: Path
    # 다음 단계에 넘길 청크 수이자, 생성 모델이 고를 수 있는 최대 수다.
    top_k: PositiveInt


@node("select")
def select(params: dict[str, Any], resources: Resources):
    """앞 노드의 청크 가운데 생성 모델이 고른 것을 앞에 둔 `top_k`개만 남기는 노드를 만든다.

    청크마다 `[1]`부터 번호를 붙여 틀의 `{documents}`에 넣고, 고른 번호를 정수 배열로 받는다.
    고른 청크를 생성 모델이 적은 순서대로 앞에 두고, 남는 자리는 앞 노드의 순서대로 채운다.

    청크 하나를 문서 하나로 보여 주므로, 문서마다 청크 하나만 오도록 rerank의 `dedup`과 함께 쓴다.

    Args:
        params: config에 적은 인자. `SelectParams`로 검사한다.
        resources: 생성 모델.

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
        ]
        order = (picks + [i for i in range(len(contexts)) if i not in picks])[: p.top_k]
        return {
            "contexts": [contexts[i] for i in order],
            "doc_ids": [doc_ids[i] for i in order],
        }

    return run
