"""answer 노드: 프롬프트 틀에 문맥과 질문을 채워 생성 모델을 부른다."""

from pathlib import Path
from typing import Any

from rag.config import Strict
from rag.pipeline import Resources, State, node

CONTEXT_SEPARATOR = "\n\n---\n\n"


class AnswerParams(Strict):
    """config에 적는 answer 노드의 인자."""

    # 저장소 루트 기준 경로. 틀에는 {context}와 {question}이 있어야 한다.
    prompt: Path


@node("answer")
def answer(params: dict[str, Any], resources: Resources):
    """문맥과 질문으로 `answer`를 채우는 노드를 만든다.

    Args:
        params: config에 적은 인자. `AnswerParams`로 검사한다.
        resources: 생성 모델.

    Returns:
        State를 받아 `answer`를 돌려주는 함수.

    Raises:
        pydantic.ValidationError: 인자가 정의되지 않았거나 값이 틀린 경우.
        ValueError: 틀에 `{context}`나 `{question}`이 없는 경우.
    """
    p = AnswerParams.model_validate(params)
    template = p.prompt.read_text(encoding="utf-8")
    missing = [f for f in ("{context}", "{question}") if f not in template]
    if missing:
        raise ValueError(f"프롬프트 틀 {p.prompt}에 {', '.join(missing)}이 없다")

    def run(state: State) -> dict[str, Any]:
        context = CONTEXT_SEPARATOR.join(state["contexts"])
        message = resources.llm.invoke(template.format(context=context, question=state["question"]))
        return {"answer": message.content}

    return run
