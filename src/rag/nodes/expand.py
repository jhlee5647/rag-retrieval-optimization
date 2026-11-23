"""expand 노드: 생성 모델이 질문에서 검색어와 가상의 글을 만들어, 뒤의 search 노드가 그것으로도 찾게 한다."""

import json
from pathlib import Path
from typing import Any

from rag.config import Strict
from rag.pipeline import Resources, State, node

# 검색어 배열은 잘리면 읽을 수 없다. 틀 하나의 검색어 몇 개가 들어가고 남는 길이로 둔다.
QUERIES_MAX_TOKENS = 300
# 가상의 글은 검색에만 쓰므로 끝이 잘려도 된다. 몇 문장이 들어가는 길이다.
PASSAGE_MAX_TOKENS = 220


class ExpandParams(Strict):
    """config에 적는 expand 노드의 인자."""

    # 저장소 루트 기준 경로들. 틀마다 한 번 물어 검색어 여러 개를 받는다. 틀에는 {question}이 있어야 한다.
    queries: list[Path]
    # 가상의 글을 받는 틀. {question}과 {style}이 있어야 한다.
    passage: Path
    # 가상의 글을 어떤 문서처럼 쓸지 적은 말들. 하나마다 한 번 물어 글 하나를 받는다.
    styles: list[str]


def _read_template(path: Path, fields: tuple[str, ...]) -> str:
    template = path.read_text(encoding="utf-8")
    missing = [field for field in fields if field not in template]
    if missing:
        raise ValueError(f"프롬프트 틀 {path}에 {', '.join(missing)}이 없다")
    return template


@node("expand")
def expand(params: dict[str, Any], resources: Resources):
    """질문에서 만든 검색어와 가상의 글로 `queries`와 `passages`를 채우는 노드를 만든다.

    search 노드의 `multi` mode가 질문과 함께 이 둘로도 검색한다.

    - 검색어: `queries`의 틀마다 생성 모델에 한 번 묻고, 문자열 배열로 받은 검색어를 틀 순서대로 모은다.
    - 가상의 글: `styles`의 문서 종류마다 한 번 묻고, 그 종류의 문서처럼 쓴 짧은 글을 받는다(HyDE).
      질문보다 문서에 가까운 말로 쓰여, 질문과 다른 말로 쓰인 문서를 찾는 데 쓴다.

    Args:
        params: config에 적은 인자. `ExpandParams`로 검사한다.
        resources: 생성 모델.

    Returns:
        State를 받아 `queries`와 `passages`를 돌려주는 함수.

    Raises:
        pydantic.ValidationError: 인자가 정의되지 않았거나 값이 틀린 경우.
        ValueError: 틀에 `{question}`이 없거나, 가상의 글 틀에 `{style}`이 없는 경우.
    """
    p = ExpandParams.model_validate(params)
    query_templates = [_read_template(path, ("{question}",)) for path in p.queries]
    passage_template = _read_template(p.passage, ("{question}", "{style}"))

    def run(state: State) -> dict[str, Any]:
        question = state["question"]
        # 출력을 문자열 배열로 제한한다. 배열 밖의 글자가 섞이면 어디까지가 검색어인지 가려 읽어야 한다.
        query_llm = resources.llm.bind(
            max_tokens=QUERIES_MAX_TOKENS,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "queries",
                    "schema": {"type": "array", "items": {"type": "string"}},
                },
            },
        )
        passage_llm = resources.llm.bind(max_tokens=PASSAGE_MAX_TOKENS)
        return {
            "queries": [
                query
                for template in query_templates
                for query in json.loads(
                    query_llm.invoke(template.format(question=question)).content
                )
            ],
            "passages": [
                passage_llm.invoke(passage_template.format(question=question, style=style)).content
                for style in p.styles
            ],
        }

    return run
