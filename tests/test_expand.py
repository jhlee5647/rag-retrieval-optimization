import pytest
from conftest import REPO_ROOT, make_config
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from rag.config import load_config
from rag.nodes.expand import PASSAGE_MAX_TOKENS, QUERIES_MAX_TOKENS, expand
from rag.pipeline import Resources, build_graph

# 프롬프트에 왼쪽 말이 들어 있으면 오른쪽 답을 돌려준다.
ANSWERS = {
    "facets": '["first query", "second query"]',
    "paraphrase": '["third query"]',
    "chat": "chat passage",
    "email": "email passage",
}
QUERY_TEMPLATES = ("facets: {question}", "paraphrase: {question}")
PASSAGE_TEMPLATE = "{style} about {question}"


class ScriptedLLM:
    """프롬프트에 든 말에 따라 정해 둔 답을 돌려주는 가짜 생성 모델. 호출마다 bind 인자와 프롬프트를 남긴다."""

    def __init__(self, calls: list | None = None, bound: dict | None = None) -> None:
        self.calls: list[tuple[dict, str]] = [] if calls is None else calls
        self.bound = bound or {}

    def bind(self, **kwargs):
        return ScriptedLLM(self.calls, kwargs)

    def invoke(self, prompt: str) -> AIMessage:
        self.calls.append((self.bound, prompt))
        return AIMessage(content=next(text for key, text in ANSWERS.items() if key in prompt))


def make_node(tmp_path, llm, queries=QUERY_TEMPLATES, passage=PASSAGE_TEMPLATE):
    paths = []
    for i, template in enumerate(queries):
        paths.append(tmp_path / f"queries-{i}.txt")
        paths[-1].write_text(template, encoding="utf-8")
    passage_path = tmp_path / "passage.txt"
    passage_path.write_text(passage, encoding="utf-8")
    params = {
        "queries": [str(path) for path in paths],
        "passage": str(passage_path),
        "styles": ["chat", "email"],
    }
    return expand(params, Resources(config=make_config(), llm=llm))


def test_collects_queries_of_every_template_and_one_passage_per_style(tmp_path):
    """검색어는 틀마다 받은 것을 틀 순서대로 모으고, 가상의 글은 문서 종류마다 하나씩 받는다."""
    # Arrange
    run = make_node(tmp_path, ScriptedLLM())

    # Act
    result = run({"question": "what?"})

    # Assert
    assert result == {
        "queries": ["first query", "second query", "third query"],
        "passages": ["chat passage", "email passage"],
    }


def test_fills_question_and_style_into_templates(tmp_path):
    """검색어 틀에는 질문을, 가상의 글 틀에는 질문과 문서 종류를 채워 생성 모델에 보낸다."""
    # Arrange
    llm = ScriptedLLM()
    run = make_node(tmp_path, llm)

    # Act
    run({"question": "what?"})

    # Assert
    assert [prompt for _, prompt in llm.calls] == [
        "facets: what?",
        "paraphrase: what?",
        "chat about what?",
        "email about what?",
    ]


def test_restricts_queries_to_string_array_and_passages_only_in_length(tmp_path):
    """검색어를 물을 때만 출력을 문자열 배열로 제한해 달라고 요청한다. 토큰 한도는 둘이 다르다."""
    # Arrange
    llm = ScriptedLLM()
    run = make_node(tmp_path, llm)

    # Act
    run({"question": "what?"})

    # Assert
    bound = [kwargs for kwargs, _ in llm.calls]
    for kwargs in bound[:2]:
        assert kwargs["max_tokens"] == QUERIES_MAX_TOKENS
        assert kwargs["response_format"]["json_schema"]["schema"] == {
            "type": "array",
            "items": {"type": "string"},
        }
    assert bound[2:] == [{"max_tokens": PASSAGE_MAX_TOKENS}] * 2


@pytest.mark.parametrize(
    "queries, passage, missing",
    [
        (("facets", "paraphrase: {question}"), PASSAGE_TEMPLATE, "{question}"),
        (QUERY_TEMPLATES, "{style}", "{question}"),
        (QUERY_TEMPLATES, "about {question}", "{style}"),
    ],
    ids=["query-without-question", "passage-without-question", "passage-without-style"],
)
def test_rejects_template_without_placeholder(tmp_path, queries, passage, missing):
    """틀에 채울 자리가 하나라도 없으면 모델을 부르기 전에 빠진 자리를 대며 거부한다."""
    # Arrange
    llm = ScriptedLLM()

    # Act / Assert
    with pytest.raises(ValueError, match=missing):
        make_node(tmp_path, llm, queries=queries, passage=passage)
    assert llm.calls == []


@pytest.mark.parametrize(
    "params",
    [
        {"queries": ["q.txt"], "passage": "p.txt"},
        {"queries": "q.txt", "passage": "p.txt", "styles": ["chat"]},
        {"queries": ["q.txt"], "passages": "p.txt", "styles": ["chat"]},
    ],
    ids=["no-styles", "queries-not-a-list", "misspelled"],
)
def test_rejects_invalid_params(params):
    """인자가 빠졌거나 형태가 틀렸거나 정의되지 않은 인자가 있으면 틀을 읽기 전에 거부한다."""
    with pytest.raises(ValidationError):
        expand(params, Resources(config=make_config(), llm=ScriptedLLM()))


def test_repo_expand_config_builds_graph(monkeypatch):
    """저장소의 r4-expand config와 prompts/expand-*.txt로 검색 그래프가 조립된다.

    틀에 노드가 채우는 자리가 모두 있는지와 노드 인자가 맞는지를 함께 확인한다.
    """
    # Arrange: 프롬프트 경로가 저장소 루트 기준이다
    monkeypatch.chdir(REPO_ROOT)
    config = load_config(REPO_ROOT / "configs" / "exp" / "r4-expand.yaml")

    # Act
    graph = build_graph(config.retrieve, Resources(config=config))

    # Assert
    assert [name for name in graph.get_graph().nodes if name[0].isdigit()] == [
        "0-expand",
        "1-search",
        "2-rerank",
    ]
