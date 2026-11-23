import pytest
from conftest import REPO_ROOT, make_config
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from rag.config import load_config
from rag.nodes.select import MAX_TOKENS, select
from rag.pipeline import Resources, build_graph

STATE = {
    "question": "which one?",
    "contexts": ["first", "second", "third"],
    "doc_ids": ["doc-a", "doc-b", "doc-c"],
}
TEMPLATE = "Q: {question}\n{documents}\nPick at most {top_k}."


class PickingLLM:
    """정해 둔 답을 돌려주는 가짜 생성 모델. 받은 프롬프트와 bind 인자를 남긴다."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.prompts: list[str] = []
        self.bound: dict = {}

    def bind(self, **kwargs):
        self.bound = kwargs
        return self

    def invoke(self, prompt: str) -> AIMessage:
        self.prompts.append(prompt)
        return AIMessage(content=self.content)


def make_node(tmp_path, llm, top_k, template=TEMPLATE):
    prompt = tmp_path / "select.txt"
    prompt.write_text(template, encoding="utf-8")
    return select({"prompt": str(prompt), "top_k": top_k}, Resources(config=make_config(), llm=llm))


def test_puts_picked_chunks_first_in_the_order_the_model_listed(tmp_path):
    """생성 모델이 고른 청크를 적은 순서대로 넘기고, 문서 ID도 청크를 따라 옮긴다."""
    # Arrange
    run = make_node(tmp_path, PickingLLM("[3, 1]"), top_k=2)

    # Act
    result = run(STATE)

    # Assert
    assert result == {"contexts": ["third", "first"], "doc_ids": ["doc-c", "doc-a"]}


@pytest.mark.parametrize(
    "content, expected",
    [("[3]", ["doc-c", "doc-a"]), ("[]", ["doc-a", "doc-b"])],
    ids=["one-pick", "no-pick"],
)
def test_fills_remaining_slots_in_incoming_order(tmp_path, content, expected):
    """고른 수가 top_k보다 적으면 남는 자리를 앞 노드의 순서대로 채운다. 하나도 안 골라도 그렇다."""
    # Arrange
    run = make_node(tmp_path, PickingLLM(content), top_k=2)

    # Act
    result = run(STATE)

    # Assert
    assert result["doc_ids"] == expected


def test_ignores_repeated_and_out_of_range_numbers(tmp_path):
    """같은 번호를 두 번 적거나 없는 번호를 적어도, 있는 번호를 처음 나온 순서로 한 번씩만 쓴다."""
    # Arrange
    run = make_node(tmp_path, PickingLLM("[2, 2, 9, 0, 3]"), top_k=3)

    # Act
    result = run(STATE)

    # Assert
    assert result["doc_ids"] == ["doc-b", "doc-c", "doc-a"]


def test_prompt_numbers_chunks_from_one_and_carries_question_and_limit(tmp_path):
    """틀의 {documents}에 청크를 [1]부터 번호를 붙여 넣고, {question}과 {top_k}를 채운다."""
    # Arrange
    llm = PickingLLM("[1]")
    run = make_node(tmp_path, llm, top_k=2)

    # Act
    run(STATE)

    # Assert
    assert llm.prompts == [
        "Q: which one?\n[1]\nfirst\n\n[2]\nsecond\n\n[3]\nthird\nPick at most 2."
    ]


def test_restricts_output_to_integer_array_within_chunk_count(tmp_path):
    """출력을 1부터 청크 수까지의 정수 배열(최대 top_k개)로 제한해 달라고 서버에 요청한다."""
    # Arrange
    llm = PickingLLM("[1]")
    run = make_node(tmp_path, llm, top_k=2)

    # Act
    run(STATE)

    # Assert
    assert llm.bound["max_tokens"] == MAX_TOKENS
    assert llm.bound["response_format"]["json_schema"]["schema"] == {
        "type": "array",
        "items": {"type": "integer", "minimum": 1, "maximum": 3},
        "maxItems": 2,
    }


@pytest.mark.parametrize("missing", ["{question}", "{documents}", "{top_k}"])
def test_rejects_template_without_placeholder(tmp_path, missing):
    """틀에 채울 자리가 하나라도 없으면 모델을 부르기 전에 빠진 자리를 대며 거부한다."""
    # Arrange
    template = TEMPLATE.replace(missing, "")

    # Act / Assert
    with pytest.raises(ValueError, match=missing):
        make_node(tmp_path, PickingLLM("[1]"), top_k=2, template=template)


@pytest.mark.parametrize(
    "params",
    [{"prompt": "x.txt", "top_k": 0}, {"prompt": "x.txt"}, {"promt": "x.txt", "top_k": 5}],
    ids=["zero-top-k", "no-top-k", "misspelled"],
)
def test_rejects_invalid_params(params):
    """top_k가 1보다 작거나 빠졌거나, 정의되지 않은 인자가 있으면 틀을 읽기 전에 거부한다."""
    with pytest.raises(ValidationError):
        select(params, Resources(config=make_config(), llm=PickingLLM("[1]")))


def test_repo_select_config_builds_graph(monkeypatch):
    """저장소의 r4-select config와 prompts/select.txt로 검색 그래프가 조립된다.

    틀에 노드가 채우는 자리가 모두 있는지와 노드 인자가 맞는지를 함께 확인한다.
    """
    # Arrange: 프롬프트 경로가 저장소 루트 기준이다
    monkeypatch.chdir(REPO_ROOT)
    config = load_config(REPO_ROOT / "configs" / "exp" / "r4-select.yaml")

    # Act
    graph = build_graph(config.retrieve, Resources(config=config))

    # Assert
    assert [name for name in graph.get_graph().nodes if name[0].isdigit()] == [
        "0-search",
        "1-rerank",
        "2-select",
    ]
