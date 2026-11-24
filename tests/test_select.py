import pytest
from conftest import REPO_ROOT, make_config
from langchain_core.messages import AIMessage
from langchain_qdrant import QdrantVectorStore
from pydantic import ValidationError
from qdrant_client import models

from rag.config import load_config
from rag.index import point_id
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


# doc_chunks 테스트의 색인. 문서마다 청크의 글과, 가짜 리랭커가 그 청크에 주는 점수다.
DOCS = {
    "doc-a": [("a0", 0.2), ("a1", 0.5), ("a2", 0.9)],
    "doc-b": [("b0", 0.3), ("b1", 0.8)],
    "doc-c": [("c0", 0.1)],
}
# rerank의 dedup 뒤처럼 문서마다 청크 하나씩 온다.
CHUNK_STATE = {
    "question": "which one?",
    "contexts": ["a1", "b0", "c0"],
    "doc_ids": ["doc-a", "doc-b", "doc-c"],
}


def make_chunks_node(tmp_path, qdrant, embeddings, content, top_k, doc_chunks=3, calls=None):
    """DOCS를 색인에 넣고 doc_chunks를 준 노드를 만든다. 리랭커가 받은 질문과 청크는 `calls`에 남긴다."""
    qdrant.create_collection(
        "data-test", vectors_config=models.VectorParams(size=1, distance=models.Distance.COSINE)
    )
    qdrant.upsert(
        "data-test",
        points=[
            models.PointStruct(
                id=point_id(doc_id, n),
                vector=[1.0],
                payload={"page_content": text, "metadata": {"doc_id": doc_id, "chunk": n}},
            )
            for doc_id, chunks in DOCS.items()
            for n, (text, _) in enumerate(chunks)
        ],
    )
    store = QdrantVectorStore(
        client=qdrant,
        collection_name="data-test",
        embedding=embeddings,
        validate_collection_config=False,
    )
    scores = {text: score for chunks in DOCS.values() for text, score in chunks}

    def reranker(question, contexts):
        if calls is not None:
            calls.append((question, sorted(contexts)))
        return [scores[text] for text in contexts]

    prompt = tmp_path / "select.txt"
    prompt.write_text(TEMPLATE, encoding="utf-8")
    return select(
        {"prompt": str(prompt), "top_k": top_k, "doc_chunks": doc_chunks},
        Resources(
            config=make_config(), vectorstore=store, reranker=reranker, llm=PickingLLM(content)
        ),
    )


def test_doc_chunks_fills_slots_with_other_chunks_of_picked_documents(tmp_path, qdrant, embeddings):
    """doc_chunks를 주면 고른 문서의 청크를 색인에서 읽어, 남는 자리를 고른 문서의 다른 청크로 채운다.

    문서는 생성 모델이 적은 순서대로, 한 문서의 청크는 문서에 나온 순서대로 놓는다.
    """
    # Arrange: doc-b, doc-a 순으로 고른다. 문서마다 가장 높은 청크는 b1과 a2고, 남은 한 자리는
    # a1(0.5)이 b0(0.3)을 앞선다. 고르지 않은 doc-c는 색인에서 읽지 않는다
    calls = []
    run = make_chunks_node(tmp_path, qdrant, embeddings, "[2, 1]", top_k=3, calls=calls)

    # Act
    result = run(CHUNK_STATE)

    # Assert: 앞 노드가 넘긴 b0 대신 b1이 가고, 점수가 더 높은 a2가 a1 뒤에 온다
    assert result == {"contexts": ["b1", "a1", "a2"], "doc_ids": ["doc-b", "doc-a", "doc-a"]}
    assert calls == [("which one?", ["a0", "a1", "a2", "b0", "b1"])]


def test_doc_chunks_keeps_a_chunk_of_every_picked_document(tmp_path, qdrant, embeddings):
    """고른 문서마다 가장 높은 청크를 먼저 넣는다. 다른 문서의 둘째 청크가 점수가 더 높아도 밀려나지 않는다."""
    # Arrange: doc-a와 doc-c를 고른다. doc-c의 청크(0.1)는 doc-a의 어느 청크보다 점수가 낮다
    run = make_chunks_node(tmp_path, qdrant, embeddings, "[1, 3]", top_k=2)

    # Act
    result = run(CHUNK_STATE)

    # Assert: 점수만으로는 a2, a1이다
    assert result == {"contexts": ["a2", "c0"], "doc_ids": ["doc-a", "doc-c"]}


@pytest.mark.parametrize(
    "content, expected, scored",
    [("[3]", ["c0", "a1", "b0"], [["c0"]]), ("[]", ["a1", "b0", "c0"], [])],
    ids=["picked-documents-run-out", "no-pick"],
)
def test_doc_chunks_fills_remaining_slots_in_incoming_order(
    tmp_path, qdrant, embeddings, content, expected, scored
):
    """고른 문서의 청크가 모자라면 남는 자리를 앞 노드의 청크로 순서대로 채운다. 안 고르면 리랭커를 부르지 않는다."""
    # Arrange
    calls = []
    run = make_chunks_node(tmp_path, qdrant, embeddings, content, top_k=3, calls=calls)

    # Act
    result = run(CHUNK_STATE)

    # Assert
    assert result["contexts"] == expected
    assert [contexts for _, contexts in calls] == scored


def test_doc_chunks_reads_only_first_chunks_of_each_document(tmp_path, qdrant, embeddings):
    """문서마다 앞에서부터 doc_chunks개만 읽는다. 그 뒤의 청크는 점수가 높아도 넘기지 않는다."""
    # Arrange: doc-a에서 점수가 가장 높은 a2는 셋째 청크다
    run = make_chunks_node(tmp_path, qdrant, embeddings, "[1]", top_k=3, doc_chunks=2)

    # Act
    result = run(CHUNK_STATE)

    # Assert
    assert result == {"contexts": ["a0", "a1", "b0"], "doc_ids": ["doc-a", "doc-a", "doc-b"]}


def test_rejects_zero_doc_chunks():
    """doc_chunks가 1보다 작으면 틀을 읽기 전에 거부한다."""
    with pytest.raises(ValidationError) as exc:
        select(
            {"prompt": "x.txt", "top_k": 5, "doc_chunks": 0},
            Resources(config=make_config(), llm=PickingLLM("[1]")),
        )

    assert [(e["type"], e["loc"]) for e in exc.value.errors()] == [
        ("greater_than", ("doc_chunks",))
    ]


def test_repo_expand_select_chunks_config_builds_graph(monkeypatch):
    """저장소의 r4-expand-select-chunks config로 검색 그래프가 조립된다. 노드 넷의 인자가 맞는지 확인한다."""
    # Arrange: 프롬프트 경로가 저장소 루트 기준이다
    monkeypatch.chdir(REPO_ROOT)
    config = load_config(REPO_ROOT / "configs" / "exp" / "r4-expand-select-chunks.yaml")

    # Act
    graph = build_graph(config.retrieve, Resources(config=config))

    # Assert
    assert [name for name in graph.get_graph().nodes if name[0].isdigit()] == [
        "0-expand",
        "1-search",
        "2-rerank",
        "3-select",
    ]
