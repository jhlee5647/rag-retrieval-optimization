import pytest
from conftest import EchoLLM, make_config
from pydantic import ValidationError

from rag.nodes.answer import answer
from rag.pipeline import Resources


@pytest.fixture
def llm():
    return EchoLLM()


def write_prompt(tmp_path, text):
    path = tmp_path / "answer.txt"
    path.write_text(text, encoding="utf-8")
    return path


def test_fills_template_with_joined_contexts_and_question(tmp_path, llm):
    """틀의 {context}에 문맥을 ---로 이어 넣고 {question}에 질문을 넣어, 모델의 답을 answer에 담는다."""
    # Arrange
    prompt = write_prompt(tmp_path, "Context:\n{context}\nQuestion: {question}")
    run = answer({"prompt": str(prompt)}, Resources(config=make_config(), llm=llm))

    # Act
    result = run({"question": "why?", "contexts": ["first", "second"], "doc_ids": ["a", "b"]})

    # Assert
    assert llm.prompts == ["Context:\nfirst\n\n---\n\nsecond\nQuestion: why?"]
    assert result == {"answer": "answer: Question: why?"}


def test_rejects_template_without_placeholder(tmp_path, llm):
    """틀에 {context}나 {question}이 없으면 모델을 부르기 전에 빠진 자리를 대며 거부한다."""
    # Arrange
    prompt = write_prompt(tmp_path, "Question: {question}")

    # Act / Assert
    with pytest.raises(ValueError, match="{context}"):
        answer({"prompt": str(prompt)}, Resources(config=make_config(), llm=llm))


def test_rejects_misspelled_param(tmp_path, llm):
    """정의되지 않은 인자(prompt → promt)는 조용히 무시하지 않고 거부한다."""
    with pytest.raises(ValidationError):
        answer({"promt": "x.txt"}, Resources(config=make_config(), llm=llm))
