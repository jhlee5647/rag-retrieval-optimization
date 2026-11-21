import json

import pytest

from rag.config import EvalFields
from rag.data import count_docs, read_questions


def test_count_docs_counts_only_top_level_txt(tmp_path):
    """바로 아래의 *.txt만 센다. manifest.jsonl 같은 다른 파일과 하위 폴더는 세지 않는다."""
    # Arrange
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b__v2.txt").write_text("b")
    (tmp_path / "manifest.jsonl").write_text("{}")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.txt").write_text("c")

    # Act
    count = count_docs(tmp_path)

    # Assert
    assert count == 2


def test_count_docs_rejects_missing_folder(tmp_path):
    """폴더가 없으면 0개로 세지 않고 멈춘다. 0개로 세면 엉뚱한 색인 이름이 나온다."""
    with pytest.raises(NotADirectoryError):
        count_docs(tmp_path / "nope")


def test_read_questions_maps_field_names_in_file_order(tmp_path):
    """eval.fields에 적힌 평가 파일의 필드를 이 도구의 이름으로 바꿔, 파일 순서대로 읽는다."""
    # Arrange: 평가 파일의 필드 이름이 이 도구의 이름과 다르고, 쓰지 않는 필드와 빈 줄이 있다
    rows = [
        {"qid": "q2", "query": "second?", "expected": ["d1"], "kind": "basic", "reference": "x"},
        {"qid": "q1", "query": "first?", "expected": [], "kind": "unanswerable"},
    ]
    path = tmp_path / "eval.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n", encoding="utf-8")
    fields = EvalFields(id="qid", question="query", doc_ids="expected", type="kind")

    # Act
    questions = read_questions(path, fields)

    # Assert
    assert questions == [
        {"id": "q2", "question": "second?", "doc_ids": ["d1"], "type": "basic"},
        {"id": "q1", "question": "first?", "doc_ids": [], "type": "unanswerable"},
    ]
