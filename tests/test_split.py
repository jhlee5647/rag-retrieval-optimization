import hashlib
import json
from collections import Counter

import pytest

from rag.split import main, split

CONFIG = """
index: {url: http://127.0.0.1:6333, exact: true}
chunking: {size: 512, overlap: 64}
embedding: {base_url: http://127.0.0.1:8003/v1, model: m, revision: r, max_input_tokens: 8192}
eval: {fields: {id: id, question: question, doc_ids: gold, type: kind}, k: [1]}
retrieve:
  - search: {mode: dense, top_k: 1}
"""


def write_eval(tmp_path, types, field="type"):
    """유형 목록대로 문항 한 줄씩을 가진 평가 파일을 만든다. id는 q0, q1, ... 순서다."""
    path = tmp_path / "eval.jsonl"
    rows = [{"id": f"q{i}", field: name} for i, name in enumerate(types)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_splits_each_type_in_half(tmp_path):
    """문항이 적은 유형이 한쪽에 몰리지 않게, 유형마다 반씩 나눈다."""
    # Arrange: 유형 a가 8문항, b가 2문항, c가 4문항이고 파일에 섞여 있다
    types = ["a", "b", "a", "c", "a", "a", "c", "a", "b", "a", "c", "a", "c", "a"]
    eval_file = write_eval(tmp_path, types)

    # Act
    written = split(eval_file, "type")

    # Assert
    dev_file, test_file = tmp_path / "eval-dev.jsonl", tmp_path / "eval-test.jsonl"
    assert written == {dev_file: 7, test_file: 7}
    for path in (dev_file, test_file):
        assert Counter(row["type"] for row in read_rows(path)) == {"a": 4, "b": 1, "c": 2}


def test_odd_sized_types_still_give_equal_halves(tmp_path):
    """문항 수가 홀수인 유형이 둘이면, 남는 문항을 한쪽에 몰아주지 않아 두 파일의 문항 수가 같다."""
    # Arrange: 유형 a가 3문항, b가 5문항
    eval_file = write_eval(tmp_path, ["a", "a", "a", "b", "b", "b", "b", "b"])

    # Act
    written = split(eval_file, "type")

    # Assert: 두 파일이 4문항씩이고, 유형마다 두 파일의 차이는 한 문항이다
    dev_file, test_file = written
    assert list(written.values()) == [4, 4]
    dev = Counter(row["type"] for row in read_rows(dev_file))
    test = Counter(row["type"] for row in read_rows(test_file))
    assert (dev, test) == ({"a": 2, "b": 2}, {"a": 1, "b": 3})


def test_writes_original_lines_in_original_order(tmp_path):
    """줄을 다시 만들지 않고 그대로, 원본 순서대로 쓴다. 두 파일을 합치면 원본의 문항 전부다."""
    # Arrange: 다시 직렬화하면 달라지는 줄(공백, 키 순서, 한글)과 빈 줄, 줄바꿈 없는 마지막 줄
    lines = [
        '{"type":"a",   "id": "q0"}',
        '{"id": "q1", "type": "a", "question": "한글 질문?"}',
        "",
        '{"id": "q2", "extra": [1,2], "type": "a"}',
        '{"type": "a", "id": "q3"}',
        '{"id":"q4","type":"a"}',
        '{"id": "q5", "type": "a"}',
    ]
    eval_file = tmp_path / "eval.jsonl"
    eval_file.write_text("\n".join(lines), encoding="utf-8")

    # Act
    dev_file, test_file = split(eval_file, "type")

    # Assert
    originals = [line for line in lines if line]
    dev_text = dev_file.read_text(encoding="utf-8")
    test_text = test_file.read_text(encoding="utf-8")
    assert dev_text.endswith("\n") and test_text.endswith("\n")
    dev, test = dev_text.splitlines(), test_text.splitlines()
    assert sorted(dev + test) == sorted(originals)
    for part in (dev, test):
        positions = [originals.index(line) for line in part]
        assert positions == sorted(positions)


def test_split_is_fixed_for_the_same_file(tmp_path):
    """시드나 배정 방식이 바뀌면 이미 본 dev 문항이 test로 넘어간다. 같은 파일의 분할이 변하지 않게 지킨다."""
    # Arrange: 유형 둘이 번갈아 나오는 20문항
    eval_file = write_eval(tmp_path, ["a", "b"] * 10)

    # Act
    dev_file, _ = split(eval_file, "type")

    # Assert: 처음 만든 분할의 dev 문항을 그대로 적어 둔 것이다
    expected = ["q1", "q2", "q4", "q6", "q7", "q14", "q15", "q17", "q18", "q19"]
    assert [row["id"] for row in read_rows(dev_file)] == expected


@pytest.mark.parametrize("existing", ["dev", "test"])
def test_refuses_when_a_split_file_exists(tmp_path, existing):
    """실험에 쓰고 있는 분할을 덮어쓰지 않는다. 한 파일만 있어도 아무것도 쓰지 않고 멈춘다."""
    # Arrange
    eval_file = write_eval(tmp_path, ["a", "a"])
    kept = tmp_path / f"eval-{existing}.jsonl"
    kept.write_text("in use\n")

    # Act / Assert
    with pytest.raises(FileExistsError, match=kept.name):
        split(eval_file, "type")
    assert kept.read_text() == "in use\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(["eval.jsonl", kept.name])


def test_main_splits_eval_file_by_configured_type_field(tmp_path, monkeypatch, capsys):
    """`EVAL_FILE`을 config의 eval.fields.type 필드로 나누고, 파일마다 문항 수와 해시를 출력한다."""
    # Arrange: 유형 필드의 이름이 type이 아니라 kind인 평가 파일과 config
    eval_file = write_eval(tmp_path, ["a", "a", "b", "b"], field="kind")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(CONFIG)
    monkeypatch.setenv("EVAL_FILE", str(eval_file))

    # Act
    main([str(config_path)])

    # Assert
    out = capsys.readouterr().out.splitlines()
    for line, part in zip(out, ("dev", "test"), strict=True):
        path = tmp_path / f"eval-{part}.jsonl"
        assert Counter(row["kind"] for row in read_rows(path)) == {"a": 1, "b": 1}
        assert line == f"{path}  2문항  sha256 {hashlib.sha256(path.read_bytes()).hexdigest()}"
