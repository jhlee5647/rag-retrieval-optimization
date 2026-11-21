import json

import pytest

from rag.score import score

CONFIG = """
index: {url: http://127.0.0.1:6333}
chunking: {size: 512, overlap: 64}
embedding: {base_url: http://127.0.0.1:8003/v1, model: m, revision: r, max_input_tokens: 8192}
eval:
  fields: {id: id, question: question, answer: reference, doc_ids: gold, type: type}
  k: [1]
retrieve:
  - search: {mode: dense, top_k: 2}
generation:
  base_url: http://127.0.0.1:8001/v1
  model: g
  revision: r
  max_model_len: 32768
  max_tokens: 512
  temperature: 0
generate:
  - answer: {prompt: prompts/answer.txt}
judge: {base_url: "http://127.0.0.1:8002/v1", model: j, revision: r}
"""

QUESTIONS = [
    {"id": "q1", "question": "one?", "reference": "ref1", "gold": ["d1"], "type": "basic"},
    {"id": "q2", "question": "two?", "reference": "ref2", "gold": ["d1", "d2"], "type": "basic"},
    {"id": "q3", "question": "three?", "reference": "ref3", "gold": ["d9"], "type": "multi"},
    {"id": "q4", "question": "four?", "reference": "ref4", "gold": [], "type": "unanswerable"},
]
RESULTS = [
    {"id": "q1", "answer": "a1", "contexts": ["c1"], "doc_ids": ["d1"]},
    {"id": "q2", "answer": "a2", "contexts": ["c2"], "doc_ids": ["d2"]},
    {"id": "q3", "answer": "a3", "contexts": ["c3"], "doc_ids": ["d1"]},
    {"id": "q4", "answer": "I don't know.", "contexts": ["c4"], "doc_ids": ["d3"]},
]


@pytest.fixture
def setup(tmp_path):
    """생성 실행 폴더(4문항)와 평가 파일을 만든다. 근거 문서는 전부·일부·못 가져옴·지정 안 됨이다."""
    run_dir = tmp_path / "runs" / "generate" / "naive" / "20260101-000000"
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(CONFIG)
    (run_dir / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in RESULTS))
    eval_file = tmp_path / "eval.jsonl"
    eval_file.write_text("".join(json.dumps(q) + "\n" for q in QUESTIONS))
    return run_dir, eval_file


def fake_scorers(calls):
    """입력을 calls에 남기는 가짜 채점기. response_relevancy는 q2에서 출력 형식을 못 맞춘다."""

    async def faithfulness(x):
        calls.append(("faithfulness", x))
        return 1.0

    async def relevancy(x):
        if x["question"] == "two?":
            raise ValueError("bad json")
        return 0.5

    async def factual(x):
        return {"ref1": 0.2, "ref2": 0.4, "ref3": 0.0, "ref4": 0.6}[x["reference"]]

    return {
        "faithfulness": faithfulness,
        "response_relevancy": relevancy,
        "factual_correctness": factual,
    }


def test_passes_question_answer_contexts_and_reference(setup):
    """채점기에 평가 파일의 질문·정답과 생성 결과의 답·문맥을 문항끼리 맞춰 넘긴다."""
    # Arrange
    run_dir, eval_file = setup
    calls = []

    # Act
    score(run_dir, eval_file, fake_scorers(calls))

    # Assert
    first = next(x for name, x in calls if x["question"] == "one?")
    assert first == {"question": "one?", "answer": "a1", "contexts": ["c1"], "reference": "ref1"}


def test_writes_scores_with_retrieval_hit_and_empty_scores(setup):
    """문항마다 유형, 검색 적중, 점수를 쓰고, 채점에 실패한 지표는 0이 아닌 빈 점수로 둔다."""
    # Arrange
    run_dir, eval_file = setup

    # Act
    score(run_dir, eval_file, fake_scorers([]))

    # Assert
    rows = [json.loads(line) for line in (run_dir / "scores.jsonl").read_text().splitlines()]
    assert [(r["id"], r["hit"]) for r in rows] == [
        ("q1", "all"),
        ("q2", "some"),
        ("q3", "none"),
        ("q4", "no_gold"),
    ]
    assert rows[1]["scores"]["response_relevancy"] is None


def test_metrics_exclude_empty_scores_and_split_by_type_and_hit(setup):
    """평균은 빈 점수를 빼고 내며, 지표마다 빈 점수 수를 적고, 유형별·검색 적중별로 나눈다."""
    # Arrange
    run_dir, eval_file = setup

    # Act
    score(run_dir, eval_file, fake_scorers([]))

    # Assert
    m = json.loads((run_dir / "metrics.json").read_text())
    assert m["n"] == 4
    assert m["empty"] == {"faithfulness": 0, "response_relevancy": 1, "factual_correctness": 0}
    assert m["mean"]["response_relevancy"] == 0.5
    assert m["mean"]["factual_correctness"] == pytest.approx(0.3)
    assert m["by_type"]["basic"]["mean"]["factual_correctness"] == pytest.approx(0.3)
    assert m["by_hit"]["no_gold"] == {
        "n": 1,
        "mean": {"faithfulness": 1.0, "response_relevancy": 0.5, "factual_correctness": 0.6},
    }


def test_rejects_config_without_reference_field(setup):
    """config의 eval.fields에 정답이 없으면 채점하기 전에 거부한다."""
    # Arrange
    run_dir, eval_file = setup
    config = run_dir / "config.yaml"
    config.write_text(config.read_text().replace("answer: reference, ", ""))
    calls = []

    # Act / Assert
    with pytest.raises(ValueError, match="eval.fields.answer"):
        score(run_dir, eval_file, fake_scorers(calls))
    assert calls == []


def test_refuses_to_overwrite_scores(setup):
    """이미 채점한 폴더는 앞선 점수를 덮어쓰지 않고 거부한다."""
    # Arrange
    run_dir, eval_file = setup
    score(run_dir, eval_file, fake_scorers([]))
    calls = []

    # Act / Assert
    with pytest.raises(FileExistsError):
        score(run_dir, eval_file, fake_scorers(calls))
    assert calls == []


def test_stops_without_writing_when_a_metric_is_empty_everywhere(setup):
    """한 지표가 모든 문항에서 비면 서버에 닿지 못한 것으로 보고, 점수를 쓰지 않고 멈춘다."""
    # Arrange: 채점 서버가 꺼져 있어 Faithfulness가 매번 실패한다
    run_dir, eval_file = setup
    scorers = fake_scorers([])

    async def down(x):
        raise ConnectionError("refused")

    scorers["faithfulness"] = down

    # Act / Assert
    with pytest.raises(RuntimeError, match="faithfulness"):
        score(run_dir, eval_file, scorers)
    assert not (run_dir / "scores.jsonl").exists()


def test_records_checked_servers_in_metrics(setup):
    """시작할 때 대조한 채점·임베딩 서버 모델을 metrics.json에 남긴다."""
    # Arrange
    run_dir, eval_file = setup
    checked = {"judge": {"model": "j"}, "embed": {"model": "m"}}

    # Act
    score(run_dir, eval_file, fake_scorers([]), checked)

    # Assert
    assert json.loads((run_dir / "metrics.json").read_text())["servers"] == checked


def test_records_host_name_in_metrics(setup, monkeypatch):
    """채점은 생성과 다른 장비에서 할 수도 있다. 채점한 장비의 호스트 이름을 metrics.json에 남긴다."""
    # Arrange
    run_dir, eval_file = setup
    monkeypatch.setattr("platform.node", lambda: "node-b")

    # Act
    score(run_dir, eval_file, fake_scorers([]))

    # Assert
    assert json.loads((run_dir / "metrics.json").read_text())["host"] == "node-b"
