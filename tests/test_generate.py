import json

import pytest
from conftest import EchoLLM

from rag.config import load_config
from rag.generate import generate, main

RETRIEVAL = """
index: {url: http://127.0.0.1:6333, exact: true}
chunking: {size: 512, overlap: 64}
embedding: {base_url: http://127.0.0.1:8003/v1, model: m, revision: r, max_input_tokens: 8192}
eval: {fields: {id: id, question: question, doc_ids: gold, type: type}, k: [1]}
retrieve:
  - search: {mode: dense, top_k: 2}
"""

GENERATION = """
generation:
  base_url: http://127.0.0.1:8001/v1
  model: g
  revision: r
  max_model_len: 32768
  max_tokens: 512
  temperature: 0
generate:
  - answer: {{prompt: {prompt}}}
judge: {{base_url: "http://127.0.0.1:8002/v1", model: j, revision: r}}
"""


@pytest.fixture
def setup(tmp_path):
    """검색 실행 폴더(2문항, 문맥 2개씩), 프롬프트 틀, 생성 config를 만든다."""
    contexts_dir = tmp_path / "runs" / "retrieve" / "naive" / "20260101-000000"
    contexts_dir.mkdir(parents=True)
    (contexts_dir / "config.yaml").write_text(RETRIEVAL)
    rows = [
        {"id": "q1", "question": "one?", "contexts": ["a1", "a2"], "doc_ids": ["d1", "d2"]},
        {"id": "q2", "question": "two?", "contexts": ["b1", "b2"], "doc_ids": ["d3", "d1"]},
    ]
    (contexts_dir / "contexts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    prompt = tmp_path / "answer.txt"
    prompt.write_text("C: {context}\nQ: {question}")
    config_path = tmp_path / "naive.yaml"
    config_path.write_text(RETRIEVAL + GENERATION.format(prompt=prompt))
    return config_path, contexts_dir, prompt


def run(config_path, contexts_dir, llm, tmp_path, servers=None):
    return generate(
        config_path,
        load_config(config_path),
        llm=llm,
        contexts_dir=contexts_dir,
        runs_root=tmp_path / "out",
        command=["python", "-m", "rag.generate", str(config_path)],
        servers=servers,
    )


def test_writes_answers_prompt_copy_and_run_record(setup, tmp_path):
    """문항마다 답을 results.jsonl에 쓰고, config와 프롬프트 틀의 사본, 사용한 검색 실행을 남긴다."""
    # Arrange
    config_path, contexts_dir, prompt = setup
    llm = EchoLLM()

    # Act
    run_dir = run(config_path, contexts_dir, llm, tmp_path)

    # Assert
    assert run_dir.parent == tmp_path / "out" / "generate" / "naive"
    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines()]
    assert rows == [
        {
            "id": "q1",
            "answer": "answer: Q: one?",
            "contexts": ["a1", "a2"],
            "doc_ids": ["d1", "d2"],
        },
        {
            "id": "q2",
            "answer": "answer: Q: two?",
            "contexts": ["b1", "b2"],
            "doc_ids": ["d3", "d1"],
        },
    ]
    assert (run_dir / "answer.txt").read_text() == prompt.read_text()
    assert (run_dir / "config.yaml").read_text() == config_path.read_text()
    assert json.loads((run_dir / "run.json").read_text())["contexts"] == str(contexts_dir)


def test_rejects_different_retrieval_conditions(setup, tmp_path):
    """config의 검색 조건이 그 검색 실행에 쓰인 것과 다르면, 모델을 부르기 전에 다른 항목을 대며 거부한다."""
    # Arrange: 검색 실행은 top_k 2였는데 config는 top_k 3이다
    config_path, contexts_dir, _ = setup
    config_path.write_text(config_path.read_text().replace("top_k: 2", "top_k: 3"))
    llm = EchoLLM()

    # Act / Assert
    with pytest.raises(ValueError, match="retrieve"):
        run(config_path, contexts_dir, llm, tmp_path)
    assert llm.prompts == []
    assert not (tmp_path / "out").exists()


def test_rejects_contexts_over_token_limit(setup, tmp_path):
    """문맥 수 × 청크 크기 + 답 한도 + 여유 1024가 max_model_len을 넘으면 모델을 부르기 전에 거부한다."""
    # Arrange: 2 × 512 + 512 + 1024 = 2560토큰이 필요한데 한도는 2559다
    config_path, contexts_dir, _ = setup
    text = config_path.read_text().replace("max_model_len: 32768", "max_model_len: 2559")
    config_path.write_text(text)
    llm = EchoLLM()

    # Act / Assert
    with pytest.raises(ValueError, match="2560"):
        run(config_path, contexts_dir, llm, tmp_path)
    assert llm.prompts == []


def test_main_rejects_config_without_generation(setup, tmp_path, capsys):
    """generation이 없는 검색 전용 config로는 서버에 연결하기 전에 거부한다."""
    # Arrange
    _, contexts_dir, _ = setup
    retrieval_only = tmp_path / "retrieval.yaml"
    retrieval_only.write_text(RETRIEVAL)

    # Act / Assert
    with pytest.raises(SystemExit):
        main([str(retrieval_only), "--contexts", str(contexts_dir)])
    assert "generation" in capsys.readouterr().err


def test_records_checked_servers(setup, tmp_path):
    """시작할 때 대조한 생성 서버 모델을 run.json에 남긴다."""
    # Arrange
    config_path, contexts_dir, _ = setup
    checked = {"gen": {"container": "rag-gen", "image": "i", "model": "g", "revision": "r"}}

    # Act
    run_dir = run(config_path, contexts_dir, EchoLLM(), tmp_path, servers=checked)

    # Assert
    assert json.loads((run_dir / "run.json").read_text())["servers"] == checked


def test_records_host_name(setup, tmp_path, monkeypatch):
    """여러 장비의 실행 폴더를 합친 뒤에도 어느 장비의 실행인지 가리도록, run.json에 호스트 이름을 남긴다."""
    # Arrange
    config_path, contexts_dir, _ = setup
    monkeypatch.setattr("platform.node", lambda: "node-b")

    # Act
    run_dir = run(config_path, contexts_dir, EchoLLM(), tmp_path)

    # Assert
    assert json.loads((run_dir / "run.json").read_text())["host"] == "node-b"


def test_rejects_different_reranker(setup, tmp_path):
    """리랭커도 검색 조건이다. 검색 실행과 config의 rerank 절이 다르면 모델을 부르기 전에 거부한다."""
    # Arrange: 검색 실행과 config 모두 rerank 노드가 있고, 리랭커 모델만 다르다
    config_path, contexts_dir, _ = setup
    node = "  - rerank: {top_k: 1}\nrerank: {base_url: http://127.0.0.1:8004/v1, model: MODEL, revision: r}\n"
    (contexts_dir / "config.yaml").write_text(RETRIEVAL + node.replace("MODEL", "a"))
    generation = config_path.read_text()[len(RETRIEVAL) :]
    config_path.write_text(RETRIEVAL + node.replace("MODEL", "b") + generation)
    llm = EchoLLM()

    # Act / Assert
    with pytest.raises(ValueError, match="rerank"):
        run(config_path, contexts_dir, llm, tmp_path)
    assert llm.prompts == []
