import subprocess
import sys
from pathlib import Path

import pytest
from conftest import make_config

from rag import pipeline
from rag.pipeline import NODES, Resources, build_graph, node


@pytest.fixture
def resources():
    return Resources(config=make_config(), vectorstore=None)


@pytest.fixture
def append_node(monkeypatch):
    """질문 끝에 인자 `s`를 붙이는 테스트용 노드를 등록하고, 테스트가 끝나면 지운다."""
    monkeypatch.setattr(pipeline, "NODES", dict(NODES))

    @node("append")
    def append(params, resources):
        return lambda state: {"question": state["question"] + params["s"]}


def test_nodes_run_in_listed_order(resources, append_node):
    """config에 적힌 순서대로 노드를 이어 실행하고, 같은 노드를 두 번 넣을 수 있다."""
    # Arrange
    steps = [{"append": {"s": "-1"}}, {"append": {"s": "-2"}}, {"append": {"s": "-3"}}]

    # Act
    result = build_graph(steps, resources).invoke({"question": "q"})

    # Assert
    assert result["question"] == "q-1-2-3"


def test_search_node_is_registered_from_nodes_folder():
    """nodes/ 폴더의 노드는 따로 불러오지 않아도 그래프를 만들 때 등록된다."""
    # Arrange: 이 테스트 프로세스는 test_search.py가 이미 노드를 불러왔으므로 새 프로세스에서 본다
    code = (
        "from conftest import make_config\n"
        "from rag import pipeline\n"
        "assert 'search' not in pipeline.NODES\n"
        "r = pipeline.Resources(config=make_config(), vectorstore=None)\n"
        "pipeline.build_graph([{'search': {'mode': 'dense', 'top_k': 1}}], r)\n"
    )

    # Act
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
    )

    # Assert
    assert result.returncode == 0, result.stderr


def test_rejects_unregistered_node(resources):
    """등록되지 않은 노드 이름은 실행 전에 이름을 대며 거부한다."""
    with pytest.raises(ValueError, match="serach"):
        build_graph([{"serach": {"mode": "dense", "top_k": 1}}], resources)
