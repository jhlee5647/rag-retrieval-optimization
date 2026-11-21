import os
import subprocess

import httpx
import pytest
from conftest import REPO_ROOT

from rag import serve

ENV = {"HF_HOME": "/models/hf", "QDRANT_STORAGE": "/store/qdrant"}


@pytest.fixture
def servers():
    return serve.load_servers(REPO_ROOT / "configs" / "servers.yaml")


class FakeDocker:
    """docker 명령을 기록하고, 컨테이너 상태를 `states` 순서대로 돌려주는 가짜.

    `states`의 값은 `docker container inspect`가 볼 상태다. None은 컨테이너 없음이다.
    """

    def __init__(self, states):
        self.states = list(states)
        self.commands = []

    def __call__(self, cmd, **kwargs):
        self.commands.append(cmd)
        if cmd[1:3] == ["container", "inspect"]:
            state = self.states.pop(0)
            if state is None:
                return subprocess.CompletedProcess(cmd, 1, "", "No such container")
            return subprocess.CompletedProcess(cmd, 0, "true\n" if state else "false\n", "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    def ran(self, *prefix):
        return [c for c in self.commands if c[1 : 1 + len(prefix)] == list(prefix)]


@pytest.fixture
def docker(monkeypatch):
    """docker를 가짜로 바꾸고 기다리는 시간을 없앤다. 테스트에서 상태를 정해 쓴다."""
    monkeypatch.setattr(serve.time, "sleep", lambda s: None)

    def make(states, ready):
        fake = FakeDocker(states)
        responses = iter(ready)
        monkeypatch.setattr(serve.subprocess, "run", fake)

        def get(url, timeout):
            if next(responses):
                return httpx.Response(200)
            raise httpx.ConnectError("not yet")

        monkeypatch.setattr(serve.httpx, "get", get)
        return fake

    return make


def test_repo_servers_file_defines_all_roles(servers):
    """저장소의 servers.yaml이 검사를 통과하고 설계의 역할을 모두 정의한다."""
    assert set(servers.vllm.servers) == {"embed", "gen", "judge"}
    assert servers.vllm.image.startswith("vllm/vllm-openai@sha256:")
    assert servers.qdrant.image.startswith("qdrant/qdrant@sha256:")


def test_vllm_command_is_local_offline_and_pinned(servers):
    """모델 서버는 127.0.0.1에만 열고, 오프라인·통계 꺼짐으로, 고정 리비전과 GPU 몫으로 띄운다."""
    # Act
    cmd = serve.docker_run_command("judge", servers, ENV)

    # Assert
    joined = " ".join(cmd)
    assert "-p 127.0.0.1:8002:8000" in joined
    for env in ["HF_HUB_OFFLINE=1", "HF_HUB_DISABLE_TELEMETRY=1", "VLLM_NO_USAGE_STATS=1"]:
        assert f"-e {env}" in joined
    assert "-v /models/hf:/models/hf" in joined
    assert cmd[cmd.index(servers.vllm.image) + 1 :] == [
        "--model", "RedHatAI/gemma-4-26B-A4B-it-NVFP4",
        "--revision", "5557756b8dce33ac72f2bd702b11729fdba3b839",
        "--gpu-memory-utilization", "0.66",
        "--max-model-len", "32768", "--language-model-only", "--reasoning-parser", "gemma4",
    ]  # fmt: skip


def test_qdrant_command_is_local_and_uses_storage_folder(servers):
    """Qdrant는 127.0.0.1에만 열고, 통계를 끄고, 기존 저장 폴더를 현재 사용자로 쓴다."""
    # Act
    joined = " ".join(serve.docker_run_command("qdrant", servers, ENV))

    # Assert
    assert "-p 127.0.0.1:6333:6333" in joined
    assert "-e QDRANT__TELEMETRY_DISABLED=true" in joined
    assert "-v /store/qdrant:/qdrant/storage" in joined
    assert f"--user {os.getuid()}:{os.getgid()}" in joined


def test_start_replaces_stopped_container_and_waits_until_ready(servers, docker):
    """멈춘 컨테이너는 지우고 새로 띄운 뒤, 상태 확인 주소가 200을 줄 때까지 기다린다."""
    # Arrange: 처음엔 멈춘 컨테이너가 있고, 띄운 뒤 두 번은 준비되지 않았다가 세 번째에 준비된다
    fake = docker(states=[False, True, True], ready=[False, False, True])

    # Act
    serve.start("embed", servers, ENV)

    # Assert
    assert fake.ran("rm") == [["docker", "rm", "rag-embed"]]
    assert len(fake.ran("run")) == 1
    # 띄우기 전 한 번, 준비되지 않은 두 번마다 한 번씩 컨테이너가 살아 있는지 봤다
    assert len(fake.ran("container", "inspect")) == 3


def test_start_refuses_when_already_running(servers, docker):
    """이미 떠 있는 서버를 다시 띄우지 않는다."""
    # Arrange
    fake = docker(states=[True], ready=[])

    # Act / Assert
    with pytest.raises(RuntimeError, match="이미 떠 있다"):
        serve.start("gen", servers, ENV)
    assert fake.ran("run") == []


def test_start_stops_waiting_when_container_exits(servers, docker):
    """기동 중에 컨테이너가 멈추면 시간 한도까지 기다리지 않고 로그 명령을 알린다."""
    # Arrange: 컨테이너가 없던 상태에서 띄웠는데, 준비되기 전에 멈춘다
    docker(states=[None, False], ready=[False])

    # Act / Assert
    with pytest.raises(RuntimeError, match="docker logs rag-judge"):
        serve.start("judge", servers, ENV)
