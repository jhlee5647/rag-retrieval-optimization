"""serve 명령: 서버 정의 파일대로 docker 컨테이너를 띄우고 내린다.

    uv run --env-file .env python -m rag.serve start embed
    uv run --env-file .env python -m rag.serve stop embed

모델 폴더와 Qdrant 저장 폴더는 환경변수 `HF_HOME`과 `QDRANT_STORAGE`에서 읽는다.
"""

import argparse
import os
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path

import httpx
import yaml
from pydantic import Field

from rag.config import Strict

SERVERS_FILE = Path("configs/servers.yaml")
# 장비에서 가장 오래 걸린 기동은 gen의 약 4분 40초다.
START_TIMEOUT_SECONDS = 600
POLL_SECONDS = 5


class Qdrant(Strict):
    """Qdrant 컨테이너의 정의."""

    image: str
    port: int


class ModelServer(Strict):
    """vLLM 모델 서버 하나의 정의."""

    model: str
    revision: str
    port: int
    gpu_share: float = Field(gt=0, le=1)
    args: list[str]


class Vllm(Strict):
    """vLLM 이미지와 역할별 모델 서버."""

    image: str
    servers: dict[str, ModelServer]


class Servers(Strict):
    """서버 정의 파일 전체."""

    qdrant: Qdrant
    vllm: Vllm


def load_servers(path: Path) -> Servers:
    """서버 정의 파일을 읽고 검사한다.

    Args:
        path: YAML 서버 정의 파일 경로.

    Returns:
        검사를 통과한 서버 정의.

    Raises:
        pydantic.ValidationError: 정의되지 않은 항목이 있거나 값의 형식이 틀린 경우.
    """
    return Servers.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def container_name(role: str) -> str:
    """역할의 컨테이너 이름."""
    return f"rag-{role}"


def health_url(role: str, servers: Servers) -> str:
    """서버가 요청을 받을 준비가 되면 200을 돌려주는 주소."""
    if role == "qdrant":
        return f"http://127.0.0.1:{servers.qdrant.port}/readyz"
    return f"http://127.0.0.1:{servers.vllm.servers[role].port}/health"


def docker_run_command(role: str, servers: Servers, env: Mapping[str, str]) -> list[str]:
    """역할의 컨테이너를 띄우는 `docker run` 명령을 만든다.

    포트는 127.0.0.1에만 열고, 컨테이너 안의 사용 통계 전송과 모델 내려받기를 끈다.

    Args:
        role: `qdrant` 또는 `servers.vllm.servers`의 역할 이름.
        servers: 서버 정의.
        env: `HF_HOME`(vLLM)이나 `QDRANT_STORAGE`(Qdrant)를 담은 환경변수.

    Returns:
        `docker`부터 시작하는 명령의 인자 목록.

    Raises:
        KeyError: 역할이 정의에 없거나 필요한 환경변수가 없는 경우.
    """
    name = container_name(role)
    if role == "qdrant":
        q, storage = servers.qdrant, env["QDRANT_STORAGE"]
        # 기본 사용자인 root로 돌면 저장 폴더에 sudo 없이 지울 수 없는 파일이 생긴다. 그 사용자는 이미지
        # 안의 스냅샷 폴더(root 소유)에 쓰지 못해 기동이 멈추므로 스냅샷 폴더를 저장 폴더 안으로 옮긴다.
        return [
            "docker", "run", "-d", "--name", name, "--user", f"{os.getuid()}:{os.getgid()}",
            "-p", f"127.0.0.1:{q.port}:6333",
            "-e", "QDRANT__TELEMETRY_DISABLED=true",
            "-e", "QDRANT__STORAGE__SNAPSHOTS_PATH=/qdrant/storage/snapshots",
            "-v", f"{storage}:/qdrant/storage",
            q.image,
        ]  # fmt: skip
    s, hf_home = servers.vllm.servers[role], env["HF_HOME"]
    # ipc=host: vLLM 내부 프로세스가 주고받는 공유 메모리에 docker 기본값(64MB)이 모자라 vLLM 문서가 권한다.
    return [
        "docker", "run", "-d", "--name", name, "--gpus", "all", "--ipc=host",
        "-p", f"127.0.0.1:{s.port}:8000",
        "-e", f"HF_HOME={hf_home}", "-e", "HF_HUB_OFFLINE=1", "-e", "HF_HUB_DISABLE_TELEMETRY=1",
        "-e", "VLLM_NO_USAGE_STATS=1", "-e", "DO_NOT_TRACK=1",
        "-v", f"{hf_home}:{hf_home}",
        servers.vllm.image,
        "--model", s.model, "--revision", s.revision,
        "--gpu-memory-utilization", str(s.gpu_share), *s.args,
    ]  # fmt: skip


def _running(name: str) -> bool | None:
    """컨테이너가 떠 있으면 True, 멈춰 있으면 False, 없으면 None."""
    result = subprocess.run(
        ["docker", "container", "inspect", "-f", "{{.State.Running}}", name],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() == "true"


def _ready(url: str) -> bool:
    try:
        return httpx.get(url, timeout=2).status_code == 200
    except httpx.TransportError:
        return False


def start(role: str, servers: Servers, env: Mapping[str, str]) -> None:
    """역할의 컨테이너를 띄우고, 요청을 받을 준비가 될 때까지 기다린다.

    멈춘 컨테이너는 기동 실패 로그를 볼 수 있게 남겨 두었다가, 다시 띄울 때 지운다.

    Raises:
        RuntimeError: 이미 떠 있거나, 준비되기 전에 컨테이너가 멈췄거나, 시간 안에 준비되지 않은 경우.
        subprocess.CalledProcessError: docker 명령이 실패한 경우.
    """
    name = container_name(role)
    state = _running(name)
    if state:
        raise RuntimeError(f"{name}이 이미 떠 있다. 다시 띄우려면 먼저 stop한다.")
    if state is False:
        subprocess.run(["docker", "rm", name], check=True)
    subprocess.run(docker_run_command(role, servers, env), check=True)

    url = health_url(role, servers)
    deadline = time.monotonic() + START_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if _ready(url):
            print(f"{name}: 준비됨 ({url})")
            return
        if not _running(name):
            raise RuntimeError(f"{name}이 준비되기 전에 멈췄다. 로그: docker logs {name}")
        time.sleep(POLL_SECONDS)
    raise RuntimeError(
        f"{name}이 {START_TIMEOUT_SECONDS}초 안에 준비되지 않았다. 로그: docker logs {name}"
    )


def stop(role: str) -> None:
    """역할의 컨테이너를 내린다. 로그를 볼 수 있게 컨테이너는 지우지 않는다."""
    subprocess.run(["docker", "stop", container_name(role)], check=True)


def main(argv: list[str] | None = None) -> None:
    """명령줄 인자를 읽고 서버를 띄우거나 내린다.

    Args:
        argv: 명령줄 인자. None이면 `sys.argv`를 쓴다.
    """
    parser = argparse.ArgumentParser(prog="python -m rag.serve", description=__doc__)
    parser.add_argument("action", choices=["start", "stop"])
    parser.add_argument("role", help="qdrant 또는 configs/servers.yaml의 vllm.servers에 적힌 역할")
    args = parser.parse_args(argv)

    servers = load_servers(SERVERS_FILE)
    roles = ["qdrant", *servers.vllm.servers]
    if args.role not in roles:
        parser.error(f"역할은 {', '.join(roles)} 중 하나다: {args.role}")
    if args.action == "start":
        start(args.role, servers, os.environ)
    else:
        stop(args.role)


if __name__ == "__main__":
    main()
