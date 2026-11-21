"""serve 명령: 서버 정의 파일대로 이미지와 모델을 내려받고, docker 컨테이너를 띄우고 내린다.

    uv run --env-file .env python -m rag.serve download embed
    uv run --env-file .env python -m rag.serve start embed
    uv run --env-file .env python -m rag.serve stop embed

내려받기는 외부 접속이 허용되는 유일한 명령이다. 받는 일은 docker가 하고, 이 프로세스의 외부 접속
차단은 풀지 않는다.

모델 폴더와 Qdrant 저장 폴더는 환경변수 `HF_HOME`과 `QDRANT_STORAGE`에서 읽는다.
"""

import argparse
import json
import os
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlsplit

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


def docker_download_commands(
    role: str, servers: Servers, env: Mapping[str, str]
) -> list[list[str]]:
    """역할의 이미지와 모델 파일을 내려받는 docker 명령들을 만든다.

    모델은 vLLM 이미지에 든 `hf download`로 `HF_HOME`에 고정 리비전을 받는다. 외부와 통신하는 것은
    docker 데몬과 그 컨테이너뿐이다.

    Raises:
        KeyError: 역할이 정의에 없거나, 모델 서버인데 `HF_HOME`이 없는 경우.
    """
    if role == "qdrant":
        return [["docker", "pull", servers.qdrant.image]]
    s, hf_home = servers.vllm.servers[role], env["HF_HOME"]
    return [
        ["docker", "pull", servers.vllm.image],
        [
            "docker", "run", "--rm", "--entrypoint", "hf",
            "-e", f"HF_HOME={hf_home}", "-e", "HF_HUB_DISABLE_TELEMETRY=1",
            "-v", f"{hf_home}:{hf_home}",
            servers.vllm.image, "download", s.model, "--revision", s.revision,
        ],
    ]  # fmt: skip


def download(role: str, servers: Servers, env: Mapping[str, str]) -> None:
    """역할의 이미지와 모델 파일을 내려받는다. 이미 받은 것은 docker와 hf가 건너뛴다."""
    for command in docker_download_commands(role, servers, env):
        subprocess.run(command, check=True)


def check_model(role: str, base_url: str, model: str, revision: str) -> dict[str, str]:
    """config의 서버 주소에 떠 있는 모델과 리비전이 config와 같은지 확인한다.

    vLLM API는 모델 이름만 알려 주므로, 그 포트를 연 컨테이너의 실행 인자에서 리비전을 읽는다. 같은
    주소의 `/v1/models`도 그 이름을 응답하는지 본다.

    Args:
        role: 안내 메시지에 쓸 역할 이름.
        base_url: config의 서버 주소. 예: `http://127.0.0.1:8003/v1`.
        model: config의 모델 이름.
        revision: config의 고정 리비전.

    Returns:
        실행 기록에 남길 `{"container", "image", "model", "revision"}`.

    Raises:
        RuntimeError: 그 포트에 컨테이너가 없거나, 모델·리비전이 다르거나, 서버가 그 이름을 응답하지
            않는 경우. 띄울 명령을 함께 알린다.
    """
    hint = f"띄울 명령: uv run --env-file .env python -m rag.serve start {role}"
    port = urlsplit(base_url).port
    found = subprocess.run(
        ["docker", "ps", "--filter", f"publish={port}", "--format", "{{.Names}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    if len(found) != 1:
        raise RuntimeError(
            f"포트 {port}에 떠 있는 {role} 컨테이너가 하나가 아니다: {found}. {hint}"
        )
    name = found[0]
    inspected = subprocess.run(
        ["docker", "inspect", name, "--format", "{{json .Args}}|{{.Config.Image}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    args_json, image = inspected.split("|", 1)
    args = json.loads(args_json)
    running = {
        flag: args[args.index(flag) + 1] for flag in ("--model", "--revision") if flag in args
    }
    if running != {"--model": model, "--revision": revision}:
        raise RuntimeError(
            f"{name}의 모델이 config와 다르다: 떠 있는 것 {running}, config {model}@{revision}. {hint}"
        )
    served = [m["id"] for m in httpx.get(f"{base_url}/models", timeout=5).json()["data"]]
    if model not in served:
        raise RuntimeError(f"{base_url}이 {model}을 응답하지 않는다: {served}. {hint}")
    return {"container": name, "image": image, "model": model, "revision": revision}


def stop(role: str) -> None:
    """역할의 컨테이너를 내린다. 로그를 볼 수 있게 컨테이너는 지우지 않는다."""
    subprocess.run(["docker", "stop", container_name(role)], check=True)


def main(argv: list[str] | None = None) -> None:
    """명령줄 인자를 읽고 서버를 띄우거나 내린다.

    Args:
        argv: 명령줄 인자. None이면 `sys.argv`를 쓴다.
    """
    parser = argparse.ArgumentParser(prog="python -m rag.serve", description=__doc__)
    parser.add_argument("action", choices=["download", "start", "stop"])
    parser.add_argument("role", help="qdrant 또는 configs/servers.yaml의 vllm.servers에 적힌 역할")
    args = parser.parse_args(argv)

    servers = load_servers(SERVERS_FILE)
    roles = ["qdrant", *servers.vllm.servers]
    if args.role not in roles:
        parser.error(f"역할은 {', '.join(roles)} 중 하나다: {args.role}")
    if args.action == "download":
        download(args.role, servers, os.environ)
    elif args.action == "start":
        start(args.role, servers, os.environ)
    else:
        stop(args.role)


if __name__ == "__main__":
    main()
