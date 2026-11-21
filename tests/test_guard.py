import os
import socket
import subprocess
import sys

import pytest

import rag

# 문서용으로 예약된 주소(RFC 5737)라 어디로도 라우팅되지 않는다. 차단이 실패해도 실제로 나가지 않는다.
EXTERNAL = ("192.0.2.1", 9)


def import_rag(env: dict[str, str]) -> subprocess.CompletedProcess:
    """주어진 환경변수만으로 새 프로세스에서 rag를 불러온다."""
    return subprocess.run(
        [sys.executable, "-c", "import rag"], env=env, check=False, capture_output=True, text=True
    )


@pytest.mark.parametrize("name", sorted(rag.TELEMETRY_SWITCHES))
def test_import_fails_without_switch(name):
    """전송 스위치가 하나라도 빠지면 패키지를 불러오지 못하고, 빠진 이름을 알린다."""
    # Arrange
    env = {k: v for k, v in os.environ.items() if k != name}

    # Act
    result = import_rag(env)

    # Assert
    assert result.returncode != 0
    assert f"{name}={rag.TELEMETRY_SWITCHES[name]}" in result.stderr


def test_import_fails_with_wrong_switch_value():
    """스위치가 있어도 값이 다르면(추적을 켜면) 불러오지 못한다."""
    # Arrange
    env = {**os.environ, "LANGSMITH_TRACING_V2": "true"}

    # Act
    result = import_rag(env)

    # Assert
    assert result.returncode != 0
    assert "LANGSMITH_TRACING_V2=false" in result.stderr


def test_blocks_external_tcp_connect():
    """루프백 밖으로의 TCP 접속은 연결을 시도하기 전에 오류를 낸다."""
    with pytest.raises(PermissionError, match="외부 접속 차단"):
        socket.create_connection(EXTERNAL, timeout=1)


def test_blocks_external_udp_send():
    """연결 없이 보내는 UDP도 루프백 밖이면 막는다."""
    with (
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock,
        pytest.raises(PermissionError, match="외부 접속 차단"),
    ):
        sock.sendto(b"x", EXTERNAL)


def test_allows_loopback_connect():
    """로컬 서버(Qdrant, vLLM)로 가는 루프백 접속은 막지 않는다."""
    # Arrange
    server = socket.create_server(("127.0.0.1", 0))

    # Act
    with server, socket.create_connection(server.getsockname(), timeout=1) as conn:
        # Assert
        assert conn.getpeername() == server.getsockname()
