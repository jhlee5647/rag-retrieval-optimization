"""RAG 실행 환경. 어떤 명령이든 이 패키지를 먼저 불러오므로, 여기서 한 번 유출 방지를 건다.

- 사용 통계와 추적 전송을 끄는 환경변수가 빠지거나 값이 다르면 불러오기를 거부한다.
- 이 프로세스가 루프백 밖으로 접속하거나 보내려 하면 오류를 낸다.
"""

import ipaddress
import os
import socket
import sys

# 설치된 의존성이 실제로 읽는 전송 스위치만 적는다 (docs/design.md의 "유출 방지").
TELEMETRY_SWITCHES = {
    # huggingface_hub는 둘 중 하나만 켜져도 통계를 보내지 않는다.
    "DO_NOT_TRACK": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    # langsmith는 이 넷을 정해진 순서로 보고 처음 값이 있는 것을 따른다. 순서에 기대지 않게 넷 다 끈다.
    "LANGSMITH_TRACING_V2": "false",
    "LANGCHAIN_TRACING_V2": "false",
    "LANGSMITH_TRACING": "false",
    "LANGCHAIN_TRACING": "false",
    "RAGAS_DO_NOT_TRACK": "true",
}
# 보낼 곳의 주소를 인자로 받는 소켓 감사 이벤트. 인자는 모두 (소켓, 주소)다.
_SEND_EVENTS = frozenset({"socket.connect", "socket.sendto", "socket.sendmsg"})


def _check_switches() -> None:
    wrong = [f"{k}={v}" for k, v in TELEMETRY_SWITCHES.items() if os.environ.get(k) != v]
    if wrong:
        raise RuntimeError(
            "전송 스위치 환경변수가 빠졌거나 값이 다르다. .env.example을 따라 .env에 적고 "
            f"`uv run --env-file .env ...`로 실행한다: {', '.join(wrong)}"
        )


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        # localhost가 아닌 이름 주소는 어디로 풀릴지 모르므로 막는다.
        return False


def _block_external(event: str, args: tuple) -> None:
    if event not in _SEND_EVENTS:
        return
    sock, address = args
    # 유닉스 소켓 등 IP가 아닌 소켓은 장비 밖으로 나가지 않는다.
    # sendmsg는 이미 연결된 소켓이면 주소가 None이고, 그 연결은 connect에서 검사했다.
    if sock.family not in (socket.AF_INET, socket.AF_INET6) or address is None:
        return
    if not _is_loopback(address[0]):
        raise PermissionError(f"외부 접속 차단: {event} {address}")


_check_switches()
# 감사 훅은 한 번 걸면 프로세스가 끝날 때까지 풀 수 없다.
sys.addaudithook(_block_external)
