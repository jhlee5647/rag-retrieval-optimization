"""색인 이름 계산과 재사용 확인."""

import hashlib
import json

from qdrant_client import QdrantClient

from rag.config import Config

# 색인을 만드는 규칙 (docs/design.md의 "색인 형식"). 규칙이 바뀌면 같은 조건이라도 다른 색인이
# 돼야 하므로 키에 넣는다. 여기 없는 규칙(점 ID, payload, 토큰 세는 법)이 바뀌면 version을 올린다.
INDEX_FORMAT = {
    "version": 1,
    "distance": "Cosine",
    "hnsw": {"m": 16, "ef_construct": 100},
    "bm25": {
        "name": "bm25",
        "modifier": "idf",
        "tokens": "lowercase, \\w+, crc32",
        "k1": 1.5,
        "b": 0,
    },
}
CHUNKING_METHOD = "RecursiveCharacterTextSplitter"


def index_name(config: Config, doc_count: int) -> str:
    """색인 조건으로 색인 이름을 계산한다.

    이 계산이 바뀌면 이미 붙여 둔 별칭을 못 찾으므로, 키의 구성과 직렬화 방식을 바꾸지 않는다.
    문서 내용은 읽지 않는다 (docs/design.md의 "색인 키").

    Args:
        config: 검사를 통과한 config. 청크 크기, 겹침, 임베딩 모델과 리비전을 쓴다.
        doc_count: 문서 폴더의 `*.txt` 파일 수.

    Returns:
        `index-<해시 앞 16자>`.
    """
    key = {
        "format": INDEX_FORMAT,
        "chunking": {
            "size": config.chunking.size,
            "overlap": config.chunking.overlap,
            "method": CHUNKING_METHOD,
        },
        "embedding": {"model": config.embedding.model, "revision": config.embedding.revision},
        "doc_count": doc_count,
    }
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
    return f"index-{digest[:16]}"


def require_index(client: QdrantClient, name: str) -> None:
    """색인 이름의 별칭이 Qdrant에 있는지 확인한다.

    별칭은 색인을 끝까지 만든 뒤에만 붙으므로, 같은 이름의 컬렉션이 아니라 별칭을 본다.

    Args:
        client: Qdrant 클라이언트.
        name: `index_name`이 계산한 색인 이름.

    Raises:
        LookupError: 그 이름의 별칭이 없는 경우.
    """
    if name not in {alias.alias_name for alias in client.get_aliases().aliases}:
        raise LookupError(
            f"색인 {name}이 Qdrant에 없다. 같은 조건으로 만든 컬렉션이 있으면 이 이름의 별칭을 "
            "붙인다. 색인을 새로 만드는 기능은 아직 없다."
        )
