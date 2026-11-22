"""색인 이름 계산, 재사용 확인, 새로 만들기."""

import hashlib
import itertools
import json
import re
import uuid
import zlib
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from huggingface_hub import hf_hub_download
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
from tokenizers import Tokenizer

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
SPARSE = INDEX_FORMAT["bm25"]["name"]
# 한 번에 읽어 자르고 임베딩해 넣는 문서 수. 문서 전체를 한꺼번에 메모리에 올리지 않기 위해서다.
FILES_PER_BATCH = 500
# Qdrant에 한 번에 보내는 점 수. 서버의 요청 크기 한도(32MB)를 넘지 않게 한다(점 하나에 약 10KB + 글).
POINTS_PER_REQUEST = 256


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


def has_index(client: QdrantClient, name: str) -> bool:
    """색인 이름의 별칭이 Qdrant에 있는지 본다.

    별칭은 색인을 끝까지 만든 뒤에만 붙으므로, 같은 이름의 컬렉션이 아니라 별칭을 본다.

    Args:
        client: Qdrant 클라이언트.
        name: `index_name`이 계산한 색인 이름.

    Returns:
        별칭이 있으면 True.
    """
    return name in {alias.alias_name for alias in client.get_aliases().aliases}


def token_counter(model: str, revision: str) -> Callable[[str], int]:
    """받아 둔 모델 파일의 토크나이저로 특수 토큰을 빼고 토큰 수를 세는 함수를 만든다.

    실행 중에는 내려받지 않는다(`local_files_only`). 모델 파일은 `HF_HOME`에 미리 있어야 한다.

    Raises:
        huggingface_hub.errors.LocalEntryNotFoundError: 그 리비전의 토크나이저 파일이 없는 경우.
    """
    path = hf_hub_download(model, "tokenizer.json", revision=revision, local_files_only=True)
    tokenizer = Tokenizer.from_file(path)
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids)


def bm25_vector(text: str) -> models.SparseVector:
    """청크의 BM25 희소 벡터. 단어의 CRC32가 색인 번호, `f·(k1+1)/(f+k1)`이 가중치다.

    단어가 드물수록 크게 치는 IDF는 Qdrant가 검색할 때 곱한다. 길이 보정은 하지 않는다(b = 0).
    """
    k1 = INDEX_FORMAT["bm25"]["k1"]
    freq = Counter(zlib.crc32(word.encode()) for word in re.findall(r"\w+", text.lower()))
    return models.SparseVector(
        indices=list(freq), values=[f * (k1 + 1) / (f + k1) for f in freq.values()]
    )


def bm25_query(text: str) -> models.SparseVector:
    """질문의 BM25 희소 벡터. 질문에 나온 단어마다 가중치가 1이다.

    단어는 `bm25_vector`와 같은 방식으로 자른다. 다르게 자르면 색인의 청크와 번호가 맞지 않는다.
    """
    ids = sorted({zlib.crc32(word.encode()) for word in re.findall(r"\w+", text.lower())})
    return models.SparseVector(indices=ids, values=[1.0] * len(ids))


def build_index(
    client: QdrantClient,
    name: str,
    config: Config,
    docs_dir: Path,
    embeddings: Embeddings,
    token_len: Callable[[str], int],
) -> None:
    """문서 폴더의 `*.txt`를 청크로 잘라 임베딩해 넣고, 다 넣은 뒤에 `name` 별칭을 붙인다.

    컬렉션 이름은 `data-<name과 같은 16자>`다. 그 컬렉션이 별칭 없이 남아 있으면 만들다 중단된
    것이라 지우고 처음부터 만든다. 규칙은 docs/design.md의 "색인 형식"을 따른다.

    Args:
        client: Qdrant 클라이언트.
        name: `index_name`이 계산한 색인 이름.
        config: 청크 크기와 겹침을 읽는다.
        docs_dir: 문서 폴더. 바로 아래의 `*.txt`를 파일 이름 순서로 읽는다.
        embeddings: 청크를 임베딩할 모델. 청크에는 질문 지시문을 붙이지 않는다.
        token_len: 글의 토큰 수를 세는 함수. 실제 실행에서는 `token_counter`다.

    Raises:
        RuntimeError: 지울 컬렉션에 다른 별칭이 붙어 있거나, 넣은 청크 수와 점 수가 다른 경우.
            점 수가 다르면 별칭을 붙이지 않으므로 다음 실행이 다시 만든다.
        ValueError: 문서에서 청크가 하나도 나오지 않은 경우.
    """
    collection = name.replace("index-", "data-", 1)
    if client.collection_exists(collection):
        others = [a.alias_name for a in client.get_collection_aliases(collection).aliases]
        if others:
            # 다른 색인으로 쓰이는 컬렉션일 수 있다. 다시 만드는 데 몇 시간이 걸리므로 지우지 않는다.
            raise RuntimeError(f"{collection}에 다른 별칭 {others}이 붙어 있어 지우지 않는다")
        client.delete_collection(collection)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunking.size,
        chunk_overlap=config.chunking.overlap,
        length_function=token_len,
    )
    files = sorted(docs_dir.glob("*.txt"))
    total = done = 0
    for batch in itertools.batched(files, FILES_PER_BATCH):
        texts, ids, payloads = [], [], []
        for path in batch:
            for i, text in enumerate(splitter.split_text(path.read_text(encoding="utf-8"))):
                texts.append(text)
                ids.append(str(uuid.uuid5(uuid.NAMESPACE_URL, f"{path.stem}:{i}")))
                # langchain-qdrant가 검색 결과를 Document로 되돌릴 때 읽는 칸 이름이다.
                payloads.append(
                    {"page_content": text, "metadata": {"doc_id": path.stem, "chunk": i}}
                )
        done += len(batch)
        if not texts:
            continue
        vectors = embeddings.embed_documents(texts)
        if not client.collection_exists(collection):
            # 벡터 크기는 임베딩 모델을 따르므로 첫 묶음을 임베딩한 뒤에 만든다.
            client.create_collection(
                collection,
                vectors_config=models.VectorParams(
                    size=len(vectors[0]),
                    distance=models.Distance.COSINE,
                    hnsw_config=models.HnswConfigDiff(**INDEX_FORMAT["hnsw"]),
                ),
                sparse_vectors_config={
                    SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
            )
        points = [
            # "": 이름 없는 기본 벡터. langchain-qdrant의 임베딩 검색이 이것을 쓴다.
            models.PointStruct(id=i, vector={"": v, SPARSE: bm25_vector(t)}, payload=p)
            for i, v, t, p in zip(ids, vectors, texts, payloads, strict=True)
        ]
        client.upload_points(collection, points, batch_size=POINTS_PER_REQUEST, wait=True)
        total += len(points)
        print(f"청크 {total}개, 문서 {done}/{len(files)}")
    if total == 0:
        raise ValueError(f"{docs_dir}의 문서에서 청크가 나오지 않았다")
    stored = client.count(collection, exact=True).count
    if stored != total:
        raise RuntimeError(
            f"{collection}: 넣은 청크 {total}개, 저장된 점 {stored}개. 별칭을 붙이지 않았다"
        )
    client.update_collection_aliases(
        change_aliases_operations=[
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(collection_name=collection, alias_name=name)
            )
        ]
    )
