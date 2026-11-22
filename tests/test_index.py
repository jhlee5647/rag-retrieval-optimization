import uuid
import zlib

import pytest
from conftest import REPO_ROOT, add_index, make_config
from qdrant_client import models

from rag.config import load_config
from rag.index import bm25_vector, build_index, has_index, index_name


def point_id(doc_id, chunk):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{chunk}"))


def test_index_name_of_existing_index_is_fixed():
    """장비의 기존 색인 조건(naive.yaml, 문서 511,956개)의 이름이 바뀌지 않는다.

    장비의 컬렉션 data-d89aa11508031fc4에 이 이름을 별칭으로 붙인다. 계산식이 바뀌어 이 테스트가
    깨지면 별칭을 못 찾아 문서 전체의 색인을 새로 만들기 시작하므로, 별칭도 다시 붙여야 한다.
    """
    # Arrange
    config = load_config(REPO_ROOT / "configs" / "naive.yaml")

    # Act
    name = index_name(config, doc_count=511_956)

    # Assert
    assert name == "index-1fd89b43b5786220"


@pytest.mark.parametrize(
    "path", sorted((REPO_ROOT / "configs" / "exp").glob("*.yaml")), ids=lambda path: path.name
)
def test_experiment_configs_use_existing_index(path):
    """실험 config는 모두 검사를 통과하고, 장비의 기존 색인을 쓴다.

    색인 조건(청크 크기, 겹침, 임베딩 모델)을 한 글자라도 다르게 적으면 `retrieve`가 문서 전체의 색인을
    새로 만들기 시작한다. 색인을 다시 만드는 실험을 하기로 하면 이 테스트를 고친다.
    """
    # Act
    name = index_name(load_config(path), doc_count=511_956)

    # Assert
    assert name == "index-1fd89b43b5786220"


@pytest.mark.parametrize(
    "overrides, doc_count",
    [
        ({"chunking": {"size": 256, "overlap": 64}}, 10),
        ({"chunking": {"size": 512, "overlap": 32}}, 10),
        (
            {
                "embedding": {
                    "base_url": "http://127.0.0.1:8003/v1",
                    "model": "other",
                    "revision": "abc",
                    "max_input_tokens": 8192,
                }
            },
            10,
        ),
        (
            {
                "embedding": {
                    "base_url": "http://127.0.0.1:8003/v1",
                    "model": "test/model",
                    "revision": "def",
                    "max_input_tokens": 8192,
                }
            },
            10,
        ),
        ({}, 11),
    ],
    ids=["size", "overlap", "model", "revision", "doc_count"],
)
def test_index_name_changes_with_each_condition(overrides, doc_count):
    """색인 조건이 하나라도 다르면 다른 색인 이름이 나온다."""
    # Act / Assert
    assert index_name(make_config(**overrides), doc_count) != index_name(make_config(), 10)


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "embedding": {
                "base_url": "http://127.0.0.1:9999/v1",
                "model": "test/model",
                "revision": "abc",
                "max_input_tokens": 8192,
                "query_instruction": "x",
            }
        },
        {"index": {"url": "http://localhost:6333", "exact": True}},
        {"retrieve": [{"search": {"mode": "dense", "top_k": 9}}]},
    ],
    ids=["embedding-server-and-instruction", "index-url-and-exact", "retrieve-nodes"],
)
def test_index_name_ignores_non_index_settings(overrides):
    """문서 청크에 영향이 없는 설정(서버 주소, 질문 지시문, 검색 노드)은 색인을 바꾸지 않는다."""
    # Act / Assert
    assert index_name(make_config(**overrides), 10) == index_name(make_config(), 10)


def test_has_index_when_alias_exists(qdrant):
    """색인 이름의 별칭이 있으면 있다고 본다."""
    # Arrange
    add_index(qdrant, "index-0123456789abcdef", [("d1", "alpha")])

    # Act / Assert
    assert has_index(qdrant, "index-0123456789abcdef")


def test_has_index_false_when_alias_missing(qdrant):
    """별칭이 없으면 없다고 본다. 다른 이름의 별칭은 대신 쓰지 않는다."""
    # Arrange
    add_index(qdrant, "index-ffffffffffffffff", [("d1", "alpha")])

    # Act / Assert
    assert not has_index(qdrant, "index-0123456789abcdef")


def test_has_index_ignores_collection_without_alias(qdrant):
    """같은 이름의 컬렉션이 있어도 별칭이 아니면 완성된 색인으로 보지 않는다."""
    # Arrange
    qdrant.create_collection(
        "index-0123456789abcdef",
        vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE),
    )

    # Act / Assert
    assert not has_index(qdrant, "index-0123456789abcdef")


def test_bm25_vector_uses_crc32_ids_and_saturating_weights():
    """소문자로 바꾼 \\w+ 단어의 CRC32가 번호이고, 가중치는 f·(k1+1)/(f+k1)이다(k1 = 1.5)."""
    # Act
    v = bm25_vector("Alpha alpha, beta!")

    # Assert
    weights = dict(zip(v.indices, v.values, strict=True))
    assert weights == {
        zlib.crc32(b"alpha"): pytest.approx(2 * 2.5 / 3.5),
        zlib.crc32(b"beta"): pytest.approx(1.0),
    }


def words(text):
    """테스트용 토큰 수: 공백으로 나눈 단어 수.

    분할기는 공백이 없는 글을 글자로 쪼개 조각마다 길이를 더하므로, 한 단어짜리 문서는 쓰지 않는다.
    """
    return len(text.split())


def write_docs(tmp_path, docs):
    folder = tmp_path / "docs"
    folder.mkdir()
    for name, text in docs.items():
        (folder / f"{name}.txt").write_text(text, encoding="utf-8")
    return folder


NAME = "index-0123456789abcdef"
SMALL = make_config(chunking={"size": 4, "overlap": 1})


def test_build_index_follows_index_format(qdrant, embeddings, tmp_path):
    """토큰 수로 청크를 자르고, 점 ID·payload·BM25 벡터를 색인 형식대로 넣은 뒤 별칭을 붙인다."""
    # Arrange: 10단어 문서는 4단어, 겹침 1단어로 세 청크가 된다
    docs = write_docs(tmp_path, {"b": "alpha beta", "a": " ".join(f"w{i}" for i in range(10))})

    # Act
    build_index(qdrant, NAME, SMALL, docs, embeddings, words)

    # Assert
    assert has_index(qdrant, NAME)
    points = qdrant.retrieve("data-0123456789abcdef", ids=[point_id("a", 1)], with_vectors=True)
    assert points[0].payload == {
        "page_content": "w3 w4 w5 w6",
        "metadata": {"doc_id": "a", "chunk": 1},
    }
    stored, expected = points[0].vector["bm25"], bm25_vector("w3 w4 w5 w6")
    # Qdrant는 희소 벡터를 번호 순서로 정렬해 저장하므로 번호 → 가중치로 비교한다
    assert dict(zip(stored.indices, stored.values, strict=True)) == pytest.approx(
        dict(zip(expected.indices, expected.values, strict=True))
    )
    assert qdrant.count(NAME, exact=True).count == 4
    found = qdrant.retrieve(NAME, ids=[point_id(d, i) for d, i in [("a", 0), ("a", 2), ("b", 0)]])
    assert [p.payload["page_content"] for p in found] == [
        "w0 w1 w2 w3",
        "w6 w7 w8 w9",
        "alpha beta",
    ]


def test_build_index_replaces_unfinished_collection(qdrant, embeddings, tmp_path):
    """같은 이름의 컬렉션이 별칭 없이 남아 있으면 만들다 중단된 것이라 지우고 처음부터 만든다."""
    # Arrange: 중단된 색인에 지금 문서 폴더에 없는 청크가 남아 있다
    qdrant.create_collection(
        "data-0123456789abcdef",
        vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE),
    )
    qdrant.upsert(
        "data-0123456789abcdef",
        points=[models.PointStruct(id=point_id("old", 0), vector=[1.0, 0.0, 0.0], payload={})],
    )
    docs = write_docs(tmp_path, {"a": "alpha beta"})

    # Act
    build_index(qdrant, NAME, SMALL, docs, embeddings, words)

    # Assert
    assert qdrant.count(NAME, exact=True).count == 1
    assert qdrant.retrieve(NAME, ids=[point_id("old", 0)]) == []


def test_build_index_refuses_to_delete_collection_with_other_alias(qdrant, embeddings, tmp_path):
    """지울 컬렉션에 다른 별칭이 붙어 있으면 다른 색인일 수 있어 지우지 않고 멈춘다."""
    # Arrange
    add_index(
        qdrant, "index-ffffffffffffffff", [("d1", "alpha")], collection="data-0123456789abcdef"
    )
    docs = write_docs(tmp_path, {"a": "alpha"})

    # Act / Assert
    with pytest.raises(RuntimeError, match="index-ffffffffffffffff"):
        build_index(qdrant, NAME, SMALL, docs, embeddings, words)
    assert qdrant.count("index-ffffffffffffffff", exact=True).count == 1


def test_build_index_leaves_no_alias_when_point_count_differs(
    qdrant, embeddings, tmp_path, monkeypatch
):
    """넣은 청크 수와 저장된 점 수가 다르면 별칭을 붙이지 않아, 다음 실행이 다시 만들게 한다."""
    # Arrange: 저장된 점 수를 하나 적게 돌려준다
    docs = write_docs(tmp_path, {"a": "alpha", "b": "beta"})
    count = qdrant.count
    monkeypatch.setattr(
        qdrant, "count", lambda *a, **k: models.CountResult(count=count(*a, **k).count - 1)
    )

    # Act / Assert
    with pytest.raises(RuntimeError, match="별칭을 붙이지 않았다"):
        build_index(qdrant, NAME, SMALL, docs, embeddings, words)
    assert not has_index(qdrant, NAME)
