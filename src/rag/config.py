"""config 파일의 스키마와 읽기."""

from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlsplit

import yaml
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator

LOCAL_HOSTS = ("127.0.0.1", "localhost")


def _require_local(url: str) -> str:
    # 데이터를 장비 밖으로 보내지 않도록 이 장비의 서버만 받는다.
    if urlsplit(url).hostname not in LOCAL_HOSTS:
        raise ValueError(f"서버 주소는 127.0.0.1이나 localhost만 받는다: {url}")
    return url


LocalUrl = Annotated[str, AfterValidator(_require_local)]


class Strict(BaseModel):
    """정의되지 않은 항목을 거부하는 config 모델의 기반.

    오타로 적은 항목(`overlap`을 `overlp`로)이 조용히 무시되지 않게 한다.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class Index(Strict):
    """Qdrant 주소와 검색 방식."""

    url: LocalUrl
    # true면 근사(HNSW)가 아닌 정확 검색을 한다. Qdrant의 기본값을 따라 기본은 근사 검색이다.
    exact: bool = False


class Chunking(Strict):
    """청크 크기와 겹침. 단위는 임베딩 모델 토크나이저의 토큰 수다."""

    size: int
    overlap: int


class Embedding(Strict):
    """임베딩 서버와 모델."""

    base_url: LocalUrl
    model: str
    revision: str
    # 질문에만 붙인다. 기본값은 Qwen3-Embedding 모델 카드의 예시 문장이다.
    query_instruction: str = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )


class Config(Strict):
    """검색에 필요한 config 전체."""

    index: Index
    chunking: Chunking
    embedding: Embedding
    # 항목마다 `{노드 이름: 인자}` 하나다. 적힌 순서대로 노드를 잇는다.
    retrieve: list[dict[str, dict[str, Any]]] = Field(min_length=1)

    @field_validator("retrieve")
    @classmethod
    def _one_node_per_step(cls, steps: list[dict[str, dict[str, Any]]]):
        for step in steps:
            if len(step) != 1:
                raise ValueError(f"retrieve의 항목 하나에는 노드 하나만 적는다: {step}")
        return steps


def load_config(path: Path) -> Config:
    """config 파일을 읽고 검사한다.

    Args:
        path: YAML config 파일 경로.

    Returns:
        검사를 통과한 config.

    Raises:
        pydantic.ValidationError: 정의되지 않은 항목이 있거나 값의 형식이 틀린 경우.
    """
    return Config.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
