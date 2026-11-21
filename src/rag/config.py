"""config 파일의 스키마와 읽기."""

from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlsplit

import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    field_validator,
    model_validator,
)

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
    # 청크 크기의 상한. 이보다 긴 입력은 임베딩 서버가 받지 않는다.
    max_input_tokens: PositiveInt
    # 질문에만 붙인다. 기본값은 Qwen3-Embedding 모델 카드의 예시 문장이다.
    query_instruction: str = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )


class EvalFields(Strict):
    """이 도구가 쓰는 이름(왼쪽)에 대응하는 평가 파일의 필드 이름(오른쪽)."""

    id: str
    question: str
    doc_ids: str
    type: str


class Eval(Strict):
    """평가 파일의 필드 이름과 검색 지표를 잴 순위."""

    fields: EvalFields
    k: list[PositiveInt] = Field(min_length=1)


class Generation(Strict):
    """생성 서버와 모델, 토큰 한도, 온도."""

    base_url: LocalUrl
    model: str
    revision: str
    # 생성 서버가 받는 요청 하나(입력 + 출력)의 최대 토큰 수. 서버의 --max-model-len 이하로 둔다.
    max_model_len: PositiveInt
    max_tokens: PositiveInt
    temperature: float = Field(ge=0)


class Config(Strict):
    """config 전체. `generation`과 `generate`를 빼면 검색만 하는 config다."""

    index: Index
    chunking: Chunking
    embedding: Embedding
    eval: Eval
    # 항목마다 `{노드 이름: 인자}` 하나다. 적힌 순서대로 노드를 잇는다.
    retrieve: list[dict[str, dict[str, Any]]] = Field(min_length=1)
    generation: Generation | None = None
    generate: list[dict[str, dict[str, Any]]] | None = Field(default=None, min_length=1)

    @field_validator("retrieve", "generate")
    @classmethod
    def _one_node_per_step(cls, steps: list[dict[str, dict[str, Any]]] | None, info):
        for step in steps or []:
            if len(step) != 1:
                raise ValueError(f"{info.field_name}의 항목 하나에는 노드 하나만 적는다: {step}")
        return steps

    @model_validator(mode="after")
    def _generation_with_nodes(self):
        if (self.generation is None) != (self.generate is None):
            raise ValueError("generation과 generate는 함께 적거나 함께 뺀다")
        return self

    @model_validator(mode="after")
    def _chunks_fit(self):
        # 색인을 만드는 데 몇 시간이 걸리므로, 만들 수 없는 청크 설정은 시작 전에 거부한다.
        size, overlap = self.chunking.size, self.chunking.overlap
        if overlap >= size:
            raise ValueError(f"chunking.overlap({overlap})은 chunking.size({size})보다 작아야 한다")
        if size > self.embedding.max_input_tokens:
            raise ValueError(
                f"chunking.size({size})가 embedding.max_input_tokens"
                f"({self.embedding.max_input_tokens})를 넘는다"
            )
        return self


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
