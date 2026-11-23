"""노드 등록과, config의 노드 목록을 일직선 그래프로 조립하기."""

import importlib
import pkgutil
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_qdrant import QdrantVectorStore
from langgraph.graph import END, START, StateGraph

from rag import nodes
from rag.config import Config


class State(TypedDict, total=False):
    """노드들이 차례로 읽고 쓰는 공용 데이터.

    LangGraph는 여기 없는 필드를 오류 없이 버린다.
    """

    question: str
    contexts: list[str]
    # contexts와 같은 순서로, 청크마다 그 청크가 나온 문서 ID다.
    doc_ids: list[str]
    answer: str


@dataclass(frozen=True)
class Resources:
    """노드가 쓰는 config와 외부 자원.

    검색 단계는 벡터 저장소를, rerank 노드가 있으면 리랭커도, select 노드가 있으면 생성 모델도 쓴다.
    생성 단계는 생성 모델을 쓴다.
    """

    config: Config
    vectorstore: QdrantVectorStore | None = None
    # 질문과 청크들을 받아, 청크 순서대로 관련도 점수를 돌려준다.
    reranker: Callable[[str, list[str]], list[float]] | None = None
    llm: BaseChatModel | None = None


# 노드 하나. config의 인자와 자원을 받아, State를 받아 바뀐 필드만 돌려주는 함수를 만든다.
NodeFactory = Callable[[dict[str, Any], Resources], Callable[[State], dict[str, Any]]]

NODES: dict[str, NodeFactory] = {}


def node(name: str) -> Callable[[NodeFactory], NodeFactory]:
    """노드를 config에서 부를 이름으로 등록하는 데코레이터.

    Args:
        name: config의 노드 목록에 적을 이름.

    Returns:
        노드를 등록하고 그대로 돌려주는 데코레이터.
    """

    def register(factory: NodeFactory) -> NodeFactory:
        NODES[name] = factory
        return factory

    return register


def build_graph(steps: list[dict[str, dict[str, Any]]], resources: Resources):
    """config의 노드 목록을 적힌 순서대로 이은 그래프를 만든다.

    `rag/nodes/` 아래의 모듈을 모두 불러와 노드를 등록한 뒤 조립한다. 노드가 인자를 검사하므로,
    인자 오타도 여기서 실행 전에 드러난다.

    Args:
        steps: config의 노드 목록. 항목마다 `{노드 이름: 인자}` 하나다.
        resources: 노드에 넘길 config와 외부 자원.

    Returns:
        컴파일된 LangGraph 그래프.

    Raises:
        ValueError: 등록되지 않은 노드 이름이 있는 경우.
    """
    for module in pkgutil.iter_modules(nodes.__path__):
        importlib.import_module(f"{nodes.__name__}.{module.name}")

    graph = StateGraph(State)
    previous = START
    for i, step in enumerate(steps):
        ((name, params),) = step.items()
        if name not in NODES:
            raise ValueError(f"등록되지 않은 노드: {name} (등록된 노드: {sorted(NODES)})")
        # 순번을 붙인다. LangGraph가 State 필드와 같은 노드 이름을 거부하고, 같은 노드를 두 번
        # 넣을 수도 있어야 해서다.
        graph_name = f"{i}-{name}"
        graph.add_node(graph_name, NODES[name](params, resources))
        graph.add_edge(previous, graph_name)
        previous = graph_name
    graph.add_edge(previous, END)
    return graph.compile()
