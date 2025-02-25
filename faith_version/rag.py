from typing import Any, AsyncIterator, Callable, Iterator, cast, final
from dataclasses import asdict, dataclass, field
import requests
import json
from abc import ABC, abstractmethod



# @dataclass
# class StorageNameSpace(ABC):
#     namespace: str
#     global_config: dict[str, Any]

#     async def initialize(self):
#         """Initialize the storage"""
#         pass

#     async def finalize(self):
#         """Finalize the storage"""
#         pass

#     @abstractmethod
#     async def index_done_callback(self) -> None:
#         """Commit the storage operations after indexing"""


# @dataclass
# class BaseVectorStorage(StorageNameSpace, ABC):
#     embedding_func: EmbeddingFunc
#     cosine_better_than_threshold: float = field(default=0.2)
#     meta_fields: set[str] = field(default_factory=set)

#     @abstractmethod
#     async def query(self, query: str, top_k: int) -> list[dict[str, Any]]:
#         """Query the vector storage and retrieve top_k results."""

#     @abstractmethod
#     async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
#         """Insert or update vectors in the storage."""

#     @abstractmethod
#     async def delete_entity(self, entity_name: str) -> None:
#         """Delete a single entity by its name."""

#     @abstractmethod
#     async def delete_entity_relation(self, entity_name: str) -> None:
#         """Delete relations for a given entity."""


# @dataclass
# class BaseKVStorage(StorageNameSpace, ABC):
#     embedding_func: EmbeddingFunc

#     @abstractmethod
#     async def get_by_id(self, id: str) -> dict[str, Any] | None:
#         """Get value by id"""

#     @abstractmethod
#     async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
#         """Get values by ids"""

#     @abstractmethod
#     async def filter_keys(self, keys: set[str]) -> set[str]:
#         """Return un-exist keys"""

#     @abstractmethod
#     async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
#         """Upsert data"""


# @dataclass
# class BaseGraphStorage(StorageNameSpace, ABC):
#     embedding_func: EmbeddingFunc

#     @abstractmethod
#     async def has_node(self, node_id: str) -> bool:
#         """Check if an edge exists in the graph."""

#     @abstractmethod
#     async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
#         """Get the degree of a node."""

#     @abstractmethod
#     async def node_degree(self, node_id: str) -> int:
#         """Get the degree of an edge."""

#     @abstractmethod
#     async def edge_degree(self, src_id: str, tgt_id: str) -> int:
#         """Get a node by its id."""

#     @abstractmethod
#     async def get_node(self, node_id: str) -> dict[str, str] | None:
#         """Get an edge by its source and target node ids."""

#     @abstractmethod
#     async def get_edge(
#         self, source_node_id: str, target_node_id: str
#     ) -> dict[str, str] | None:
#         """Get all edges connected to a node."""

#     @abstractmethod
#     async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
#         """Upsert a node into the graph."""

#     @abstractmethod
#     async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
#         """Upsert an edge into the graph."""

#     @abstractmethod
#     async def upsert_edge(
#         self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
#     ) -> None:
#         """Delete a node from the graph."""

#     @abstractmethod
#     async def delete_node(self, node_id: str) -> None:
#         """Embed nodes using an algorithm."""

#     @abstractmethod
#     async def embed_nodes(
#         self, algorithm: str
#     ) -> tuple[np.ndarray[Any, Any], list[str]]:
#         """Get all labels in the graph."""

#     @abstractmethod
#     async def get_all_labels(self) -> list[str]:
#         """Get a knowledge graph of a node."""

#     @abstractmethod
#     async def get_knowledge_graph(
#         self, node_label: str, max_depth: int = 5
#     ) -> KnowledgeGraph:
#         """Retrieve a subgraph of the knowledge graph starting from a given node."""


@final
@dataclass
class RAG:
    
    llm_model_func = None
    

def sendGLM(content):
    json_d = {"content": content}
    resp = requests.post(
            # "http://192.168.31.61:8089/predict", data=json.dumps(json_d), headers=headers, verify=False)
            "http://192.168.1.77:8089/predict", data=json.dumps(json_d), verify=False)
    parsed = json.loads(resp.content)
    # print(parsed)
    try:
        return parsed['resp']
    except:
        return ""


async def call_LLM(input_text, history_messages=None):
    # final_result = '''(“entity”<|>“正方云曦山”<|>“geo”<|>“正方云曦山是一个位于香洲区珠海大道157号的塔楼建筑，包含366户，4栋楼，绿化率为35%，容积率为3.3，属于商品房，无供暖，使用民水和民电，属于香洲区容国团中学和容国团小学的学区。”)## (“entity”<|>“香洲区”<|>“geo”<|>“香洲区是正方云曦山所在的地理区域，也是容国团中学和容国团小学所在的学区。”)## (“entity”<|>“珠海大道157号”<|>“geo”<|>“珠海大道157号是正方云曦山的具体地址。”)## (“entity”<|>“容国团中学”<|>“organization”<|>“容国团中学是正方云曦山所属的初中学区。”)## (“entity”<|>“容国团小学”<|>“organization”<|>“容国团小学是正方云曦山所属的小学学区。”)## (“relationship”<|>“正方云曦山”<|>“香洲区”<|>“正方云曦山位于香洲区，表明地理位置的从属关系。”<|>“地理位置, 从属关系”<|>9)## (“relationship”<|>“正方云曦山”<|>“珠海大道157号”<|>“正方云曦山的地址是珠海大道157号，表明具体的地理位置信息。”<|>“具体地址, 地理位置”<|>10)## (“relationship”<|>“正方云曦山”<|>“容国团中学”<|>“正方云曦山属于容国团中学的学区，表明教育资源的分配关系。”<|>“学区划分, 教育资源”<|>8)## (“relationship”<|>“正方云曦山”<|>“容国团小学”<|>“正方云曦山属于容国团小学的学区，表明教育资源的分配关系。”<|>“学区划分, 教育资源”<|>8)## (“content_keywords”<|>“房地产, 地理位置, 建筑特点, 学区划分”)<|COMPLETE|>'''

    # final_result = '(“entity”<|>“7月5日”<|>“event”<|>“7月5日是一个具体的日期，可能与交通费用相关的事件。”)## (“entity”<|>“广州”<|>“geo”<|>“广州是一个地理位置，作为交通费用的起始地点。”)## (“entity”<|>“佛山”<|>“geo”<|>“佛山是一个地理位置，作为交通费用的目的地。”)## (“entity”<|>“城市内交通费”<|>“category”<|>“城市内交通费是一个费用类别，指的是在城市内部产生的交通费用。”)## (“relationship”<|>“广州”<|>“佛山”<|>“广州与佛山之间存在交通费用，表明两地之间的交通联系。”<|>“交通联系, 费用”<|>8)## (“relationship”<|>“7月5日”<|>“城市内交通费”<|>“7月5日与城市内交通费相关，表明在这一天产生了交通费用。”<|>“日期, 费用”<|>7)## (“content_keywords”<|>“交通费用, 地理位置关系, 日期事件”)<|COMPLETE|>'
    # (“entity”<|>“五洲花城”<|>“geo”<|>“五洲花城是一个地理位置，可能是一个社区或者城市区域。”)## (“entity”<|>“学区房”<|>“category”<|>“学区房是指位于学校附近，因教育资源而具有特定价值的房产。”)## (“relationship”<|>“五洲花城”<|>“学区房”<|>“五洲花城与学区房之间的关系是地理位置与房产类别的关联，表明在五洲花城这个地区寻找或讨论的是学区房。”<|>“地理位置, 房产类别”<|>8)## (“content_keywords”<|>“地理位置, 学区房, 房产信息”)<|COMPLETE|>
    
    final_result = sendGLM(input_text)
    
    return final_result



RAG.llm_model_func = call_LLM

if __name__ == "__main__":
    # RAG.llm_model_func("")
    print(sendGLM("你好"))