"""rag.py — 偏好檢索（原生 pymilvus，不經 langchain）。

讀取過往台灣旅遊紀錄、切塊、以 NVIDIA embedding 向量化存入 Milvus，並提供檢索器。
再把檢索器包成 CrewAI 工具，交給「偏好分析師」agent 於執行時呼叫（tool 回傳偏好原文片段）。

沿用 Lab 4 的 Milvus，但改用官方 pymilvus 的 MilvusClient，不再依賴 langchain-milvus /
langchain-nvidia-ai-endpoints / langchain-community / langchain-text-splitters。
"""

import os
from pathlib import Path
from typing import Any, Type

from pydantic import BaseModel, Field
from pymilvus import MilvusClient
from crewai.tools import BaseTool

from embed import embed


def _chunk(text: str, size: int = 256, overlap: int = 50) -> list[str]:
    # ponytail: 純字元滑窗，近似 langchain RecursiveCharacterTextSplitter。
    # 旅遊紀錄是短段落，不做遞迴分隔符切分；要更精準的語意邊界再換。
    step = size - overlap
    chunks: list[str] = []
    for i in range(0, len(text), step):
        chunks.append(text[i:i + size])
        if i + size >= len(text):  # 已覆蓋到結尾就停，避免尾段被漏掉
            break
    return chunks or [text]


class _Retriever:
    """薄包裝：持有 MilvusClient 與 collection，query() 回傳 Top-K 偏好原文片段。"""

    def __init__(self, client: MilvusClient, collection: str, top_k: int):
        self.client, self.collection, self.top_k = client, collection, top_k

    def query(self, text: str) -> list[str]:
        qvec = embed([text], input_type="query")[0]  # 查詢用 query 向量（asymmetric）
        hits = self.client.search(
            collection_name=self.collection,
            data=[qvec],
            limit=self.top_k,
            output_fields=["text"],
        )
        return [h["entity"]["text"] for h in hits[0]]


def build_retriever(data_dir: str = "./data",
                    collection: str = "travel_preferences",
                    top_k: int = 5) -> _Retriever:
    """讀取旅遊紀錄、切塊、向量化存入 Milvus，回傳 Top-K 檢索器。"""
    texts = [p.read_text(encoding="utf-8") for p in Path(data_dir).glob("*.txt")]
    chunks = [c for t in texts for c in _chunk(t)]
    vectors = embed(chunks, input_type="passage")  # 文件用 passage 向量

    client = MilvusClient(uri=os.environ.get("MILVUS_URI", "http://localhost:19530"))
    if client.has_collection(collection):
        client.drop_collection(collection)  # 每次啟動重建，與 Lab 4 一致，避免重複匯入
    client.create_collection(collection_name=collection, dimension=len(vectors[0]))
    client.insert(collection_name=collection, data=[
        {"id": i, "vector": v, "text": c}
        for i, (v, c) in enumerate(zip(vectors, chunks))
    ])
    return _Retriever(client, collection, top_k)


class _PreferenceQuery(BaseModel):
    query: str = Field(
        ...,
        description="要查詢的偏好主題，如住宿風格、景點類型、飲食偏好、預算或旅遊步調",
    )


class PreferenceSearchTool(BaseTool):
    """把 RAG 檢索器包成 CrewAI 工具。tool 回傳偏好原文片段，不做推斷。"""

    name: str = "search_travel_preferences"
    description: str = (
        "從使用者過往台灣旅遊紀錄檢索個人偏好。輸入要查詢的偏好主題，"
        "回傳最相關的偏好原文片段，供你歸納使用者的旅行風格。"
    )
    args_schema: Type[BaseModel] = _PreferenceQuery
    retriever: Any = None

    def _run(self, query: str) -> str:
        docs = self.retriever.query(query)
        if not docs:
            return "查無相關偏好紀錄。"
        return "\n\n".join(f"[紀錄 {i + 1}] {d}" for i, d in enumerate(docs))


if __name__ == "__main__":  # 自我檢查 _chunk：切片不超長、且完整覆蓋到結尾不漏尾段
    assert _chunk("short") == ["short"]
    c = _chunk("x" * 600)
    assert all(len(x) <= 256 for x in c), c
    for text in ("a" * 300, "b" * 256, "c" * 601):  # 逐字元檢查覆蓋率
        covered = set()
        for j, seg in enumerate(_chunk(text)):
            start = j * (256 - 50)
            covered |= set(range(start, start + len(seg)))
        assert covered == set(range(len(text))), f"尾段被漏掉: len={len(text)}"
    print("chunk OK:", [len(x) for x in c])
