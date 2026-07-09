"""rag.py — 偏好檢索。

沿用 Lab 4 的 Milvus 向量庫做法：讀取過往台灣旅遊紀錄、切塊、以 NVIDIA
embedding 向量化存入 Milvus，並提供一個檢索器。再把檢索器包成 CrewAI 工具，
交給「偏好分析師」agent 於執行時呼叫（tool 回傳的是偏好原文片段，供 agent 推斷）。
"""

import os
from typing import Any, Type

from pydantic import BaseModel, Field

from crewai.tools import BaseTool

from langchain_milvus import Milvus
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def build_retriever(data_dir: str = "./data",
                    collection: str = "travel_preferences",
                    top_k: int = 5):
    """讀取旅遊紀錄、切塊、向量化存入 Milvus，回傳 Top-K 檢索器。"""
    embeddings = NVIDIAEmbeddings(
        model=os.environ["EMBEDDING_MODEL"],
        api_key=os.environ["NVIDIA_API_KEY"],
        base_url=os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
    )

    loader = DirectoryLoader(
        data_dir, glob="*.txt",
        loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    store = Milvus.from_documents(
        chunks, embeddings,
        collection_name=collection,
        connection_args={"uri": os.environ.get("MILVUS_URI", "http://localhost:19530")},
        drop_old=True,   # 每次啟動重建，避免重複匯入；與 Lab 4 from_documents 一致
    )
    return store.as_retriever(search_kwargs={"k": top_k})


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
        docs = self.retriever.invoke(query)
        if not docs:
            return "查無相關偏好紀錄。"
        return "\n\n".join(
            f"[紀錄 {i + 1}] {d.page_content}" for i, d in enumerate(docs)
        )
