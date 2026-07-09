"""embed.py — NVIDIA NIM embedding（直接用 openai 相容端點，不經 langchain）。

rag.py 與 crew.py 共用。NVIDIA 的 embedding 端點吃兩個非標準參數：
- input_type：asymmetric 模型必需，存文件用 "passage"、查詢用 "query"
- truncate：超過模型 512 token 上限時由伺服器端截斷（"END"），而非回 400
兩者以 openai SDK 的 extra_body 帶入。金鑰沿用 NVIDIA_API_KEY（第一把）。
"""

import os
from functools import lru_cache

from openai import OpenAI


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["NVIDIA_API_KEY"],
        base_url=os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
    )


def embed(texts: list[str], input_type: str = "passage", batch: int = 50) -> list[list[float]]:
    # ponytail: 每次 50 筆分批送，避免大語料超過端點單次批次上限；量大再調
    texts = list(texts)
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        resp = _client().embeddings.create(
            model=os.environ["EMBEDDING_MODEL"],
            input=texts[i:i + batch],
            extra_body={"input_type": input_type, "truncate": "END"},
        )
        out.extend(d.embedding for d in resp.data)
    return out
