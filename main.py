"""main.py — 主程式入口。

依序：載入環境變數 → 建立 LLM → 建立 RAG 檢索工具 → 啟動三個 MCP server →
建立 manager 與五個 worker → 組裝 hierarchical Crew → 啟動終端對話迴圈。
結束時於 finally 關閉 MCP server。
"""

import os

from dotenv import load_dotenv
from crewai import LLM

from rag import build_retriever, PreferenceSearchTool
from tools import start_mcp_adapters
from agents import build_agents
from tasks import build_task
from crew import build_crew
from chat import run_chat
from key_rotation import install_key_rotation


def _api_keys() -> list[str]:
    # 收集 NVIDIA_API_KEY1, NVIDIA_API_KEY2, ...（連號到斷為止）輪流用
    keys, i = [], 1
    while (k := os.environ.get(f"NVIDIA_API_KEY{i}")):
        keys.append(k.strip())
        i += 1
    # 退回：逗號分隔 NVIDIA_API_KEYS，或單把 NVIDIA_API_KEY
    if not keys:
        raw = os.environ.get("NVIDIA_API_KEYS") or os.environ["NVIDIA_API_KEY"]
        keys = [k.strip() for k in raw.split(",") if k.strip()]
    return keys


def build_llm() -> LLM:
    keys = _api_keys()
    install_key_rotation(keys)  # 每次 LLM 呼叫 round-robin 換 key，分散 429
    # openai/ 前綴讓 litellm 走 OpenAI 相容端點；base_url 指向 NVIDIA NIM
    return LLM(
        model=f"openai/{os.environ['CHAT_MODEL']}",
        base_url=os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        api_key=keys[0],  # 兜底：拷貝遺失 params 時用第一把
        temperature=0.3,
        # 調低：litellm 的重試在單次呼叫內綁同一把 key，429 時重試幾乎必再 429，
        # 只是白燒退避預算。設 1（共 2 次）失敗即返回，讓下一次呼叫 round-robin 換到新 key。
        num_retries=1,
        # NVIDIA 會無預警下架模型（跑到一半 404 Function not found）；
        # 主模型失敗時 litellm 自動改打備用模型，沿用同一組 base_url / api_key
        fallbacks=[f"openai/{os.environ['FALLBACK_MODEL']}"] if os.environ.get("FALLBACK_MODEL") else None,
    )


def main() -> None:
    load_dotenv()
    # embedding（crew.py/rag.py）與下方兜底仍讀單把 NVIDIA_API_KEY，用第一把回填
    os.environ.setdefault("NVIDIA_API_KEY", _api_keys()[0])
    # crewai 內部（如 Guardrail Agent）複製 LLM 時會遺失 api_key 與 base_url 參數；
    # litellm 找不到參數時會退回讀環境變數，這裡補上兜底讓它仍指向 NVIDIA
    os.environ.setdefault("OPENAI_API_KEY", os.environ["NVIDIA_API_KEY"])
    os.environ.setdefault("OPENAI_API_BASE", os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"))

    llm = build_llm()
    rag_tool = PreferenceSearchTool(retriever=build_retriever())

    adapters, bundles = start_mcp_adapters()
    try:
        manager, workers = build_agents(
            llm,
            rag_tool,
            bundles["tavily"],
            bundles["open_meteo"],
            bundles["frankfurter"],
        )
        crew = build_crew(manager, workers, build_task(llm), llm)
        run_chat(crew)
    finally:
        for adapter in adapters:
            try:
                adapter.stop()
            except Exception:
                pass


if __name__ == "__main__":
    main()
