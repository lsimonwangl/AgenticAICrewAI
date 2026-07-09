"""crew.py — 組裝 hierarchical Crew。

- process=Process.hierarchical：啟用 manager 動態指派（預設是 sequential）。
- manager_agent=manager：使用自訂 manager；agents 只放五個 worker。
  （若把 manager 也放進 agents，會踩到「只有 manager 進 self.agents、
   找不到 coworker 只好自己做完」的問題。）
- memory：跨輪對話記憶。不能用 embedder dict 走 openai 相容協定指向 NVIDIA —
  crewai 底層是 chromadb 的 OpenAI client，傳不了 NVIDIA asymmetric embedding
  模型必需的 input_type 參數（會 400）。改傳 Memory 實例，embedder 用官方支援的
  「custom callable」直接接 NVIDIA（見 embed.py，不經 langchain）；llm 也指向
  NVIDIA（Memory 預設 gpt-5.4-mini 會要求 OPENAI_API_KEY）。
"""

from crewai import Crew, Process
from crewai.memory.unified_memory import Memory

from embed import embed


def _build_memory(llm) -> Memory:
    # 官方支援的 custom callable embedder，直接接 NVIDIA（見 embed.py）。
    # 記憶存/查都用 passage 向量，asymmetric 的 query/passage 區分先不做，檢索品質有感再說。
    # 不清存檔：官方 memory 本意就是跨 session 持久化（要重置就手動刪 memory 資料夾）。
    return Memory(
        embedder=lambda texts: embed(list(texts), input_type="passage"),
        llm=llm,
    )


def build_crew(manager, workers, task, llm) -> Crew:
    return Crew(
        agents=workers,           # 只放 worker，manager 不放進來
        tasks=[task],
        process=Process.hierarchical,
        manager_agent=manager,    # 自訂 manager
        memory=_build_memory(llm),
        tracing=True,  # CrewAI 自家 tracing：托管的執行時間軸/token/成本視圖（首次跑會要登入 CrewAI 帳號）
        skills=["skills/common"],  # 全員共用的 skills；角色專屬的在 agents.py 各自掛載
        max_rpm=20,  # 全 crew 共用的節流：超過每分鐘 20 次 LLM 呼叫就等待，避開 NVIDIA 免費額度 429
        output_log_file="output/執行過程.json",  # 完整過程：每個 agent 的任務與產出（.json 結尾存結構化格式）
        verbose=True,
    )
