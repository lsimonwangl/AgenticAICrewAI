"""建立由 Manager 動態委派工作的 Hierarchical Crew。"""

# ── 載入套件與 Agent 建立函數 ───────────────────────

from crewai import Crew, Process, Task

from agents import build_agents, build_llm

# ── Task 行為規格 ────────────────────────────────────

# 通用 Task 接收本輪問題與對話背景，實際工作流程由 Manager 動態決定

USER_REQUEST_DESCRIPTION = (
    "今天是 {today}。使用者本輪的旅遊問題是：{query}\n\n"
    "先前對話背景如下：\n{conversation_context}\n\n"
    "本輪問題優先於先前內容；若本輪指涉先前內容，應依對話背景理解，"
    "最新修正覆蓋舊要求，也不要再次推薦已被否決的選項。"
    "依經理的動態委派與審查規則完成本輪需求，最後直接回答原始問題。"
    "不要輸出內部委派、審查或重試過程。全文使用繁體中文與純文字；"
    "只有使用者要求匯出 PDF 時，交給 PDF 轉換工具的內容可以使用 Markdown。"
)

USER_REQUEST_EXPECTED_OUTPUT = (
    "一份只回答本輪問題、同時正確承接先前對話的繁體中文答案。"
    "所有可能變動的資訊都附有實際查詢來源；查不到的資訊明確標示需確認，"
    "不得臆測。若問題要求完整行程，答案應包含每日安排、交通、預算與注意事項。"
    "終端回答使用純文字格式。PDF 匯出成功時，只輸出 exports/ 開頭的 .pdf 相對路徑。"
)

# ── 組裝 Hierarchical Crew ──────────────────────────

def build_crew(tools: list) -> Crew:
    """建立本輪要執行的 Hierarchical Crew。"""
    # 每輪建立新的共用 LLM、一位 Manager 與三位 Worker
    manager, workers = build_agents(build_llm(), tools)

    # 將通用 Task、Manager 與 Worker 組成 Hierarchical Crew
    return Crew(
        # agents 只放入可被委派的 Worker，Manager 由 manager_agent 指定
        agents=workers,
        # 每輪只有一張通用入口 Task，實際子工作由 Manager 現場判斷
        tasks=[
            Task(
                # description 會在 kickoff 時帶入日期、問題與先前對話
                description=USER_REQUEST_DESCRIPTION,
                # expected_output 定義最終答案必須符合的品質與格式
                expected_output=USER_REQUEST_EXPECTED_OUTPUT,
                # 不指定專責 Agent，由 Manager 依本輪問題動態選擇 Worker
            )
        ],
        # 使用 Hierarchical Process，讓 Manager 負責動態委派與整合
        process=Process.hierarchical,
        # 指定本輪負責管理三位 Worker 的 Manager Agent
        manager_agent=manager,
        # 在終端機顯示 Crew、Agent 與工具的完整執行過程
        verbose=True,
        # 保留 Agent 與工具執行畫面，只關閉可能插入下一輪輸入提示的背景 Trace
        tracing=False,
    )
