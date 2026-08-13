"""使用 Conversational Flow 啟動個人化旅遊規劃系統。"""

# ── 載入套件與專案模組 ──────────────────────────────

from datetime import date

from dotenv import load_dotenv

from crewai import Flow
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.skill_events import SkillUsedEvent
from crewai.events.utils.console_formatter import ConsoleFormatter
from crewai.experimental.conversational import ConversationState
from crewai.flow import listen

from crew import build_crew
from tools import start_mcp_tools

# 載入模型、MCP 與知識庫連線所需的環境變數
load_dotenv()


# ── 顯示 Skill 套用紀錄 ─────────────────────────────

@crewai_event_bus.on(SkillUsedEvent)
def show_skill_usage(_, event: SkillUsedEvent) -> None:
    """在終端機顯示本輪實際套用的 Skill 與執行 Agent。"""
    # 顯示 CrewAI 實際載入的 Skill 名稱
    print(f"\n── 套用 Skill：{event.skill_name}")

    # 顯示套用 Skill 的 Agent；事件未提供角色時顯示「未知 Agent」
    print(f"   執行 Agent：{event.agent_role or '未知 Agent'}")


# ── 建立多輪對話 Flow ────────────────────────────────

class TravelFlow(Flow[ConversationState]):
    """保存對話歷史，並將每輪問題交給 Hierarchical Crew。"""

    # 開啟 CrewAI 官方多輪對話功能，由 Flow 管理訊息與每輪執行狀態
    conversational = True

    # 保存 main() 啟動的 MCP 工具，供每輪建立的 Crew 共用
    mcp_tools: list

    def route_turn(self, context: dict) -> str:
        """所有旅遊問題都交給 Crew，工作流程由 Manager 動態判斷。"""
        return "travel"

    @listen("travel")
    def handle_travel(self) -> str:
        """建立本輪 Crew，帶入問題與最近兩輪對話後回傳最終答案。"""
        # Flow 已加入本輪 user 訊息，因此排除最後一則，只取前兩輪作為對話背景
        history = self.conversation_messages[:-1][-4:]
        conversation_context = "\n\n".join(
            f"{message['role']}：{message['content']}" for message in history
        ) or "（沒有先前對話。）"

        # 每輪建立新的 Crew，避免上一輪的 Manager 與 Task 狀態影響後續委派
        crew = build_crew(self.mcp_tools)

        # 清除上一輪完成事件，讓本輪重新等待 Crew Completion 面板
        ConsoleFormatter.crew_completion_printed.clear()

        # 將本輪問題、今天日期與 Flow 保存的對話背景交給通用 Task
        result = crew.kickoff(
            inputs={
                "query": self.state.current_user_message,
                "today": date.today().isoformat(),
                "conversation_context": conversation_context,
            }
        )

        # 等待完成面板輸出，避免它插入下一輪輸入提示
        ConsoleFormatter.crew_completion_printed.wait(timeout=10)
        return str(result)


# ── 啟動旅遊問答系統 ─────────────────────────────────

def main() -> None:
    """啟動 MCP 工具，並使用 Flow.chat() 進行多輪對話。"""
    # MCP Server 只啟動一次，後續每輪 Crew 共用相同工具連線
    adapter, tools = start_mcp_tools()

    try:
        # 顯示系統名稱與三個可直接測試的示範問題
        print(
            """
==================================================
個人化旅遊規劃 Agentic AI（CrewAI）已就緒
範例問題：
   1. 幫我安排下週二三天兩夜的大阪古蹟參訪行程
   2. 幫我把行程調整成以室內景點為主
   3. 請把剛剛的行程匯出成 Markdown
==================================================
"""
        )

        # Flow.chat() 負責輸入迴圈、對話歷史、空白輸入與離開指令
        TravelFlow(
            mcp_tools=tools,
            suppress_flow_events=True,
            tracing=False,
        ).chat(
            prompt="\n你：",
            assistant_prefix="\n旅遊助理：",
            exit_commands=("exit", "quit", "離開", "結束"),
        )
        print("\n👋 再見")
    finally:
        # 關閉 npx 啟動的 MCP 子程序，避免離開主程式後繼續留在背景
        adapter.stop()


if __name__ == "__main__":
    main()
