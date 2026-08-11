"""多輪對話主程式。

執行流程：
    1. 載入 .env（必須在其他模組 import 之前，它們在載入時就會讀環境變數）
    2. 啟動一次 MCP 工具
    3. 用 while 迴圈接收多輪輸入，並用 list 保存歷史
    4. 每一輪建立新的 hierarchical Crew
    5. 離開對話時關掉 MCP 子程序

跑法：venv\\Scripts\\Activate.ps1 後在專案目錄下 python main.py
（一定要在專案目錄下執行，knowledge/ 是相對路徑）
"""

from datetime import date

from dotenv import load_dotenv

load_dotenv()

from crew import build_crew  # noqa: E402
from tools import start_mcp_tools  # noqa: E402

WEEKDAYS = ["一", "二", "三", "四", "五", "六", "日"]


def main() -> None:
    adapter, tools = start_mcp_tools()
    history: list[str] = []
    try:
        print(
            """
==================================================
個人化旅遊規劃 Agentic AI（CrewAI）已就緒
範例問題：
   1. 幫我安排下週二三天兩夜的大阪古蹟參訪行程
   2. 幫我把第二天改成以室內景點為主
   3. 依照我的旅遊偏好，推薦適合我的京都住宿區域
==================================================
"""
        )

        while True:
            try:
                query = input("\n你：").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if query.lower() in {"exit", "quit", "離開", "結束"}:
                break
            if not query:
                continue

            today = date.today()
            result = build_crew(tools).kickoff(
                inputs={
                    "query": query,
                    "today": f"{today:%Y-%m-%d}（星期{WEEKDAYS[today.weekday()]}）",
                    "conversation_context": "\n\n".join(history)
                    or "（第一輪對話，沒有先前內容。）",
                }
            )

            answer = str(result)
            print(f"\n旅遊助理：{answer}")
            history.extend([f"user：{query}", f"assistant：{answer}"])
    finally:
        # npx 起的是本機子程序，不關掉會留在背景
        adapter.stop()


if __name__ == "__main__":
    main()
