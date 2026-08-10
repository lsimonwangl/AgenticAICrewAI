"""主程式。

執行流程：
    1. 載入 .env（必須在其他模組 import 之前，它們在載入時就會讀環境變數）
    2. 讀取使用者輸入
    3. 啟動 MCP 工具
    4. 組裝 Crew 並執行
    5. 不論成功失敗都關掉 MCP 子程序

跑法：venv\\Scripts\\Activate.ps1 後在專案目錄下 python main.py
（一定要在專案目錄下執行，knowledge/ 是相對路徑）
"""

from datetime import date

from dotenv import load_dotenv

load_dotenv()

from crew import build_crew  # noqa: E402
from tools import start_mcp_tools  # noqa: E402

# 星期幾要一起給，否則 LLM 無法把「下週二」換算成實際日期
WEEKDAYS = ["一", "二", "三", "四", "五", "六", "日"]


def main() -> None:
    query = input("請輸入你的旅遊需求：").strip()
    if not query:
        print("沒有輸入需求，結束。")
        return

    today = date.today()
    today_str = f"{today:%Y-%m-%d}（星期{WEEKDAYS[today.weekday()]}）"

    adapter, tools = start_mcp_tools()
    try:
        result = build_crew(tools).kickoff(
            inputs={"query": query, "today": today_str}
        )
    finally:
        # npx 起的是本機子程序，不關掉會留在背景
        adapter.stop()

    print("\n" + "=" * 60)
    print("處理完成")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    main()
