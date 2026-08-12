"""多輪對話主程式。

執行流程：
    1. 載入 .env（必須在其他模組 import 之前，它們在載入時就會讀環境變數）
    2. 啟動一次 MCP 工具
    3. 用 while 迴圈接收多輪輸入，並用 list 保存歷史
    4. 每一輪建立新的 hierarchical Crew，避免保留上一輪 Task 執行狀態
    5. 離開對話時關掉 MCP 子程序

跑法：venv\\Scripts\\Activate.ps1 後在專案目錄下 python main.py
（一定要在專案目錄下執行，knowledge/ 是相對路徑）
"""

from datetime import date
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from crewai.events.event_bus import crewai_event_bus  # noqa: E402
from crewai.events.types.skill_events import SkillUsedEvent  # noqa: E402
from crewai.events.utils.console_formatter import ConsoleFormatter  # noqa: E402

from crew import build_crew  # noqa: E402
from tools import start_mcp_tools  # noqa: E402

WEEKDAYS = ["一", "二", "三", "四", "五", "六", "日"]
MAX_HISTORY_MESSAGES = 4  # 最近兩輪 user/assistant 訊息
CREW_COMPLETION_TIMEOUT_SECONDS = 10
EXPORTS_DIR = Path(__file__).resolve().parent / "exports"

SKILL_LABELS = {
    "itinerary-markdown-exporter": "旅遊行程 Markdown 匯出",
}


@crewai_event_bus.on(SkillUsedEvent)
def show_skill_usage(_, event: SkillUsedEvent) -> None:
    """把條件式 Skill 的實際載入顯示在終端機，方便課堂觀察。"""
    label = SKILL_LABELS.get(event.skill_name, event.skill_name)
    agent_role = event.agent_role or "未知 Agent"
    print(f"\n── 套用 Skill：{label}（{event.skill_name}）")
    print(f"   執行 Agent：{agent_role}")


def snapshot_markdown_exports() -> dict[Path, tuple[int, int]]:
    """記錄現有 Markdown 匯出檔，供本輪結束後判斷是否真的寫檔。"""
    if not EXPORTS_DIR.exists():
        return {}
    return {
        path.resolve(): (path.stat().st_mtime_ns, path.stat().st_size)
        for path in EXPORTS_DIR.glob("*.md")
        if path.is_file()
    }


def find_changed_export(before: dict[Path, tuple[int, int]]) -> Path | None:
    """取得本輪新增或更新的 Markdown；沒有實際寫檔時回傳 None。"""
    after = snapshot_markdown_exports()
    changed = [path for path, state in after.items() if before.get(path) != state]
    return max(changed, key=lambda path: after[path][0], default=None)


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
   3. 請把剛剛的行程匯出成 Markdown，檔名叫大阪三日遊
==================================================
"""
        )

        turn = 1
        while True:
            try:
                query = input(f"\n[第 {turn} 輪] 你：").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if query.lower() in {"exit", "quit", "離開", "結束"}:
                break
            if not query:
                continue

            today = date.today()
            exports_before = snapshot_markdown_exports()
            # 每輪建立全新的 Crew，避免 hierarchical Task 保留上一輪的
            # manager/執行狀態，導致後續輪次無法委派給 worker Agent。
            crew = build_crew(tools)
            ConsoleFormatter.crew_completion_printed.clear()
            result = crew.kickoff(
                inputs={
                    "query": query,
                    "today": f"{today:%Y-%m-%d}（星期{WEEKDAYS[today.weekday()]}）",
                    "conversation_context": "\n\n".join(
                        history[-MAX_HISTORY_MESSAGES:]
                    )
                    or "（第一輪對話，沒有先前內容。）",
                }
            )

            # CrewAI 的 verbose 完成面板可能比 kickoff() 稍晚輸出；先等面板完成，
            # 才印本輪回答並進入下一次 input()，避免面板插進使用者提示。
            ConsoleFormatter.crew_completion_printed.wait(
                timeout=CREW_COMPLETION_TIMEOUT_SECONDS
            )

            exported_file = find_changed_export(exports_before)
            if exported_file is not None:
                # 寫檔結果以實際檔案為準，匯出輪不再把 Markdown 全文印到終端。
                answer = exported_file.relative_to(EXPORTS_DIR.parent).as_posix()
            else:
                answer = str(result)
            print(f"\n旅遊助理：{answer}")
            history.extend([f"user：{query}", f"assistant：{answer}"])
            print("\n✅ 本輪規劃完成，可以繼續提問。")
            turn += 1

        print("\n👋 再見")
    finally:
        # npx 起的是本機子程序，不關掉會留在背景
        adapter.stop()


if __name__ == "__main__":
    main()
