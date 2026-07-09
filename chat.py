"""chat.py — 多輪對話迴圈。

每輪把使用者需求與當天日期作為 inputs 傳入 crew.kickoff。Crew 的 verbose 已負責
在終端顯示各 agent 的委派、工具呼叫與推理過程，因此不需自訂串流。
memory=True 讓同一個 crew 實例在輪次間保留脈絡。
"""

from datetime import date


def read_query() -> str:
    try:
        return input("\n請輸入旅遊需求（輸入 exit 結束）：\n> ").strip()
    except (EOFError, KeyboardInterrupt):
        return "exit"


def run_chat(crew) -> None:
    print("個人化旅遊規劃 Multi-Agent 系統已啟動。")
    while True:
        question = read_query()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("結束。")
            break

        result = crew.kickoff(inputs={
            "user_request": question,
            "today": str(date.today()),
        })

        print("\n========== 規劃結果 ==========\n")
        print(result)
