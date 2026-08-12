"""Crew 組裝。

crew.py 定義通用入口 task，並把它與 agents.py 的 agent 組成 hierarchical Crew。

通用 task 刻意不綁 agent。hierarchical manager 會根據使用者問題，從三位 worker
中動態選擇必要的專員、安排委派順序並整合結果。代價是多一層 LLM 呼叫，而且
每次執行的路徑不保證相同。

此模組提供 build_crew() 供 main.py 呼叫。
"""

from crewai import Crew, Process, Task

from agents import EMBEDDER, build_agents, build_llm


def build_tasks() -> list[Task]:
    """建立不綁定 agent 的通用入口 task，交由 manager 動態委派。"""
    user_request_task = Task(
        description=(
            "今天是 {today}。使用者本輪的旅遊問題是：{query}\n\n"
            "先前對話背景如下：\n{conversation_context}\n\n"
            "本輪問題的優先級最高。若出現『那間』、『第二天』、『改便宜一點』"
            "等指涉或修改要求，應根據先前對話理解對象；使用者最新的修正會覆蓋"
            "先前要求。不得再次推薦先前已被使用者否決的選項。\n\n"
            "先分析使用者真正想完成的事情，再判斷需要哪些資訊與專員。"
            "只委派必要的專員，不要固定呼叫全部專員。\n\n"
            "可用專員與使用時機：\n"
            "- 旅遊偏好分析師：行程規劃或推薦問題必須使用，以過往台灣旅遊紀錄推導偏好。\n"
            "- 旅遊情報研究員：需要景點、住宿、交通、票價、天氣、匯率等"
            "外部或即時資訊時使用。\n"
            "- 個人化行程規劃師：需要整合偏好與情報產出完整行程，或把先前完成的"
            "行程匯出成 Markdown 文件時使用。\n\n"
            "執行規則：\n"
            "1. 單一景點、天氣、匯率或交通問題，通常只需旅遊情報研究員。\n"
            "2. 個人偏好問題只需旅遊偏好分析師。\n"
            "3. 行程規劃或景點、住宿、區域與活動推薦，必須先取得偏好與即時情報；"
            "不需要使用者特別說『依我的偏好』。再把和本題直接相關的"
            "結論、數字與來源交給個人化行程規劃師。\n"
            "4. 若日期、預算等關鍵條件未提供，不得自行假定；應在答案中清楚說明"
            "缺少的條件，或提供不依賴該條件的有限回答。\n"
            "5. 價格、營業時間、票價、天氣與匯率等可能變動的資訊，必須交給"
            "旅遊情報研究員實際查詢並附來源。\n"
            "6. 不得要求使用者重新提供系統 knowledge 中已有的旅遊紀錄；需要時"
            "應委派旅遊偏好分析師查詢。\n"
            "7. 專員的中間產出只保留本題需要的結論、必要數字、最短依據與來源；"
            "不得貼出工具原始回傳、長篇知識庫內容、搜尋過程或重複資訊。\n"
            "8. 若偏好會影響外部搜尋，必須先完成偏好分析，再把偏好放進情報研究員的"
            "委派內容；兩者不得平行。研究結果應指出主要候選項目與偏好的符合或衝突。\n"
            "9. 完整行程或推薦完成後，經理必須依核心規則確認品質；"
            "最多修正一輪，通過內部審查後才能回答，不得為了展示流程而重複委派。\n\n"
            "10. 使用者明確要求把先前完成的行程儲存、匯出或整理成 Markdown 文件時，"
            "只把最近一次完成版行程交給個人化行程規劃師；不得重新分析偏好、搜尋情報或"
            "重新規劃。委派時要求規劃師把完整 Markdown 寫入檔案，寫檔後只回傳"
            "exports/ 開頭的相對路徑，不得回傳 Markdown 內文。匯出成功時，經理的最終"
            "回答也只能是該相對路徑，不得重印文件或增加說明。如果先前對話沒有完成版"
            "行程，直接說明目前沒有可匯出的內容。\n\n"
            "最後直接回答使用者原始問題。不要輸出內部委派過程、工作報告、"
            "退件報告或『請重新產出』等內部訊息。全文使用繁體中文；外文專有名詞"
            "可保留原文並附上中譯。專員中間產出與終端回答一律使用純文字和換行，"
            "不得使用 Markdown 標題、表格、粗體或項目符號；需要列舉時使用阿拉伯數字。"
            "只有使用者明確要求匯出時，寫入 exports 的 .md 檔案內容可以使用 Markdown。"
        ),
        expected_output=(
            "一份只回答本輪問題、同時正確承接先前對話的繁體中文答案。"
            "所有可能變動的資訊都附有實際查詢來源；查不到的資訊明確標示需確認，"
            "不得臆測。若問題要求完整行程，答案應包含每日安排、交通、預算與注意事項。"
            "終端回答使用純文字格式；只有匯出的 .md 檔案使用 Markdown。匯出成功時"
            "只輸出 exports/ 開頭的 .md 相對路徑。"
        ),
        # 不指定 agent：由 hierarchical manager 從全部 workers 中動態選擇。
    )

    return [user_request_task]


def build_crew(tools: list) -> Crew:
    llm = build_llm()
    manager, workers = build_agents(llm, tools)
    return Crew(
        agents=workers,  # manager 不放進來，理由見 agents.build_agents
        tasks=build_tasks(),
        process=Process.hierarchical,
        manager_agent=manager,
        embedder=EMBEDDER,  # 沒自帶 embedder 的 agent 的 fallback
        verbose=True,
        # 保留 Agent/工具的 verbose 畫面，只關閉背景 Trace 批次訊息，
        # 避免它在下一輪 input() 等待時插入終端。
        tracing=False,
    )
