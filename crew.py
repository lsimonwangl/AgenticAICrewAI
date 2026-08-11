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
            "- 旅遊偏好分析師：需要分析使用者過往旅遊紀錄或個人偏好時使用。\n"
            "- 旅遊情報研究員：需要景點、住宿、交通、票價、天氣、匯率等"
            "外部或即時資訊時使用。\n"
            "- 個人化行程規劃師：只有需要整合偏好與情報、產出完整行程時使用。\n\n"
            "執行規則：\n"
            "1. 單一景點、天氣、匯率或交通問題，通常只需旅遊情報研究員。\n"
            "2. 個人偏好問題只需旅遊偏好分析師。\n"
            "3. 完整行程規劃應先取得必要的偏好與即時情報，再把和本題直接相關的"
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
            "9. 收到產出後由經理判斷下一步：候選資料或來源有問題就重新委派情報研究員；"
            "資料足夠但行程編排有問題就重新委派行程規劃師。最多修正兩輪，通過內部審查"
            "後才能回答；不得為了展示流程而重複委派。\n\n"
            "最後直接回答使用者原始問題。不要輸出內部委派過程、工作報告、"
            "退件報告或『請重新產出』等內部訊息。全文使用繁體中文；外文專有名詞"
            "可保留原文並附上中譯。"
        ),
        expected_output=(
            "一份只回答本輪問題、同時正確承接先前對話的繁體中文答案。"
            "所有可能變動的資訊都附有實際查詢來源；查不到的資訊明確標示需確認，"
            "不得臆測。若問題要求完整行程，答案應包含每日安排、交通、預算與注意事項。"
        ),
        # 不指定 agent：由 hierarchical manager 從全部 workers 中動態選擇。
    )

    return [user_request_task]


def build_crew(tools: list) -> Crew:
    manager, workers = build_agents(build_llm(), tools)
    return Crew(
        agents=workers,  # manager 不放進來，理由見 agents.build_agents
        tasks=build_tasks(),
        process=Process.hierarchical,
        manager_agent=manager,
        embedder=EMBEDDER,  # 沒自帶 embedder 的 agent 的 fallback
        verbose=True,
    )
