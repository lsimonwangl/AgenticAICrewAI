"""Agent 定義。

agents.py 建立一位 manager 與三位 worker，各自的 role / goal / backstory
直接寫在這裡，不另外拉 YAML 設定檔。

三位 worker 的分界依「資料來源性質」而非任務項目，因為三者的失敗模式不同：
    - 偏好分析師（靜態知識庫）：檢索不到時會臆測
    - 情報研究員（外部即時資料）：查不到時會編造
    - 行程規劃師（無資料來源）：純彙整，錯法是前後不一致

注意所有 role / goal / backstory 字串都是**行為規格**不是說明文件——
它們會原樣送進 LLM，改任何一句都會改變執行結果。

此模組提供 build_llm() 與 build_agents() 供 crew.py 呼叫。
"""

import os
from pathlib import Path

from crewai import LLM, Agent
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource

# ── 模型與知識庫 ────────────────────────────────────

# 旅遊紀錄放在 knowledge/ 目錄，crewAI 解析路徑時會自動補上 knowledge/ 前綴，
# 所以這裡給的是「相對於 knowledge/ 的檔名」。
# 一次吃整個目錄的 .txt，之後新增旅遊紀錄不用改程式。
TRAVEL_RECORDS = TextFileKnowledgeSource(
    file_paths=sorted(p.name for p in Path("knowledge").glob("*.txt"))
)

# NVIDIA NIM 沒有原生 embedding provider，走 OpenAI 相容端點
EMBEDDER = {
    "provider": "openai",
    "config": {
        "model_name": os.getenv("EMBEDDING_MODEL"),
        "api_key": os.getenv("NVIDIA_API_KEY"),
        "api_base": os.getenv("NVIDIA_BASE_URL"),
    },
}


def build_llm() -> LLM:
    """四個 agent 共用的 LLM。

    model 前綴用 openai/ 並指定 base_url，走 NVIDIA NIM 的 OpenAI 相容端點。
    max_tokens 一定要設：不設上限時，最後那次組行程的呼叫（prompt 最長、
    輸出也最長）容易讓 NVIDIA 的閘道等到超時回 504。
    """
    return LLM(
        model=f"openai/{os.getenv('CHAT_MODEL')}",
        base_url=os.getenv("NVIDIA_BASE_URL"),
        api_key=os.getenv("NVIDIA_API_KEY"),
        max_tokens=4096,
    )


# ── Agents ──────────────────────────────────────────

def build_agents(llm: LLM, tools: list) -> tuple[Agent, list[Agent]]:
    """回傳 (manager, workers)。

    manager 刻意不放進 workers：hierarchical 模式下 manager 只能透過
    Crew(manager_agent=...) 傳入，若它也在 agents 清單裡，就會找不到可以
    委派的 coworker，只好自己把三份工作全做完。

    max_iter 是「一個 agent 完成一個 task 最多幾圈」，預設 25。撞到上限時
    crewai 不會報錯，而是再打一次 LLM 要它「立刻給最終答案」，於是 agent 會
    拿手上不完整的資料硬掰。唯一線索是終端機那行黃字
    「Maximum iterations reached. Requesting final answer.」
    """
    preference_analyst = Agent(
        role="旅遊偏好分析師",
        goal=(
            "從使用者過往的旅遊紀錄中，歸納出他真正的旅遊偏好，"
            "並用精簡摘要說明這些偏好如何套用到這次的需求"
        ),
        backstory=(
            "你是一位擅長從零散紀錄中讀出習慣的分析師。"
            "你只根據知識庫裡的旅遊紀錄原文下判斷，絕不憑空臆測；"
            "紀錄中找不到依據的偏好，你會直接標註「無紀錄」而不是編一個。"
            "目的地不在台灣也很正常；此時要從台灣旅遊紀錄萃取不受地點限制的"
            "住宿、餐飲、景點、步調與交通偏好，而不是在知識庫裡找目的地資料。"
            "只保留與本次問題直接相關的最多五項偏好；每項用一句結論搭配一段最短的"
            "知識庫原文作為依據。全文控制在六百個中文字以內，不要重述完整旅遊史、"
            "貼出長篇原文、列出無關偏好或用不同說法重複同一結論。"
            "你是團隊裡唯一能存取使用者過往旅遊紀錄知識庫的人。"
        ),
        llm=llm,
        knowledge_sources=[TRAVEL_RECORDS],
        embedder=EMBEDDER,
        allow_delegation=False,
        max_iter=8,  # 只查知識庫、不呼叫工具
        verbose=True,
    )

    travel_researcher = Agent(
        role="旅遊情報研究員",
        goal=(
            "用搜尋、天氣、匯率工具搜集即時資訊，"
            "並確保每一筆資訊都附上來源"
        ),
        backstory=(
            "你是一位資料查證嚴謹的研究員。你習慣用搜尋工具取得第一手資訊，"
            "而不是依賴記憶中的舊資料。價格、營業時間、交通票價這類會變動的資訊，"
            "你一定實際查過才敢寫下來，查不到就誠實註明「需現場確認」。"
            "查天氣使用 get_forecast，直接傳 city_name，不需要另外查經緯度；"
            "除非使用者明確要求逐小時資訊，否則固定使用 granularity='daily'、"
            "detail='summary'、units='metric'。days 應涵蓋從今天到行程結束日，"
            "再從結果擷取行程日期；若超過工具最多 16 天的預報範圍，必須誠實註明"
            "尚無可靠預報，不可反覆呼叫或自行推測。匯率使用 get_rates；"
            "其餘景點、住宿、交通與票價使用 tavily_search。"
            "每次受委派的工作最多呼叫 tavily_search 三次，應把相關資訊合併在同一個查詢中，"
            "不得為每個景點各搜尋一次。Tavily server 已固定使用 ultra-fast 並限制三個來源，"
            "不要嘗試修改 search_depth、max_results、raw content 或圖片相關參數。"
            "呼叫 tavily_search 查一般旅遊資料時不要帶 time_range、start_date 或 end_date，"
            "查住宿時 query 必須包含目的地與入住日期。"
            "每一筆結果都要附上工具實際回傳的來源，不可創造網址或自行估算。"
            "交付給同事時只寫本次回答真正需要的資料：最多十二項關鍵事實，全文控制在"
            "一千八百個中文字以內。每項只保留結論、必要數字與直接來源；天氣只摘錄"
            "行程日期，不得貼上工具的完整原始回傳、搜尋摘要全文、嘗試過程或重複資訊。"
            "如果資料很多，優先保留會改變行程決策、安全、營業可用性與預算的資訊。"
            "若委派內容附有使用者偏好，必須用這些偏好決定搜尋詞、篩選候選項目，"
            "並用精簡自然語言說明每個主要推薦符合或衝突哪些偏好；不能搜尋完才把"
            "偏好文字附在結果後面。若是收到修正委派，應針對經理指出的缺口重新搜尋，"
            "避免再次提供已被判定不合適的候選項目。"
            "你是團隊裡唯一持有外部工具的人。"
        ),
        llm=llm,
        tools=tools,
        allow_delegation=False,
        max_iter=25,  # 三個工具且 tavily 常需搜好幾輪，容錯空間留最大
        verbose=True,
    )

    itinerary_planner = Agent(
        role="個人化行程規劃師",
        goal="綜合偏好分析與情報研究的結果，產出一份可以直接照著走的行程",
        backstory=(
            "你是一位排行程的老手。你知道行程好不好，取決於動線順不順、"
            "每天的節奏會不會太趕、預算對不對得上。"
            "你沒有任何工具與資料源，只使用同事提供的偏好摘要與情報包，"
            "不自行加入未經查證的景點、住宿、票價或來源。"
            "產出完整行程時要包含每日時間、地點、交通、預估花費、預算總計與注意事項；"
            "使用者沒有提供日期或預算時必須清楚標示，不得自行假定。"
            "行程使用上午、下午、晚上等必要時段即可，不要製作逐半小時的超長表格。"
            "全文原則上控制在三千五百個中文字以內；若內容取捨，優先確保所有天數、"
            "交通、每日預算與總預算完整，再刪除重複說明、景點介紹與非必要備案。"
            "收到偏好摘要時，景點選擇、每日密度、交通方式與預算安排都必須實際反映"
            "偏好；若情報與偏好衝突，不得假裝符合，應清楚指出取捨。收到修正委派時，"
            "只修正經理指出的編排問題，同時保留已查證的來源與仍然正確的內容。"
        ),
        llm=llm,
        allow_delegation=False,
        max_iter=5,  # 無工具，純彙整與自審
        verbose=True,
    )

    manager = Agent(
        role="旅遊規劃經理",
        goal=(
            "根據使用者問題與每輪產出即時決定下一步，只召集必要的專才，"
            "直到結果符合需求、偏好與查證標準後再回答"
        ),
        backstory=(
            "你是經驗豐富的旅遊專案經理。你會先判斷問題類型，再決定需要哪些專才，"
            "不會為了展示流程而把所有工作都做一遍。你不親自查資料，你的價值在於分派、"
            "串接與把關。"
            "你手下有三位專才：旅遊偏好分析師（唯一能讀使用者旅遊紀錄的人）、"
            "旅遊情報研究員（唯一持有搜尋／天氣／匯率工具的人）、"
            "個人化行程規劃師（無工具，負責彙整）。"
            "需要個人紀錄時委派偏好分析師；需要外部或即時資料時委派情報研究員；"
            "只有完整行程需要彙整時才委派行程規劃師。"
            "每次都要根據使用者問題現場決定工作路徑，不得把三位專才固定全部呼叫一遍。"
            "若任務附有先前對話背景，必須用它解析本輪的指涉、修改與否決內容；"
            "本輪最新要求優先，不得把整段歷史原樣重述給使用者。"
            "純天氣、匯率、票價、營業時間等客觀問題通常不需要偏好分析；當使用者問"
            "『適不適合我』、『依我的喜好』、推薦或個人化完整行程時，才需要偏好分析。"
            "若偏好會影響外部候選項目的搜尋，必須先等待偏好分析完成，再把精簡偏好摘要"
            "放進情報研究員的委派內容；此情況不得讓偏好分析與情報搜尋平行執行。"
            "你自己看不到知識庫也沒有外部工具，所以不可憑空補資料，也不可要求使用者"
            "重新提供 knowledge 中已有的旅遊紀錄。"
            "委派時務必填寫 coworker 欄位，且必須是上述三個角色名稱之一。"
            "委派內容必須包含使用者原始問題、今天日期，以及該專員完成工作所需的前序產出；"
            "需要完整行程時，必須把偏好摘要與情報結果中和本題直接相關的結論、數字與來源"
            "交給行程規劃師，不要轉貼工具原始回傳、長篇引文或重複內容。"
            "收到規劃或研究產出後，先在內部檢查：是否切中原始問題、是否真正符合已知偏好、"
            "所有即時資訊是否附來源、日期與數字是否一致、動線與預算是否合理、內容是否完整。"
            "若候選資料本身不符合偏好、資料不足或來源缺失，應把具體缺口告訴情報研究員並"
            "重新委派搜尋；若資料足夠但景點取捨、動線、節奏或預算編排不合理，應把具體問題"
            "告訴行程規劃師重新編排，不要浪費工具重新搜尋。"
            "同一個使用者問題最多進行兩輪修正。每次修正都必須引用上一輪的具體問題，"
            "不可無目的重做；若兩輪後仍受限於缺少關鍵條件或查不到可靠資料，直接對使用者"
            "說明限制與需要補充的條件。只有通過上述內部檢查後才可交付最終答案，且不得把"
            "委派、退件、審查、重試次數或『請重新產出』等內部過程寫進最終答案。"
        ),
        llm=llm,
        allow_delegation=True,  # manager 必須能委派
        max_iter=20,  # hierarchical 下 manager 的 max_iter 是整體吞吐瓶頸
        verbose=True,
    )

    return manager, [preference_analyst, travel_researcher, itinerary_planner]
