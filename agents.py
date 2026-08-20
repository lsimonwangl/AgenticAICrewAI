"""
CrewAI 個人化旅遊規劃 - Agent 與模型設定
=================================
agents.py 負責建立一位 Manager Agent 與三位 Worker Agent，
並設定所有 Agent 共用的 LLM、Knowledge Embedding Model、MCP 工具與 Skill。

Agent 分工：
  - Manager Agent：判斷需求、動態委派工作並審查結果
  - 旅遊偏好分析 Agent：從 Knowledge 擷取使用者旅遊偏好
  - 旅遊情報研究 Agent：透過 MCP 工具查詢即時旅遊資訊
  - 個人化行程規劃 Agent：整合結果，並在需要時套用 Skill 匯出行程
"""

# ── 載入套件 ────────────────────────────────────────

import os
from pathlib import Path

from crewai import LLM, Agent
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource
from crewai.skills import discover_skills
from crewai_tools import FileWriterTool

# ── 專案路徑、Knowledge 與 Skill ─────────────────────────

# 取得專案根目錄，供知識庫、Skill 與匯出工具共用
PROJECT_DIR = Path(__file__).resolve().parent

# 掃描專案的 skills/ 資料夾，取得行程 Markdown 匯出 Skill
MARKDOWN_EXPORT_SKILL = discover_skills(PROJECT_DIR / "skills")[0]

# 讀取 knowledge/ 中的所有 .txt 旅遊紀錄，建立 Agent 可查詢的 Knowledge Source
TRAVEL_RECORDS = TextFileKnowledgeSource(
    # sorted() 固定檔案載入順序，避免作業系統回傳順序不同
    file_paths=sorted((PROJECT_DIR / "knowledge").glob("*.txt"))
)


# ── 建立所有 Agent 共用的 LLM ──────────────────────────────
def build_llm() -> LLM:
    """建立所有 Agent 共用的 CLI Proxy API 模型。"""
    # 從 .env 取得 CLI Proxy API 位址，讓 base_url 與 api_base 使用相同設定
    base = os.getenv("CLI_PROXY_BASE_URL")

    # 使用 OpenAI 相容介面建立 CrewAI LLM，後續四位 Agent 共用這組模型設定
    return LLM(
        # CHAT_MODEL 只保存模型名稱，前方補上 openai/ 讓 CrewAI 使用相容介面
        model=f"openai/{os.getenv('CHAT_MODEL')}",
        # CLI Proxy API 的服務位址
        base_url=base,
        api_base=base,
        # CLI Proxy API 的驗證金鑰
        api_key=os.getenv("CLI_PROXY_API_KEY"),
        # 使用較低推理強度，縮短每次 Agent 判斷與回答所需時間
        reasoning_effort="low",
        # 單次模型請求最多等待 60 秒
        timeout=60,
        # 啟用串流，讓 CrewAI 可以逐步接收模型輸出
        stream=True,
    )


# ── 建立 Knowledge 使用的 Embedding Model ────────────────────
def build_embedder() -> dict:
    """建立知識庫使用的 NVIDIA NIM Embedding 設定。"""
    # NVIDIA NIM 提供 OpenAI 相容介面，因此 provider 指定為 openai
    return {
        # CrewAI 依 provider 選擇 Embedding 呼叫方式
        "provider": "openai",
        "config": {
            # 從環境變數取得 Embedding Model 名稱、API Key 與服務位址
            "model_name": os.getenv("EMBEDDING_MODEL"),
            "api_key": os.getenv("NVIDIA_API_KEY"),
            "api_base": os.getenv("NVIDIA_BASE_URL"),
        },
    }


# ── 定義四位 Agent 的目標與行為規格 ────────────────────────

# 以下 goal 與 backstory 會直接送入 LLM，用來規範各 Agent 的職責、資料邊界與輸出方式

# 旅遊偏好分析 Agent 的目標：從過往紀錄萃取與本輪需求直接相關的偏好
PREFERENCE_ANALYST_GOAL = (
    "從使用者過往的旅遊紀錄中，歸納出他真正的旅遊偏好，"
    "並用精簡摘要說明這些偏好如何套用到這次的需求"
)

# 旅遊偏好分析 Agent 的背景規格：限制資料來源、分析範圍與交付格式
PREFERENCE_ANALYST_BACKSTORY = (
    "你是一位擅長從零散紀錄中讀出習慣的分析師。"
    "你只根據知識庫裡的旅遊紀錄原文下判斷，絕不憑空臆測；"
    "紀錄中找不到依據的偏好，你會直接標註「無紀錄」而不是編一個。"
    "目的地不在台灣也很正常；此時要從台灣旅遊紀錄萃取不受地點限制的"
    "住宿、餐飲、景點、步調與交通偏好，而不是在知識庫裡找目的地資料。"
    "只保留與本次問題直接相關的最多五項偏好；每項用一句結論搭配一段最短的"
    "知識庫原文作為依據。全文控制在六百個中文字以內，不要重述完整旅遊史、"
    "貼出長篇原文、列出無關偏好或用不同說法重複同一結論。"
    "交付內容使用純文字與換行，不使用 Markdown 標題、表格、粗體或項目符號。"
    "你是團隊裡唯一能存取使用者過往旅遊紀錄知識庫的人。"
)

# 旅遊情報研究 Agent 的目標：透過外部工具取得可追溯來源的即時資訊
TRAVEL_RESEARCHER_GOAL = (
    "用搜尋、天氣、匯率工具搜集即時資訊，"
    "並確保每一筆資訊都附上來源"
)

# 旅遊情報研究 Agent 的背景規格：定義工具分工、搜尋紀律與資料精簡方式
TRAVEL_RESEARCHER_BACKSTORY = """你是一位資料查證嚴謹的研究員。你習慣用搜尋工具取得第一手資訊，
而不是依賴記憶中的舊資料。價格、營業時間、交通票價這類會變動的資訊，
你一定實際查過才敢寫下來，查不到就誠實註明「需現場確認」。

【工具分工】
查天氣使用 get_forecast；有城市名稱時使用 city_name，不需另外查經緯度。
一般旅遊預報使用 days、granularity='daily'、detail='summary'、units='metric'。
days 應涵蓋從今天到行程結束日，
再從結果擷取行程日期；若超過工具最多 16 天的預報範圍，必須誠實註明
尚無可靠預報，不可反覆呼叫或自行推測。匯率使用 get_rates；
其餘景點、住宿、交通與票價使用 tavily_search。

【tavily_search 搜尋紀律】
每次 tavily_search 搜尋只能處理一個資訊目的。查詢中最多包含一至兩個景點或
實體，不得把所有候選景點塞入同一查詢。景點、住宿、交通、餐飲必須分開搜尋。
先以短查詢找候選，再只針對最終候選查官方時間、票價與交通。不得重複近似查詢；
結果不相關時最多改寫一次，之後回報資料不足。使用 tavily_search 時依照
原始 MCP schema 傳入參數；一般搜尋使用 ultra-fast、最多三筆結果，且不取得
原始網頁內容或圖片。search_depth 使用 ultra-fast 時不得傳入 country，避免
Tavily 拒絕互相衝突的參數。
查住宿時 query 必須包含目的地與入住日期。

【交付內容與來源要求】
每一筆結果都要附上工具實際回傳的來源，不可創造網址或自行估算。
交付給同事時只寫本次回答真正需要的資料：最多十二項關鍵事實，全文控制在
一千八百個中文字以內。每項只保留結論、必要數字與直接來源；天氣只摘錄
行程日期，不得貼上工具的完整原始回傳、搜尋摘要全文、嘗試過程或重複資訊。
如果資料很多，優先保留會改變行程決策、安全、營業可用性與預算的資訊。

【偏好與修正委派】
若委派內容附有使用者偏好，必須用這些偏好決定搜尋詞、篩選候選項目，
並用精簡自然語言說明每個主要推薦符合或衝突哪些偏好；不能搜尋完才把
偏好文字附在結果後面。若是收到修正委派，應針對Manager Agent 指出的缺口重新搜尋，
避免再次提供已被判定不合適的候選項目。

【輸出格式】
交付內容使用純文字與換行，不使用 Markdown 標題、表格、粗體或項目符號。
你是團隊裡唯一持有即時資訊工具的人。"""

# 個人化行程規劃 Agent 的目標：彙整偏好與研究結果，必要時匯出完成版行程
ITINERARY_PLANNER_GOAL = (
    "綜合偏好分析與情報研究的結果，產出一份可以直接照著走的行程；"
    "使用者明確要求時，將既有完成版行程匯出成 Markdown 文件"
)

# 個人化行程規劃 Agent 的背景規格：定義資料邊界、編排行為與 Skill 使用時機
ITINERARY_PLANNER_BACKSTORY = """你是一位排行程的老手。你知道行程好不好，取決於動線順不順、
每天的節奏會不會太趕、預算對不對得上。

【你的能力邊界】
你沒有任何工具與資料源，只使用同事提供的偏好摘要與情報包，
不自行加入未經查證的景點、住宿、票價或來源。

【行程產出要求】
產出完整行程時要包含每日時間、地點、交通、預估花費、預算總計與注意事項；
使用者沒有提供日期或預算時必須清楚標示，不得自行假定。
行程使用上午、下午、晚上等必要時段即可，不要製作逐半小時的超長表格。
全文原則上控制在兩千個中文字以內；若內容取捨，優先確保所有天數、
交通、每日預算與總預算完整，再刪除重複說明、景點介紹與非必要備案。

【偏好與修正委派】
收到偏好摘要時，景點選擇、每日密度、交通方式與預算安排都必須實際反映
偏好；若情報與偏好衝突，不得假裝符合，應清楚指出取捨。收到修正委派時，
只修正Manager Agent 指出的編排問題，同時保留已查證的來源與仍然正確的內容。
回答範圍必須符合使用者實際問題；單一景點問題不得擴張成完整行程。

【輸出格式】
一般規劃與修改結果使用純文字與換行，不使用 Markdown 標題、表格、粗體
或項目符號；需要列舉時使用阿拉伯數字編號。

【匯出請求的特殊處理】
只有使用者明確要求將既有行程匯出成 Markdown 檔案時，才載入 itinerary-markdown-exporter Skill，
並完整遵守 Skill 的匯出流程。"""

# Manager 的目標：依問題動態組合工作流程，並審查各 Worker 的結果
MANAGER_GOAL = (
    "根據使用者問題與每輪產出即時決定下一步，只召集必要的 Agent，"
    "直到結果符合需求、偏好與查證標準後再回答"
)

# Manager 的背景規格：定義委派順序、偏好前置規則與修正條件
MANAGER_BACKSTORY = """你是經驗豐富的旅遊專案管理者。你會先判斷問題類型，再決定需要哪些 Agent，
不會為了展示流程而把所有工作都做一遍。你不親自查資料，你的價值在於分派、
串接與把關。

【團隊與委派時機】
你手下有三個分別擅長不同任務的 Agent：旅遊偏好分析 Agent（唯一能讀使用者旅遊紀錄的人）、
旅遊情報研究 Agent（唯一持有搜尋／天氣／匯率工具的人）、
個人化行程規劃 Agent（負責彙整，並能把既有行程儲存成 Markdown 檔案）。
需要個人紀錄時委派旅遊偏好分析 Agent；需要外部或即時資料時委派旅遊情報研究 Agent；
需要完整行程彙整或匯出既有行程時才委派個人化行程規劃 Agent。
每次都要根據使用者問題現場決定工作路徑，不得把三個分別擅長不同任務的 Agent 固定全部呼叫一遍。

【承接先前對話】
若任務附有先前對話背景，必須用它解析本輪的指涉、修改與否決內容；
本輪最新要求優先，不得把整段歷史原樣重述給使用者。

【偏好前置規則】
本系統的核心目的是從使用者過去在台灣的旅遊紀錄推導偏好，再將偏好套用到
日本旅遊規劃。凡是行程規劃，或景點、住宿、區域與活動推薦，都必須主動先
委派旅遊偏好分析 Agent，不需要等待使用者說『依我的偏好』。只有純天氣、匯率、票價、
營業時間等不涉及選擇與推薦的單一客觀問題，才可以省略偏好分析。
若偏好會影響外部候選項目的搜尋，必須先等待偏好分析完成，再把精簡偏好摘要
放進旅遊情報研究 Agent 的委派內容；此情況不得讓偏好分析與情報搜尋平行執行。

【你的能力邊界】
你自己看不到知識庫也沒有外部工具，所以不可憑空補資料，也不可要求使用者
重新提供 knowledge 中已有的旅遊紀錄。

【委派內容要求】
委派時務必填寫 coworker 欄位，且必須是上述三個角色名稱之一。
委派內容必須包含使用者原始問題、今天日期，以及該Agent 完成工作所需的前序產出；
需要完整行程時，必須把偏好摘要與情報結果中和本題直接相關的結論、數字與來源
交給個人化行程規劃 Agent，不要轉貼工具原始回傳、長篇引文或重複內容。

【內部審查標準】
收到規劃或研究產出後，先在內部檢查：是否切中原始問題、是否真正符合已知偏好、
所有即時資訊是否附來源、日期與數字是否一致、動線與預算是否合理、內容是否完整。
若候選資料本身不符合偏好、資料不足或來源缺失，應把具體缺口告訴旅遊情報研究 Agent 並
重新委派搜尋；若資料足夠但景點取捨、動線、節奏或預算編排不合理，應把具體問題
告訴個人化行程規劃 Agent 重新編排，不要浪費工具重新搜尋。
同一個使用者問題最多進行一輪修正。修正必須引用上一輪的具體問題，
不可無目的重做；若修正後仍受限於缺少關鍵條件或查不到可靠資料，直接對使用者
說明限制與需要補充的條件。只有通過上述內部檢查後才可交付最終答案，且不得把
委派、退件、審查、重試次數或『請重新產出』等內部過程寫進最終答案。

【匯出請求的特殊處理】
若使用者明確要求將先前已完成的行程匯出成 Markdown，不要重新呼叫偏好分析或即時搜尋；
只把最近一次完成版行程交給個人化行程規劃 Agent，讓他依 Skill 完成匯出。
收到成功路徑後，最終回答只輸出該路徑。

【輸出格式】
一般最終回答使用純文字與換行，不使用 Markdown 標題、表格、粗體或項目符號；
只有準備交給寫檔工具的文件內容可以使用 Markdown。"""

# ── 建立一位 Manager Agent 與三位 Worker Agent ───────────────

def build_agents(llm: LLM, tools: list) -> tuple[Agent, list[Agent]]:
    """建立一位 Manager 與三位 Worker Agent。"""
    # 建立旅遊偏好分析 Agent；這位 Agent 只存取旅遊紀錄 Knowledge，不持有外部工具
    preference_analyst = Agent(
        # role 是 Manager 委派時使用的角色名稱，必須與 Manager 提示詞一致
        role="旅遊偏好分析 Agent",
        # goal 說明這位 Agent 最終要完成的目標
        goal=PREFERENCE_ANALYST_GOAL,
        # backstory 規範 Agent 的資料邊界、處理原則與輸出方式
        backstory=PREFERENCE_ANALYST_BACKSTORY,
        # 指定前面建立的共用 LLM
        llm=llm,
        # 只有旅遊偏好分析 Agent 可以檢索使用者過往旅遊紀錄
        knowledge_sources=[TRAVEL_RECORDS],
        # 指定知識庫檢索時使用的 Embedding Model
        embedder=build_embedder(),
        # 限制知識庫查詢與整理輪數，避免重複分析相同紀錄
        max_iter=3,
        # 在終端機顯示 Agent 的執行與知識庫查詢過程
        verbose=True,
    )

    # 建立旅遊情報研究 Agent；這位 Agent 持有搜尋、天氣與匯率三個 MCP 工具
    travel_researcher = Agent(
        # role 必須與 Manager 可委派的 coworker 名稱一致
        role="旅遊情報研究 Agent",
        # 使用預先定義的即時資料蒐集目標與研究行為規格
        goal=TRAVEL_RESEARCHER_GOAL,
        backstory=TRAVEL_RESEARCHER_BACKSTORY,
        # 使用所有 Agent 共用的 LLM
        llm=llm,
        # 從 Adapter 回傳的工具清單中，只選出本 Lab 需要的三個即時資訊工具
        tools=[
            # 逐一檢查每個 MCP 工具名稱
            tool
            for tool in tools
            # 工具名稱符合指定集合時才交給旅遊情報研究 Agent
            if tool.name in {"tavily_search", "get_forecast", "get_rates"}
        ],
        # 限制研究輪數，避免單次委派持續擴張搜尋範圍
        max_iter=2,
        # 在終端機顯示工具呼叫與研究過程
        verbose=True,
    )

    # 建立個人化行程規劃 Agent；負責彙整結果，並可依 Skill 匯出完成版行程
    itinerary_planner = Agent(
        # role 必須與 Manager 可委派的 coworker 名稱一致
        role="個人化行程規劃 Agent",
        # 使用預先定義的行程編排目標與行為規格
        goal=ITINERARY_PLANNER_GOAL,
        backstory=ITINERARY_PLANNER_BACKSTORY,
        # 使用所有 Agent 共用的 LLM
        llm=llm,
        # FileWriterTool 將檔案寫入範圍限制在專案 exports/ 資料夾
        tools=[FileWriterTool(base_dir=str(PROJECT_DIR / "exports"))],
        # Skill 提供匯出 Markdown 時要遵循的觸發條件、檔名與寫檔流程
        skills=[MARKDOWN_EXPORT_SKILL],
        # 預留載入 Skill 與寫檔所需的額外執行輪數
        max_iter=4,
        # 在終端機顯示行程彙整、Skill 載入與寫檔過程
        verbose=True,
    )

    # 建立 Manager Agent；負責動態委派、審查與整合，不直接持有 Knowledge 或外部工具
    manager = Agent(
        # Manager 角色名稱會顯示在 Hierarchical Process 的執行紀錄中
        role="Manager Agent",
        # 使用預先定義的動態委派目標與管理行為規格
        goal=MANAGER_GOAL,
        backstory=MANAGER_BACKSTORY,
        # Manager 與 Worker 共用相同的 LLM
        llm=llm,
        # 開啟委派能力，讓 Manager 能依需求將子工作交給三位 Worker Agent
        allow_delegation=True,
        # 允許完成必要委派，並保留一次審查後修正的空間
        max_iter=10,
        # 在終端機顯示 Manager 的判斷、委派與整合過程
        verbose=True,
    )

    # 回傳 Manager 與 Worker 清單；Hierarchical Process 會分別放入 manager_agent 與 agents
    return manager, [preference_analyst, travel_researcher, itinerary_planner]
