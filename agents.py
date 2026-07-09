"""agents.py — 定義 manager 與五個角色專精的 worker。

- worker 各自持有自己的工具，allow_delegation=False（不互相委派）。
- manager 不持工具、allow_delegation=True，負責動態指派與審查退回。
- manager 以 manager_agent 傳入 Crew，不放進 agents 清單（見 crew.py）。
"""

from itertools import cycle

from crewai import Agent


def build_agents(llms, rag_tool, tavily_tools, meteo_tools, fx_tools):
    # 一把 key 一顆 LLM，round-robin 分給各 agent：不同 agent 用不同帳號，429 自然分散。
    # next(pick) 每呼叫一次取下一顆，繞回頭；6 個 agent、3 顆 LLM → 每顆由 2 個 agent 共用。
    pick = cycle(llms)
    preference_analyst = Agent(
        role="偏好分析師",
        goal="從使用者過往台灣旅遊紀錄，推斷其對住宿、景點、飲食、預算與旅遊步調的個人偏好",
        backstory=(
            "你擅長閱讀旅遊紀錄並歸納旅人的風格傾向。你只根據檢索到的紀錄原文推斷，"
            "絕不臆測或捏造未出現在紀錄中的偏好。"
        ),
        tools=[rag_tool],
        skills=["skills/preference"],  # 專屬 skill：RAG 檢索紀律
        llm=next(pick),
        allow_delegation=False,
        max_iter=3,   # 每個任務最多 3 圈思考迴圈，嚴控 LLM 呼叫次數
        verbose=True,
    )

    attraction_researcher = Agent(
        role="景點研究員",
        goal="搜尋日本大阪最新的景點、住宿與交通資訊，並驗證營業狀態與票價",
        backstory=(
            "你是熟悉日本在地旅遊的研究員，善用網路搜尋取得第一手最新資訊，"
            "並會查證景點是否仍營業、票價是否異動。"
        ),
        tools=tavily_tools,
        skills=["skills/research"],  # 專屬 skill：搜尋查證紀律
        llm=next(pick),
        allow_delegation=False,
        max_iter=6,   # 搜尋型 agent 需多次查詢，3 圈會在完成前被掐斷
        verbose=True,
    )

    weather_agent = Agent(
        role="天氣查詢員",
        goal="查詢行程期間大阪的天氣預報，供行程安排室內外備案",
        backstory="你負責提供準確的天氣資訊，讓行程能依降雨機率與氣溫調整室內外安排。",
        skills=["skills/weather"],  # 專屬 skill：weather_forecast 參數鐵則
        tools=meteo_tools,
        llm=next(pick),
        allow_delegation=False,
        max_iter=6,   # 工具參數試錯需要空間，3 圈會被 schema 錯誤吃光
        verbose=True,
    )

    fx_agent = Agent(
        role="匯率換算員",
        goal="換算日圓與新台幣匯率，供預算估算使用",
        backstory="你負責提供最新匯率，讓預算數字精準可信，換算過程清楚可查。",
        tools=fx_tools,
        skills=["skills/fx"],  # 專屬 skill：匯率精度紀律
        llm=next(pick),
        allow_delegation=False,
        max_iter=3,   # 每個任務最多 3 圈思考迴圈，嚴控 LLM 呼叫次數
        verbose=True,
    )

    itinerary_writer = Agent(
        role="行程統整師",
        goal="彙整偏好、景點、天氣與匯率資訊，產出完整的每日行程、住宿、交通與預算",
        backstory=(
            "你把各專才的產出組織成一份條理清楚、貼合使用者偏好的完整行程，"
            "確保各時段安排合理、預算前後一致。"
        ),
        tools=[],  # 無工具，只做彙整
        skills=["skills/itinerary"],  # 專屬 skill：統整一致性紀律
        llm=next(pick),
        allow_delegation=False,
        max_iter=3,   # 每個任務最多 3 圈思考迴圈，嚴控 LLM 呼叫次數
        verbose=True,
    )

    manager = Agent(
        role="行程規劃經理",
        goal=(
            "協調團隊完成一份貼合偏好且預算可行的大阪行程：依需求判斷該指派哪位專才、"
            "以什麼順序進行，收到行程草案後嚴格審查，不合格則退回重做直到通過"
        ),
        backstory=(
            "你是經驗豐富的旅遊專案經理。你不親自查資料，而是判斷該把工作交給哪位專才、"
            "以什麼順序協作；收到行程草案後，你會嚴格審查預算算術、費用一致性、偏好一致性"
            "與資訊正確性；只要發現預算不可行、費用矛盾或住宿偏離偏好，就退回要求修訂，"
            "直到合格才交付最終行程。"
        ),
        llm=next(pick),
        skills=["skills/manager"],  # 專屬 skill：委派與審查紀律
        allow_delegation=True,  # manager 必須能委派
        max_iter=12,  # 一圈只能做一個動作：委派 5 位專才 + 審查彙整的最低限度，再低會跳過專才
        verbose=True,
    )

    workers = [
        preference_analyst,
        attraction_researcher,
        weather_agent,
        fx_agent,
        itinerary_writer,
    ]
    return manager, workers
