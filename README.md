# Lab5：Multi-Agent AI Agent — crewAI 框架

可連續對話的通用旅遊助理。一位 manager 依每輪問題與先前對話，動態協調偏好分析師、情報研究員與行程規劃師。

## 環境需求

- **Python 3.12**（crewai 需要 `>=3.10,<3.14`，不能用 Lab4 的 3.14）
- Node.js 20+（tavily 與 weather-mcp 走 `npx` 啟動；frankfurter 是遠端 server）

## 安裝

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 設定

1. 複製 `.env.example` 為 `.env`，填入 NVIDIA NIM 與 Tavily 的 API Key。
2. 把旅遊紀錄 `.txt` 放進 `knowledge/`（目錄下所有 `.txt` 都會被讀入）。

所有 Agent 共用 `CHAT_MODEL`，目前透過本機 CLI Proxy API 使用
`gpt-5.5`。聊天模型的端點與金鑰分別設定在 `CLI_PROXY_BASE_URL` 和
`CLI_PROXY_API_KEY`；RAG 的 embedding 仍透過 NVIDIA 使用專用的
`nv-embed-v1`。

## 執行

```powershell
python main.py
# 範例輸入：我下週二想去大阪三天兩夜
# 可以繼續輸入：幫我把行程調整得輕鬆一點
# Skill 示範：請把剛剛的行程匯出成 Markdown，檔名叫大阪三日遊
# 輸入 exit、quit、離開或結束可關閉
```

結果會直接顯示在終端機，並保留同一次執行期間的近期對話背景。

## 檔案結構

```
AIAgentCrewAI/
├── .env                 金鑰與模型名稱
├── requirements.txt
├── main.py              入口：啟動工具 → Conversational Flow 多輪對話 → 關閉工具
├── crew.py              定義通用 task 並組裝 hierarchical Crew
├── agents.py            一位 manager 與三位 worker、LLM、知識庫、embedder
├── tools.py             MCP server 設定 + 原始工具載入
├── skills/              itinerary-markdown-exporter Skill
├── exports/             Skill 產生的 Markdown 行程
└── knowledge/           旅遊紀錄 .txt
```

各檔案的參數直接寫在檔案裡，不使用 YAML 設定檔。

## 執行流程

```
main.py  啟動 MCP；由 Conversational Flow 保存對話
   │
   ▼
crew.py  每輪建立新 Crew（manager + 3 worker + 1 個通用 task），避免殘留執行狀態
   │
   ▼
旅遊規劃經理  分析問題，只委派必要的專員並整合結果
   │
   ├─ 需要個人偏好 ─→ 偏好分析師（讀 knowledge/）
   ├─ 需要即時情報 ─→ 情報研究員（用 MCP 工具）
   └─ 需要完整行程 ─→ 行程規劃師（整合前兩位專員的結果）
   │
   ▼
main.py  由 Flow 保存回答並等待下一輪；MCP 不會重新啟動
```

Python 迴圈只管理連續對話，不預先固定 Agent 工作流程。每一輪真正需要呼叫哪些專員，
仍由 hierarchical manager 現場判斷。這個版本會把最近兩輪對話交給 Crew，
避免 prompt 隨對話無限增長；關閉程式後不會跨程序保存 session。每位 Agent 都有
迭代上限，避免同一題反覆搜尋而拖慢回應；MCP 工具保留 CrewAI 原生平行執行。

## Agent Skills

課堂展示版只保留一個 `itinerary-markdown-exporter`。一般問答與行程規劃不會載入；
只有使用者明確要求把先前完成的行程儲存、匯出或整理成 Markdown 時，個人化行程
規劃師才會透過 CrewAI 的 `load_skill` 讀取完整 SOP。

| Agent | 可按需載入的 Skill |
|---|---|
| 旅遊規劃經理 | 無 |
| 旅遊偏好分析師 | 無 |
| 旅遊情報研究員 | 無 |
| 個人化行程規劃師 | `itinerary-markdown-exporter` |

Skill 規定觸發條件、檔名、Markdown 格式與不得重新搜尋等工作流程；實際寫檔由
`FileWriterTool` 執行，且只能寫入 `exports/`。旅遊紀錄仍是
`knowledge`，Tavily、天氣與匯率仍是即時資訊工具。終端機只在 Skill 真正載入時顯示：

```text
── 套用 Skill：itinerary-markdown-exporter
   執行 Agent：個人化行程規劃師
```

建議依序示範三輪：

1. `幫我安排下週二三天兩夜的大阪古蹟參訪行程`
2. `幫我把行程調整成以室內景點為主`
3. `請把剛剛的行程匯出成 Markdown，檔名叫大阪三日遊`

前兩輪不使用 Skill；第三輪才載入 Skill 並產生 `exports/大阪三日遊.md`。

## MCP servers

| server | 啟動方式 | 取用的工具 |
|---|---|---|
| tavily | `npx -y tavily-mcp@0.2.21`（stdio，需 `TAVILY_API_KEY`） | `tavily_search` |
| weather-mcp | `npx -y @dangahagan/weather-mcp@1.13.0`（stdio，免金鑰） | `get_forecast` |
| frankfurter | `https://mcp.frankfurter.dev/`（streamable-http，免金鑰） | `get_rates` |

兩個是本機 stdio 子程序、一個是遠端 HTTP endpoint，同一個 list 混著放。
載入時只從 server 選出本 Lab 需要的 3 個工具，不修改工具參數或回傳值。

## 直接使用 MCP 工具

`tools.py` 透過 CrewAI 的 `MCPServerAdapter` 載入工具，再把原始 MCP 工具直接交給 Agent：

```text
MCP Server → MCPServerAdapter → 原始 MCP Tools → CrewAI Agent
```

程式不再固定 Tavily 的搜尋深度與結果數、不再固定天氣格式，也不再轉換匯率參數；
Agent 依照 MCP 原始 schema 決定呼叫參數。`adapter.stop()` 仍負責關閉 stdio 子程序。

## 延伸練習

- 把 `Process.hierarchical` 改回 `Process.sequential`，比較兩者的差異與穩定度。
- 用 `from crewai.tools import tool` 自寫工具，取代其中一個 MCP server。
- 在通用 task 加上自適應 `guardrail`，做一層決定性的產出把關。
