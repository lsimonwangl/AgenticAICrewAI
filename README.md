# Lab5：Multi-Agent AI Agent — crewAI 框架

可連續對話的通用旅遊助理。一位 manager 依每輪問題與先前對話，動態協調偏好分析師、情報研究員與行程規劃師。

## 環境需求

- **Python 3.12**（crewai 需要 `>=3.10,<3.14`，不能用 Lab4 的 3.14）
- Node.js 18+（tavily 與 weather-mcp 走 `npx` 啟動；frankfurter 是遠端 server）

## 安裝

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 設定

1. 複製 `.env.example` 為 `.env`，填入 NVIDIA NIM 與 Tavily 的 API Key。
2. 把旅遊紀錄 `.txt` 放進 `knowledge/`（目錄下所有 `.txt` 都會被讀入）。

## 執行

```powershell
python main.py
# 範例輸入：我下週二想去大阪三天兩夜
# 可以繼續輸入：第二天改輕鬆一點
# 輸入 exit、quit、離開或結束可關閉
```

結果會直接顯示在終端機，並保留同一次執行期間的近期對話背景。**必須在專案目錄下執行**，`knowledge/` 是相對路徑。

## 檔案結構

```
AIAgentCrewAI/
├── .env                 金鑰與模型名稱
├── requirements.txt
├── main.py              入口：啟動工具 → while 多輪輸入與歷史 → 關閉工具
├── crew.py              定義通用 task 並組裝 hierarchical Crew
├── agents.py            一位 manager 與三位 worker、LLM、知識庫、embedder
├── tools.py             MCP server 設定 + crewAI 參數驗證的修補
├── test_tool_args.py    tools.py 那段修補的行為測試（16 個 case）
└── knowledge/           旅遊紀錄 .txt
```

各檔案的參數直接寫在檔案裡，不使用 YAML 設定檔。

## 執行流程

```
main.py  啟動 MCP；用 while + list 保存對話
   │
   ▼
crew.py  每輪建立新的 Crew（manager + 3 worker + 1 個通用 task）
   │
   ▼
旅遊規劃經理  分析問題，只委派必要的專員並整合結果
   │
   ├─ 需要個人偏好 ─→ 偏好分析師（讀 knowledge/）
   ├─ 需要即時情報 ─→ 情報研究員（用 MCP 工具）
   └─ 需要完整行程 ─→ 行程規劃師（整合前兩位專員的結果）
   │
   ▼
main.py  把回答加入 history 並等待下一輪；MCP 不會重新啟動
```

Python 迴圈只管理連續對話，不預先固定 Agent 工作流程。每一輪真正需要呼叫哪些專員，
仍由 hierarchical manager 現場判斷。這個基礎版本會把本次執行的完整歷史
交給 Crew；對話很長時 token 會隨之增加，且關閉程式後不會跨程序保存 session。

## MCP servers

| server | 啟動方式 | 取用的工具 |
|---|---|---|
| tavily | `npx -y tavily-mcp@latest`（stdio，需 `TAVILY_API_KEY`） | `tavily_search` |
| weather-mcp | `npx -y @dangahagan/weather-mcp@latest`（stdio，免金鑰） | `get_forecast` |
| frankfurter | `https://mcp.frankfurter.dev/`（streamable-http，免金鑰） | `get_rates` |

兩個是本機 stdio 子程序、一個是遠端 HTTP endpoint，同一個 list 混著放。
每個 server 都只取用必要工具，共 3 個。weather-mcp 另外用 `ENABLED_TOOLS=forecast`
在 server 端只註冊預報工具，並以 `WEATHER_UNITS=metric` 固定公制。Agent 查一般天氣時
使用每日、summary 輸出，避免原本 `weather_forecast` 的巨大 schema 與 15 分鐘時間序列。

## 已知問題與修補

`tools.py` 頂端有一段對 crewAI `BaseTool._validate_kwargs` 的修補。原因是
LLM 不會「省略」選填欄位，只會每格填佔位值，而 crewAI 的驗證層又會把沒填的
欄位補成 `None` 送給 MCP server：

| 症狀 | 處理 |
|---|---|
| 陣列被寫成字串 `'["TWD"]'` | 驗證前先 `json.loads` 還原 |
| 選填欄位填佔位值 `days=0` | 驗證失敗時丟掉該欄位再驗一次 |
| 模型發明 schema 沒有的欄位 | 一併丟掉 |
| 沒填的欄位被補成 `None` 送出 | 輸出前剝掉 |

必填欄位錯了照樣拋錯。行為由 `test_tool_args.py` 的 14 個 case 涵蓋。

## 延伸練習

- 把 `Process.hierarchical` 改回 `Process.sequential`，比較兩者的差異與穩定度。
- 用 `from crewai.tools import tool` 自寫工具，取代其中一個 MCP server。
- 在通用 task 加上自適應 `guardrail`，做一層決定性的產出把關。
