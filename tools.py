"""MCP 工具載入。

tools.py 負責啟動 MCP server、過濾出真正用得到的工具，
並修補 crewAI 的參數驗證讓 MCP 工具能正常運作。

三個 server 的連線方式不同：
    - tavily（網路搜尋）：npx stdio，需 API Key
    - weather-mcp（天氣預報）：npx stdio，免金鑰
    - frankfurter（匯率）：遠端 streamable-http，不需要 npx

MCPServerAdapter 同時吃 StdioServerParameters 與 {"url", "transport"} dict，
所以兩種連線方式可以放在同一個 list 裡。

此模組提供 start_mcp_tools() 供 main.py 呼叫。
"""

import json
import os

from crewai.tools.base_tool import BaseTool
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters
from pydantic import ValidationError

# ── 修補 crewAI 的工具參數驗證 ──────────────────────
# crewai 用工具的 args_schema 對參數做 pydantic 驗證（base_tool.py:279）。
# 這層的型別轉換是必要的：LLM 會把參數寫成字串（'10'、'False'），靠它轉回來。
# 但它有三個問題會讓 MCP 工具必定失敗，源頭都是「LLM 不會省略選填欄位」：
#
#   1. 陣列被序列化成字串：模型寫 '["TWD"]' 而不是 ["TWD"]，pydantic 不會轉。
#   2. 選填欄位被填佔位值：days=0（schema 要求 >= 1）、start_hour='None'。
#   3. 驗證通過時用 model_dump() 且沒有 exclude_none，模型沒填的欄位會變成
#      None 送給 MCP server，而 server 端的 zod 多半不收 null。
#
# 對策：先把 JSON 字串還原並移除值完全等於字串 'null' 的欄位，驗證失敗時
# 丟掉「出錯且非必填」與「schema 沒有」的欄位再驗一次，最後剝掉 None。
# 必填欄位若被移除，仍會在 pydantic 驗證時報錯，不會被靜靜吞掉。
# 行為由 test_tool_args.py 的 16 個 case 涵蓋。

_original_validate_kwargs = BaseTool._validate_kwargs


def _parse_json_strings(kwargs: dict) -> dict:
    """還原 JSON 字串，並移除 LLM 產生的字串 null。"""
    out = {}
    for key, value in kwargs.items():
        if isinstance(value, str):
            text = value.strip()
            if text.lower() == "null":
                continue
            if text[:1] in ("[", "{"):
                try:
                    value = json.loads(text)
                except ValueError:
                    pass
        out[key] = value
    return out


def _validate_kwargs_dropping_bad_optionals(self, kwargs: dict) -> dict:
    kwargs = _parse_json_strings(kwargs)
    try:
        validated = _original_validate_kwargs(self, kwargs)
    except Exception as first_error:
        cause = first_error.__cause__
        if not isinstance(cause, ValidationError) or self.args_schema is None:
            raise
        fields = self.args_schema.model_fields
        drop = set()
        for err in cause.errors():
            if not err["loc"]:
                continue
            name = str(err["loc"][0])
            if name not in fields or not fields[name].is_required():
                drop.add(name)
        if not drop:
            raise
        validated = _original_validate_kwargs(
            self, {k: v for k, v in kwargs.items() if k not in drop}
        )
    return {k: v for k, v in validated.items() if v is not None}


BaseTool._validate_kwargs = _validate_kwargs_dropping_bad_optionals


# ── MCP server 設定 ─────────────────────────────────

# Tavily MCP 的工具 schema 目前沒有暴露 include_answer，因此先用 ultra-fast
# 讓每個來源只回傳一段摘要，並在 server 端強制限制來源數與多餘欄位。
# DEFAULT_PARAMETERS 會覆寫 Agent 傳入的同名參數，避免模型擅自改回 advanced。
TAVILY_DEFAULT_PARAMETERS = (
    '{"search_depth":"ultra-fast",'
    '"max_results":3,'
    '"include_raw_content":false,'
    '"include_images":false,'
    '"include_image_descriptions":false,'
    '"include_favicon":false}'
)


def build_server_params() -> list:
    """回傳三個 MCP server 的連線設定。

    env 要帶 **os.environ，否則 npx 找不到 PATH。
    """
    return [
        # tavily：網路搜尋，查景點／住宿／交通即時資訊
        StdioServerParameters(
            command="npx",
            args=["-y", "tavily-mcp@latest"],
            env={
                **os.environ,
                "TAVILY_API_KEY": os.getenv("TAVILY_API_KEY", ""),
                "DEFAULT_PARAMETERS": TAVILY_DEFAULT_PARAMETERS,
            },
        ),
        # weather-mcp：免費全球天氣預報。只註冊 forecast，避免其餘工具占用 context；
        # 預設使用公制，實際呼叫再要求 daily + summary，防止大型 hourly 輸出。
        StdioServerParameters(
            command="npx",
            args=["-y", "@dangahagan/weather-mcp@latest"],
            env={
                **os.environ,
                "ENABLED_TOOLS": "forecast",
                "WEATHER_UNITS": "metric",
            },
        ),
        # frankfurter：免費匯率換算，官方託管的遠端 server
        {
            "url": "https://mcp.frankfurter.dev/",
            "transport": "streamable-http",
        },
    ]


# 每個 server 都只暴露必要工具，避免把大量 schema 全部送進 LLM。
KEEP = (
    "tavily_search",     # 景點／住宿／交通／票價
    "get_forecast",      # 依城市查每日精簡天氣，不需另外 geocoding
    "get_rates",         # 日圓對新台幣匯率
)


def start_mcp_tools() -> tuple[MCPServerAdapter, list]:
    """啟動 MCP server，回傳 (adapter, 過濾後的工具清單)。

    adapter 要交給呼叫端在結束時 .stop()，否則 npx 子程序不會被關掉。
    """
    adapter = MCPServerAdapter(build_server_params(), *KEEP)
    tools = list(adapter.tools)
    print(f"已載入 {len(tools)} 個 MCP 工具：{[t.name for t in tools]}")
    return adapter, tools
