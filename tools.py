"""啟動旅遊規劃所需的 MCP Server 並載入原始工具。"""

# ── 載入套件 ────────────────────────────────────────

import os

from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters


# ── 啟動 MCP Server ──────────────────────────────────

def start_mcp_tools() -> tuple[MCPServerAdapter, list]:
    """啟動 MCP Server，回傳生命週期 Adapter 與三個原始工具。"""
    # MCPServerAdapter 統一管理兩個本機 Stdio Server 與一個遠端 HTTP Server
    adapter = MCPServerAdapter(
        [
            # Tavily MCP：搜尋景點、住宿與交通等即時旅遊資訊
            StdioServerParameters(
                # 固定 Tavily MCP 版本，避免套件更新後行為或 schema 改變
                command="npx",
                args=["-y", "tavily-mcp@0.2.21"],
                # 將 .env 中的 Tavily API Key 傳入 MCP 子程序
                env={"TAVILY_API_KEY": os.getenv("TAVILY_API_KEY", "")},
            ),
            # Weather MCP：查詢全球天氣預報
            StdioServerParameters(
                # 固定 Weather MCP 版本，避免工具清單與 schema 自動改變
                command="npx",
                args=["-y", "@dangahagan/weather-mcp@1.13.0"],
            ),
            # Frankfurter MCP：透過遠端 Server 查詢匯率
            {
                # 使用官方託管的 MCP Server，不需要啟動本機子程序
                "url": "https://mcp.frankfurter.dev/",
                "transport": "streamable-http",
            },
        ],
        # 只載入本 Lab 需要的 Tavily 搜尋工具
        "tavily_search",
        # 只載入 Weather MCP 的天氣預報工具
        "get_forecast",
        # 只載入 Frankfurter MCP 的匯率查詢工具
        "get_rates",
        # 第一次下載固定版本可能超過預設 30 秒，只延長 MCP 啟動等待時間
        connect_timeout=120,
    )
    # 將 Adapter 載入的工具轉為 list，供 CrewAI Agent 直接使用
    tools = list(adapter.tools)

    # 顯示實際載入的工具數量與名稱，方便確認 MCP 是否成功啟動
    print(f"已載入 {len(tools)} 個 MCP 工具：{[t.name for t in tools]}")

    # Adapter 交由 main.py 管理生命週期，工具清單則提供給 Crew 與 Agent 使用
    return adapter, tools
