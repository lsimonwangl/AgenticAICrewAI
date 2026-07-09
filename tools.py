"""tools.py — MCP 外部工具。

以 crewai-tools 的 MCPServerAdapter 分別啟動三個 MCP server（tavily / open-meteo /
frankfurter），每個 adapter 的 .tools 對應該 server 的工具清單，之後依 server 切給
對應 agent（角色化取用）。adapter 需在程式結束時 .stop()，由 main.py 的 finally 負責。

沿用 Lab 4 設定：tavily / open-meteo 走 npx stdio；frankfurter 走遠端 streamable-http。
MCPServerAdapter 同時接受 StdioServerParameters 與 {"url", "transport"} dict。
"""

import os
from typing import Any

from mcp import StdioServerParameters
from crewai_tools import MCPServerAdapter


def _tavily_params() -> StdioServerParameters:
    return StdioServerParameters(
        command="npx",
        args=["-y", "tavily-mcp@latest"],     # Lab 4：tavily 官方 stdio server
        env={**os.environ, "TAVILY_API_KEY": os.environ["TAVILY_API_KEY"]},
    )


def _open_meteo_params() -> StdioServerParameters:
    return StdioServerParameters(
        command="npx",
        args=["-y", "open-meteo-mcp-server"],  # Lab 4：免金鑰天氣 stdio server
        env={**os.environ},
    )


def _frankfurter_params() -> dict[str, Any]:
    # Lab 4：frankfurter 是遠端 HTTP MCP server，非 npx stdio。
    # crewai/mcpadapt 只認 "streamable-http"（不是 "http"），端點以 POST / 回 JSON-RPC。
    return {
        "url": "https://mcp.frankfurter.dev/",
        "transport": "streamable-http",
    }


def start_mcp_adapters():
    """啟動三個 MCP server。

    回傳 (adapters, bundles)：
      adapters — 供程式結束時逐一 .stop()
      bundles  — {"tavily": [...], "open_meteo": [...], "frankfurter": [...]}，
                 依 server 切分的工具清單，分別交給對應 agent
    """
    tavily = MCPServerAdapter(_tavily_params())
    open_meteo = MCPServerAdapter(_open_meteo_params())
    frankfurter = MCPServerAdapter(_frankfurter_params())

    bundles = {
        "tavily": tavily.tools,
        "open_meteo": open_meteo.tools,
        "frankfurter": frankfurter.tools,
    }
    return [tavily, open_meteo, frankfurter], bundles
