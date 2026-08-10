"""驗證 tools.py 的 _validate_kwargs 修補：型別轉換要保留，壞的選填欄位要剝掉。

跑法：venv\\Scripts\\python.exe test_tool_args.py
"""

from typing import Any, Literal

from crewai.tools import BaseTool
from crewai.tools.base_tool import BaseTool as BT
from pydantic import BaseModel, ConfigDict, Field

# 匯入 tools.py 會直接把修補掛到 BaseTool 上，所以下面測的就是正式程式碼
import tools  # noqa: F401

_original_validate_kwargs = tools._original_validate_kwargs
patched = BT._validate_kwargs


# ── 模擬 MCP 工具的 args_schema ──────────────────────

class WeatherArgs(BaseModel):
    """比照 weather-mcp get_forecast 的精簡形狀。"""

    city_name: str = Field(..., min_length=1)
    days: int | None = Field(default=None, ge=1, le=16)
    granularity: Literal["daily", "hourly"] | None = None
    detail: Literal["summary", "standard", "full"] | None = None
    units: Literal["metric", "imperial"] | None = None


class WeatherArgsForbidExtra(WeatherArgs):
    model_config = ConfigDict(extra="forbid")


class RatesArgs(BaseModel):
    """比照 frankfurter get_rates 的形狀。"""

    base: str | None = None
    date: str | None = None
    quotes: list[str] | None = None


class SearchArgs(BaseModel):
    """比照 tavily_search 的形狀。"""

    query: str = Field(...)
    country: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    max_results: int | None = None
    include_raw_content: bool | None = None


def make_tool(schema: type[BaseModel], tool_name: str) -> BaseTool:
    class _T(BaseTool):
        name: str = tool_name
        description: str = "test tool"
        args_schema: type[BaseModel] = schema

        def _run(self, **kwargs: Any) -> str:
            return "ok"

    return _T(name=tool_name, description="test tool", args_schema=schema)


# ── 測試 ────────────────────────────────────────────

RESULTS: list[tuple[str, bool, str]] = []


def check(label: str, tool: BaseTool, kwargs: dict, expect) -> None:
    """expect 為 dict 表示期望輸出；為 "raise" 表示期望拋錯。"""
    try:
        got = patched(tool, dict(kwargs))
        ok = expect != "raise" and got == expect
        detail = f"got={got}" if ok else f"got={got}  expected={expect}"
    except Exception as e:
        ok = expect == "raise"
        detail = f"raised: {str(e)[:70]}" if ok else f"UNEXPECTED raise: {str(e)[:70]}"
    RESULTS.append((label, ok, detail))


def baseline(label: str, tool: BaseTool, kwargs: dict) -> None:
    """記錄「原版 crewai 驗證」的行為，用來對照，不判定成敗。"""
    try:
        got = _original_validate_kwargs(tool, dict(kwargs))
        print(f"  [原版] {label:38} -> {got}")
    except Exception as e:
        print(f"  [原版] {label:38} -> raise: {str(e)[:60]}")


weather = make_tool(WeatherArgs, "get_forecast")
weather_strict = make_tool(WeatherArgsForbidExtra, "get_forecast_strict")
rates = make_tool(RatesArgs, "get_rates")
search = make_tool(SearchArgs, "tavily_search")

print("=" * 78)
print("對照組：原版 crewai _validate_kwargs 的行為")
print("=" * 78)
baseline("quotes 給真的 list", rates, {"base": "JPY", "quotes": ["TWD"]})
baseline("quotes 給 JSON 字串", rates, {"base": "JPY", "quotes": '["TWD"]'})
baseline("quotes 給裸字串", rates, {"base": "JPY", "quotes": "TWD"})
baseline("bool 給 'False' 字串", search, {"query": "x", "include_raw_content": "False"})
baseline("int 給 '10' 字串", search, {"query": "x", "max_results": "10"})
baseline("days=0 值域錯", weather, {"city_name": "Osaka, Japan", "days": 0})
baseline("多帶未知欄位 cold", weather, {"city_name": "Osaka, Japan", "cold": "false"})
baseline("未知欄位 + extra=forbid", weather_strict, {"city_name": "Osaka, Japan", "cold": "false"})
baseline("必填給空字串", weather, {"city_name": ""})

print()
print("=" * 78)
print("測試組：修補後的行為")
print("=" * 78)

# 1. 型別轉換必須保留（這是我上一版關掉驗證後弄壞的部分）
check("字串 'False' 轉成 bool", search,
      {"query": "x", "include_raw_content": "False"},
      {"query": "x", "include_raw_content": False})

check("字串 '10' 轉成 int", search,
      {"query": "x", "max_results": "10"},
      {"query": "x", "max_results": 10})

check("真的 list 原樣通過", rates,
      {"base": "JPY", "quotes": ["TWD"]},
      {"base": "JPY", "quotes": ["TWD"]})

# 2. 沒填的選填欄位不可以變成 None 送出去
check("未填的選填欄位被剝掉", rates,
      {"base": "JPY"},
      {"base": "JPY"})

# 3. 選填欄位值域錯 → 丟掉該欄位，其餘照送
check("days=0 被丟掉", weather,
      {"city_name": "Osaka, Japan", "days": 0},
      {"city_name": "Osaka, Japan"})

check("days=0 + detail 錯誤都丟掉", weather,
      {"city_name": "Osaka, Japan", "days": 0, "detail": "compact"},
      {"city_name": "Osaka, Japan"})

check("值域錯的丟掉、正確的留下", weather,
      {"city_name": "Osaka, Japan", "days": 0, "detail": "summary"},
      {"city_name": "Osaka, Japan", "detail": "summary"})

# 4. schema 沒有的欄位（模型憑空發明的 cold / temp / topic）
check("未知欄位（extra 預設）", weather,
      {"city_name": "Osaka, Japan", "cold": "false"},
      {"city_name": "Osaka, Japan"})

check("未知欄位（extra=forbid）", weather_strict,
      {"city_name": "Osaka, Japan", "cold": "false"},
      {"city_name": "Osaka, Japan"})

# 5. 必填欄位錯了必須照樣報錯，不可以被靜靜丟掉
check("必填欄位錯 -> 拋錯", weather,
      {"city_name": ""},
      "raise")

check("必填欄位缺 -> 拋錯", weather,
      {"days": 3},
      "raise")

# 6. JSON 字串形式的陣列（這輪 log 實際看到的 quotes='["JPY"]'）
check("quotes JSON 字串", rates,
      {"base": "JPY", "quotes": '["TWD"]'},
      {"base": "JPY", "quotes": ["TWD"]})

check("看起來像 JSON 但不是 -> 原樣", search,
      {"query": "[未收合的中括號"},
      {"query": "[未收合的中括號"})

check("普通字串不受影響", search,
      {"query": "大阪 民宿"},
      {"query": "大阪 民宿"})

# 7. LLM 會用字串 "null" 填滿選填欄位；呼叫 MCP 前必須整欄移除
check("字串 null 欄位被移除", search,
      {
          "query": "大阪景點",
          "country": "null",
          "start_date": "NULL",
          "end_date": " null ",
          "max_results": "3",
      },
      {"query": "大阪景點", "max_results": 3})

# 無條件移除發生在驗證前；必填 query 被移除後仍應由 pydantic 報錯
check("必填欄位為字串 null -> 拋錯", search,
      {"query": "null"},
      "raise")

print()
passed = sum(1 for _, ok, _ in RESULTS if ok)
for label, ok, detail in RESULTS:
    print(f"  {'PASS' if ok else 'FAIL'}  {label:34} {detail}")
print()
print(f"{passed}/{len(RESULTS)} 通過")
