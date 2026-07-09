---
name: openmeteo-params
description: weather_forecast 工具的參數呼叫鐵則，避免 schema 驗證錯誤與涵蓋不到目標日期
---

# weather_forecast 參數鐵則

- 用不到的選填參數一律省略，絕不傳 null 或 0（不要帶 models、current、current_weather、past_days）。
- forecast_days 必須大到涵蓋行程目標日期：目標日距今天數 +1，最大 16。
- 查未來行程一律用 weather_forecast 預報，禁止改查歷史年份的平均資料替代。
- 必填參數只有 latitude 與 longitude；座標先用 geocoding 取得。
