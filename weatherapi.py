"""
Weather Pulse REST API (Single-File Edition)
Author: Pranshu Panwar | 1st Year BCA (AI & Data Science), K.R. Mangalam University

Features:
- Real-time current weather, multi-day forecast, and coordinate lookup endpoints
- SQLite response caching with TTL expiration (X-Cache: HIT / MISS)
- API Key authentication (header or query parameter) & sliding-window rate limiting
- Built-in interactive Web Dashboard (/) and automatic Swagger UI (/docs)
"""

from collections import defaultdict
from datetime import datetime, timezone
import json
import os
import sqlite3
import time
from typing import Optional
import urllib.error
import urllib.parse
import urllib.request

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Security
from fastapi.responses import HTMLResponse
from fastapi.security import APIKeyHeader, APIKeyQuery
import uvicorn

# ==============================================================================
# 1. CONFIGURATION & CONSTANTS
# ==============================================================================
APP_TITLE = "Weather Pulse REST API"
APP_VERSION = "1.0.0"
CACHE_DB_PATH = "weather_cache.db"
CACHE_TTL_SECONDS = 600  # Cache responses for 10 minutes
RATE_LIMIT_PER_MINUTE = 60
VALID_API_KEYS = {"krmu-dev-key-2026", "demo-key"}
START_TIME = time.time()

# WMO Meteorological Weather Interpretation Codes
WMO_CODES = {
    0: ("Clear sky", "Clear"),
    1: ("Mainly clear", "Clear"),
    2: ("Partly cloudy", "Cloudy"),
    3: ("Overcast", "Cloudy"),
    45: ("Fog", "Fog"),
    48: ("Depositing rime fog", "Fog"),
    51: ("Light drizzle", "Drizzle"),
    53: ("Moderate drizzle", "Drizzle"),
    55: ("Dense drizzle", "Drizzle"),
    61: ("Slight rain", "Rain"),
    63: ("Moderate rain", "Rain"),
    65: ("Heavy rain", "Rain"),
    71: ("Slight snowfall", "Snow"),
    73: ("Moderate snowfall", "Snow"),
    75: ("Heavy snowfall", "Snow"),
    80: ("Slight rain showers", "Rain"),
    81: ("Moderate rain showers", "Rain"),
    82: ("Violent rain showers", "Rain"),
    95: ("Thunderstorm", "Thunderstorm"),
    96: ("Thunderstorm with slight hail", "Thunderstorm"),
    99: ("Thunderstorm with heavy hail", "Thunderstorm"),
}

# ==============================================================================
# 2. SQLITE CACHING LAYER
# ==============================================================================
def init_cache_db() -> None:
    """Create the SQLite cache table if it does not already exist."""
    with sqlite3.connect(CACHE_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS weather_cache (
                cache_key TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                expires_at REAL NOT NULL
            )
            """
        )
        conn.commit()


def get_cached_response(cache_key: str) -> Optional[dict]:
    """Retrieve a non-expired JSON payload from SQLite cache."""
    now = time.time()
    with sqlite3.connect(CACHE_DB_PATH) as conn:
        row = conn.execute(
            "SELECT payload, expires_at FROM weather_cache WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if not row:
            return None
        payload_str, expires_at = row
        if now < expires_at:
            return json.loads(payload_str)
        conn.execute("DELETE FROM weather_cache WHERE cache_key = ?", (cache_key,))
        conn.commit()
    return None


def set_cached_response(cache_key: str, payload: dict, ttl: int = CACHE_TTL_SECONDS) -> None:
    """Store a JSON-serializable dictionary in SQLite cache with a TTL."""
    expires_at = time.time() + ttl
    with sqlite3.connect(CACHE_DB_PATH) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO weather_cache (cache_key, payload, expires_at) VALUES (?, ?, ?)",
            (cache_key, json.dumps(payload), expires_at),
        )
        conn.commit()


def count_active_cache_entries() -> int:
    """Return the count of currently valid cached records."""
    now = time.time()
    with sqlite3.connect(CACHE_DB_PATH) as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM weather_cache WHERE expires_at > ?", (now,)
        ).fetchone()
    return row[0] if row else 0


# ==============================================================================
# 3. SECURITY: API KEY AUTH & SLIDING-WINDOW RATE LIMITER
# ==============================================================================
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
api_key_query = APIKeyQuery(name="api_key", auto_error=False)
request_timestamps: dict[str, list[float]] = defaultdict(list)


def verify_api_key(
    header_key: Optional[str] = Security(api_key_header),
    query_key: Optional[str] = Security(api_key_query),
) -> str:
    """Validate API key from either X-API-Key header or ?api_key= query parameter."""
    provided_key = header_key or query_key
    if provided_key in VALID_API_KEYS:
        return provided_key
    raise HTTPException(
        status_code=401,
        detail={
            "error": "Unauthorized: Invalid or missing API key.",
            "hint": "Provide header 'X-API-Key: krmu-dev-key-2026' or query '?api_key=krmu-dev-key-2026'.",
        },
    )


def enforce_rate_limit(request: Request) -> int:
    """Allow up to RATE_LIMIT_PER_MINUTE requests per client IP every 60 seconds."""
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    window_start = now - 60.0

    recent = [ts for ts in request_timestamps[client_ip] if ts > window_start]
    if len(recent) >= RATE_LIMIT_PER_MINUTE:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Too Many Requests",
                "limit": f"{RATE_LIMIT_PER_MINUTE} requests per minute",
            },
        )
    recent.append(now)
    request_timestamps[client_ip] = recent
    return RATE_LIMIT_PER_MINUTE - len(recent)


# ==============================================================================
# 4. UPSTREAM WEATHER DATA CLIENT & UNIT HELPERS
# ==============================================================================
def http_get_json(url: str) -> dict:
    """Perform an upstream HTTP GET request using Python's standard urllib."""
    req = urllib.request.Request(
        url, headers={"User-Agent": "WeatherPulseAPI/1.0 (KRMU Student Project)"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Upstream weather provider HTTP error: {exc.code}",
        ) from exc
    except urllib.error.URLError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Unable to reach upstream weather provider: {exc.reason}",
        ) from exc


def geocode_city(city: str) -> dict:
    """Resolve a city name into latitude, longitude, and administrative metadata."""
    encoded_city = urllib.parse.quote(city.strip())
    geo_url = (
        f"https://geocoding-api.open-meteo.com/v1/search"
        f"?name={encoded_city}&count=1&language=en&format=json"
    )
    geo_data = http_get_json(geo_url)
    results = geo_data.get("results")
    if not results:
        raise HTTPException(
            status_code=404,
            detail=f"Location '{city}' could not be found. Check spelling or try coordinates.",
        )
    top = results[0]
    return {
        "name": top.get("name"),
        "region": top.get("admin1", ""),
        "country": top.get("country", ""),
        "latitude": top.get("latitude"),
        "longitude": top.get("longitude"),
        "timezone": top.get("timezone", "UTC"),
    }


def format_temp(celsius: float, units: str) -> float:
    return round((celsius * 9 / 5) + 32, 1) if units == "imperial" else round(celsius, 1)


def format_speed(kmh: float, units: str) -> float:
    return round(kmh * 0.621371, 1) if units == "imperial" else round(kmh, 1)


def decode_weather_code(code: int) -> dict:
    description, category = WMO_CODES.get(code, ("Unknown condition", "Unknown"))
    return {"code": code, "category": category, "description": description}


# ==============================================================================
# 5. FASTAPI APPLICATION & REST ENDPOINTS
# ==============================================================================
app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=(
        "A single-file RESTful Weather API built by **Pranshu Panwar** "
        "(1st Year BCA AI & Data Science, K.R. Mangalam University). "
        "Supports live geocoding, current conditions, 7-day forecasts, SQLite caching, "
        "rate limiting, and API key authentication."
    ),
)


@app.on_event("startup")
def on_startup() -> None:
    init_cache_db()


@app.get("/api/v1/health", tags=["System"])
def health_check():
    """Public health-check endpoint reporting API status, uptime, and cache metrics."""
    return {
        "status": "healthy",
        "service": APP_TITLE,
        "version": APP_VERSION,
        "author": "Pranshu Panwar",
        "uptime_seconds": round(time.time() - START_TIME, 2),
        "cache": {
            "backend": "SQLite",
            "ttl_seconds": CACHE_TTL_SECONDS,
            "active_entries": count_active_cache_entries(),
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/v1/weather/current", tags=["Weather"])
def get_current_weather(
    city: str = Query(..., min_length=2, description="City name (e.g., New Delhi, London, Tokyo)"),
    units: str = Query("metric", pattern="^(metric|imperial)$", description="metric (°C, km/h) or imperial (°F, mph)"),
    api_key: str = Depends(verify_api_key),
    remaining_quota: int = Depends(enforce_rate_limit),
):
    """Fetch real-time current weather conditions for a given city."""
    start = time.perf_counter()
    cache_key = f"current:{city.strip().lower()}:{units}"
    cached = get_cached_response(cache_key)

    if cached:
        cached["meta"]["cache"] = "HIT"
        cached["meta"]["response_time_ms"] = round((time.perf_counter() - start) * 1000, 2)
        cached["meta"]["rate_limit_remaining"] = remaining_quota
        return cached

    location = geocode_city(city)
    weather_url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={location['latitude']}&longitude={location['longitude']}"
        f"&current=temperature_2m,relative_humidity_2m,apparent_temperature,"
        f"is_day,precipitation,weather_code,wind_speed_10m,surface_pressure"
        f"&timezone=auto"
    )
    raw = http_get_json(weather_url)
    current = raw.get("current", {})

    temp_unit = "°F" if units == "imperial" else "°C"
    speed_unit = "mph" if units == "imperial" else "km/h"

    response_payload = {
        "status": "success",
        "location": location,
        "current": {
            "observation_time": current.get("time"),
            "is_day": bool(current.get("is_day", 1)),
            "condition": decode_weather_code(current.get("weather_code", -1)),
            "temperature": format_temp(current.get("temperature_2m", 0.0), units),
            "feels_like": format_temp(current.get("apparent_temperature", 0.0), units),
            "humidity_percent": current.get("relative_humidity_2m"),
            "wind_speed": format_speed(current.get("wind_speed_10m", 0.0), units),
            "precipitation_mm": current.get("precipitation", 0.0),
            "pressure_hpa": current.get("surface_pressure"),
            "units": {
                "temperature": temp_unit,
                "wind_speed": speed_unit,
                "pressure": "hPa",
            },
        },
        "meta": {
            "cache": "MISS",
            "source": "Open-Meteo Meteorological Engine",
            "response_time_ms": round((time.perf_counter() - start) * 1000, 2),
            "rate_limit_remaining": remaining_quota,
        },
    }

    set_cached_response(cache_key, response_payload)
    return response_payload


@app.get("/api/v1/weather/forecast", tags=["Weather"])
def get_weather_forecast(
    city: str = Query(..., min_length=2, description="City name (e.g., New Delhi, Mumbai)"),
    days: int = Query(5, ge=1, le=7, description="Number of forecast days (1 to 7)"),
    units: str = Query("metric", pattern="^(metric|imperial)$"),
    api_key: str = Depends(verify_api_key),
    remaining_quota: int = Depends(enforce_rate_limit),
):
    """Fetch a multi-day daily weather forecast for a given city."""
    start = time.perf_counter()
    cache_key = f"forecast:{city.strip().lower()}:{days}:{units}"
    cached = get_cached_response(cache_key)

    if cached:
        cached["meta"]["cache"] = "HIT"
        cached["meta"]["response_time_ms"] = round((time.perf_counter() - start) * 1000, 2)
        cached["meta"]["rate_limit_remaining"] = remaining_quota
        return cached

    location = geocode_city(city)
    forecast_url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={location['latitude']}&longitude={location['longitude']}"
        f"&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max"
        f"&timezone=auto&forecast_days={days}"
    )
    raw = http_get_json(forecast_url)
    daily = raw.get("daily", {})

    dates = daily.get("time", [])
    codes = daily.get("weather_code", [])
    max_temps = daily.get("temperature_2m_max", [])
    min_temps = daily.get("temperature_2m_min", [])
    precip_probs = daily.get("precipitation_probability_max", [])

    forecast_days = []
    for i in range(len(dates)):
        forecast_days.append(
            {
                "date": dates[i],
                "condition": decode_weather_code(codes[i]),
                "temp_max": format_temp(max_temps[i], units),
                "temp_min": format_temp(min_temps[i], units),
                "precipitation_probability_percent": precip_probs[i],
            }
        )

    response_payload = {
        "status": "success",
        "location": location,
        "units": "°F" if units == "imperial" else "°C",
        "forecast_days": forecast_days,
        "meta": {
            "cache": "MISS",
            "source": "Open-Meteo Meteorological Engine",
            "response_time_ms": round((time.perf_counter() - start) * 1000, 2),
            "rate_limit_remaining": remaining_quota,
        },
    }

    set_cached_response(cache_key, response_payload)
    return response_payload


@app.get("/api/v1/weather/coordinates", tags=["Weather"])
def get_weather_by_coordinates(
    lat: float = Query(..., ge=-90.0, le=90.0, description="Latitude (-90 to 90)"),
    lon: float = Query(..., ge=-180.0, le=180.0, description="Longitude (-180 to 180)"),
    units: str = Query("metric", pattern="^(metric|imperial)$"),
    api_key: str = Depends(verify_api_key),
    remaining_quota: int = Depends(enforce_rate_limit),
):
    """Fetch real-time current weather directly via GPS latitude and longitude."""
    start = time.perf_counter()
    cache_key = f"coords:{round(lat, 3)}:{round(lon, 3)}:{units}"
    cached = get_cached_response(cache_key)

    if cached:
        cached["meta"]["cache"] = "HIT"
        cached["meta"]["response_time_ms"] = round((time.perf_counter() - start) * 1000, 2)
        cached["meta"]["rate_limit_remaining"] = remaining_quota
        return cached

    weather_url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,relative_humidity_2m,apparent_temperature,"
        f"weather_code,wind_speed_10m"
        f"&timezone=auto"
    )
    raw = http_get_json(weather_url)
    current = raw.get("current", {})

    response_payload = {
        "status": "success",
        "coordinates": {"latitude": lat, "longitude": lon, "timezone": raw.get("timezone")},
        "current": {
            "observation_time": current.get("time"),
            "condition": decode_weather_code(current.get("weather_code", -1)),
            "temperature": format_temp(current.get("temperature_2m", 0.0), units),
            "feels_like": format_temp(current.get("apparent_temperature", 0.0), units),
            "humidity_percent": current.get("relative_humidity_2m"),
            "wind_speed": format_speed(current.get("wind_speed_10m", 0.0), units),
        },
        "meta": {
            "cache": "MISS",
            "response_time_ms": round((time.perf_counter() - start) * 1000, 2),
            "rate_limit_remaining": remaining_quota,
        },
    }

    set_cached_response(cache_key, response_payload)
    return response_payload


# ==============================================================================
# 6. EMBEDDED INTERACTIVE DEVELOPER SANDBOX UI (GET /)
# ==============================================================================
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Weather Pulse API — Interactive Sandbox</title>
  <style>
    :root { --bg: #0f172a; --card: #1e293b; --accent: #38bdf8; --text: #f8fafc; --muted: #94a3b8; }
    * { box-sizing: border-box; font-family: system-ui, -apple-system, sans-serif; }
    body { margin: 0; background: var(--bg); color: var(--text); padding: 2rem; }
    .container { max-width: 900px; margin: 0 auto; }
    header { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem; margin-bottom: 1.5rem; }
    h1 { margin: 0; font-size: 1.6rem; color: var(--accent); }
    .subtitle { color: var(--muted); font-size: 0.9rem; margin-top: 0.25rem; }
    .docs-btn { background: #334155; color: var(--text); text-decoration: none; padding: 0.6rem 1rem; border-radius: 8px; font-weight: 600; font-size: 0.9rem; }
    .docs-btn:hover { background: #475569; }
    .card { background: var(--card); padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem; border: 1px solid #334155; }
    .controls { display: grid; grid-template-columns: 2fr 1fr 1fr auto; gap: 0.75rem; }
    @media (max-width: 680px) { .controls { grid-template-columns: 1fr; } }
    input, select, button { padding: 0.7rem 0.9rem; border-radius: 8px; border: 1px solid #475569; background: #0f172a; color: var(--text); font-size: 0.95rem; }
    button { background: var(--accent); color: #0f172a; font-weight: 700; border: none; cursor: pointer; }
    button:hover { opacity: 0.9; }
    .badges { display: flex; gap: 0.75rem; flex-wrap: wrap; margin-bottom: 1rem; font-size: 0.85rem; }
    .badge { background: #0f172a; padding: 0.35rem 0.75rem; border-radius: 999px; border: 1px solid #334155; color: var(--accent); }
    pre { background: #020617; padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem; line-height: 1.45; color: #e2e8f0; border: 1px solid #1e293b; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>Weather Pulse REST API</h1>
        <div class="subtitle">Built by Pranshu Panwar • BCA (AI &amp; Data Science), K.R. Mangalam University</div>
      </div>
      <a class="docs-btn" href="/docs" target="_blank">Open Swagger UI (/docs) ↗</a>
    </header>

    <div class="card">
      <div class="controls">
        <input id="city" type="text" value="New Delhi" placeholder="Enter city name..." />
        <select id="endpoint">
          <option value="current">Current Weather</option>
          <option value="forecast">5-Day Forecast</option>
        </select>
        <select id="units">
          <option value="metric">Metric (°C, km/h)</option>
          <option value="imperial">Imperial (°F, mph)</option>
        </select>
        <button onclick="runRequest()">Send Request</button>
      </div>
    </div>

    <div class="card">
      <div class="badges">
        <span class="badge" id="statusBadge">HTTP Status: —</span>
        <span class="badge" id="cacheBadge">Cache: —</span>
        <span class="badge" id="latencyBadge">Latency: —</span>
      </div>
      <div style="font-size: 0.85rem; color: var(--muted); margin-bottom: 0.5rem;" id="requestUrl">GET /api/v1/weather/current?city=New%20Delhi&amp;units=metric</div>
      <pre id="output">Click "Send Request" to test the live REST API...</pre>
    </div>
  </div>

  <script>
    async function runRequest() {
      const city = encodeURIComponent(document.getElementById('city').value || 'New Delhi');
      const endpoint = document.getElementById('endpoint').value;
      const units = document.getElementById('units').value;
      const url = `/api/v1/weather/${endpoint}?city=${city}&units=${units}`;

      document.getElementById('requestUrl').textContent = `GET ${url}  [Header: X-API-Key: krmu-dev-key-2026]`;
      document.getElementById('output').textContent = 'Fetching live meteorological data...';

      const res = await fetch(url, { headers: { 'X-API-Key': 'krmu-dev-key-2026' } });
      const data = await res.json();

      document.getElementById('statusBadge').textContent = `HTTP Status: ${res.status}`;
      document.getElementById('cacheBadge').textContent = `Cache: ${data.meta?.cache || 'N/A'}`;
      document.getElementById('latencyBadge').textContent = `Latency: ${data.meta?.response_time_ms || '—'} ms`;
      document.getElementById('output').textContent = JSON.stringify(data, null, 2);
    }
    window.onload = runRequest;
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def developer_dashboard():
    """Serve the built-in interactive HTML/JS API testing console."""
    return DASHBOARD_HTML


# ==============================================================================
# 7. ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)