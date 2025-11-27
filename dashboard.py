# dashboard.py
"""
Streamlit app (dense mode) with improved hourly forecast behavior:
- Hourly starts at next full hour
- First hour labelled "Now" if within 60 minutes
- Feels-like calculation (wind chill / heat index) when possible
- Color-coded temps (cold/hot)
- Hourly icons restored + cached
- 5-day highs/lows present (with red low if below freezing)
- Enphase iframe (650px) + dense news below
Run:
    streamlit run dashboard.py
"""

from typing import Optional, Tuple, Dict, Any, List, Sequence
import streamlit as st
import requests
import os
import json
import sys
import time
import math
import hashlib
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta, date

# zoneinfo optional
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

# ---------------- Constants / Paths ----------------
CONFIG_PATH = Path(".user_config.json")
ICON_CACHE_DIR = Path(".cache_icons")
ICON_CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_ZIP = "84124"
DEFAULT_NEWS_API_KEY = "79c6795338c44f249006e46e2ab64456"
ENPHASE_PUBLIC_URL = "https://enlighten.enphaseenergy.com/mobile/HRDg1683634/history/graph/hours?public=1"

# ---------------- Utilities ----------------
def safe_rerun():
    try:
        if hasattr(st, "rerun"):
            st.rerun()
        elif hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
        else:
            raise AttributeError("st.rerun not available")
    except Exception:
        print("Rerun not available, restart app if needed.")
        sys.exit(0)

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def chunked_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

# ---------------- Network helper ----------------
def retry_request(method: str, url: str, **kwargs) -> requests.Response:
    max_attempts = 3
    backoff = 0.8
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.request(method, url, **kwargs)
            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                if attempt == max_attempts:
                    return resp
                time.sleep(backoff * attempt)
                continue
            return resp
        except requests.RequestException:
            if attempt == max_attempts:
                raise
            time.sleep(backoff * attempt)
    raise RuntimeError("Failed after retries")

# ---------------- Icon caching ----------------
@st.cache_data(ttl=86400)
def get_cached_icon_path(icon_url: str) -> Optional[str]:
    if not icon_url:
        return None
    try:
        h = chunked_hash(icon_url)
        suffix = os.path.splitext(icon_url.split("?")[0])[-1] or ".png"
        fname = ICON_CACHE_DIR / f"{h}{suffix}"
        if fname.exists():
            return str(fname)
        resp = retry_request("GET", icon_url, timeout=8, stream=True, headers={"User-Agent": "streamlit-icon-cache/1"})
        if resp.status_code == 200:
            with open(fname, "wb") as f:
                for chunk in resp.iter_content(1024):
                    if chunk:
                        f.write(chunk)
            return str(fname)
        return None
    except Exception:
        return None

# ---------------- Persistence ----------------
def load_config() -> Dict[str, Any]:
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_config(cfg: Dict[str, Any]) -> None:
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        st.error(f"Failed to save config: {e}")

def clear_config() -> None:
    try:
        if CONFIG_PATH.exists():
            CONFIG_PATH.unlink()
    except Exception as e:
        st.error(f"Failed to clear config: {e}")

config = load_config()

# ---------------- Time helpers ----------------
def parse_iso_to_dt(iso_ts: str) -> Optional[datetime]:
    if not iso_ts:
        return None
    try:
        if iso_ts.endswith("Z"):
            iso_ts = iso_ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(iso_ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        try:
            return datetime.fromisoformat(iso_ts)
        except Exception:
            return None

def to_user_tz(dt: datetime, user_tz):
    if dt is None:
        return dt
    try:
        return dt.astimezone(user_tz)
    except Exception:
        return dt.astimezone(timezone.utc)

def format_time_short(dt_user: datetime) -> str:
    try:
        # Use 12-hour without leading zero
        return dt_user.strftime("%-I %p")
    except Exception:
        return dt_user.strftime("%I %p").lstrip("0")

# ---------------- Temperature conversion & feels-like ----------------
def convert_temp_for_display(temp_value: Any, from_unit: str, to_celsius: bool) -> Optional[int]:
    try:
        t = float(temp_value)
    except Exception:
        return None
    if from_unit.upper() == "F":
        if to_celsius:
            return int(round((t - 32.0) * 5.0 / 9.0))
        else:
            return int(round(t))
    else:  # input in °C
        if to_celsius:
            return int(round(t))
        else:
            return int(round((t * 9.0 / 5.0) + 32.0))

def parse_wind_mph(wind_str: str) -> Optional[float]:
    """Attempt to extract a representative wind speed in mph from a string like '5 mph' or '5 to 10 mph'."""
    if not wind_str:
        return None
    try:
        # find all numbers
        import re
        nums = re.findall(r"[-+]?\d+\.?\d*", wind_str)
        if not nums:
            return None
        vals = [float(n) for n in nums]
        # if range, take average
        return sum(vals) / len(vals)
    except Exception:
        return None

def compute_wind_chill(temp_f: float, wind_mph: float) -> float:
    """Return wind chill in °F using standard formula (input temp in F)."""
    # formula valid for T <= 50°F and wind > 3 mph
    wc = 35.74 + 0.6215*temp_f - 35.75*(wind_mph**0.16) + 0.4275*temp_f*(wind_mph**0.16)
    return wc

def compute_heat_index(temp_f: float, rh: float) -> float:
    """Simple Rothfusz heat index approximation (input temp in °F and relative humidity in %)."""
    # use simplified formula
    T = temp_f
    R = rh
    # Rothfusz regression
    HI = (-42.379 + 2.04901523*T + 10.14333127*R - 0.22475541*T*R - 6.83783e-3*(T**2)
          - 5.481717e-2*(R**2) + 1.22874e-3*(T**2)*R + 8.5282e-4*T*(R**2)
          - 1.99e-6*(T**2)*(R**2))
    return HI

# ---------------- Cached API calls ----------------
@st.cache_data(ttl=86400)
def geocode_zip_to_latlon_cached(zip_code: str) -> Tuple[float, float]:
    resp = retry_request("GET", f"https://api.zippopotam.us/us/{zip_code}", timeout=8)
    if resp.status_code != 200:
        raise RuntimeError(f"Zippopotam failed for {zip_code}: {resp.status_code}")
    d = resp.json()
    place = d["places"][0]
    return float(place["latitude"]), float(place["longitude"])

@st.cache_data(ttl=600)
def get_nws_forecast_cached(lat: float, lon: float) -> dict:
    base = "https://api.weather.gov/points/{lat},{lon}"
    headers = {"User-Agent": "streamlit-nws-app (contact@example.com)"}
    r = retry_request("GET", base.format(lat=lat, lon=lon), headers=headers, timeout=10)
    r.raise_for_status()
    point = r.json()
    forecast_url = point["properties"].get("forecast")
    forecast_hourly_url = point["properties"].get("forecastHourly")
    forecast = None
    forecast_hourly = None
    if forecast_url:
        rf = retry_request("GET", forecast_url, headers=headers, timeout=10)
        rf.raise_for_status()
        forecast = rf.json()
    if forecast_hourly_url:
        rh = retry_request("GET", forecast_hourly_url, headers=headers, timeout=10)
        rh.raise_for_status()
        forecast_hourly = rh.json()
    return {"point": point, "forecast": forecast, "forecastHourly": forecast_hourly}

@st.cache_data(ttl=300)
def fetch_news_cached(topic: str, api_key: str, page_size: int = 8) -> List[dict]:
    url = "https://newsapi.org/v2/top-headlines"
    params = {"category": topic if topic != "general" else None, "country": "us", "pageSize": page_size}
    params = {k: v for k, v in params.items() if v is not None}
    headers = {"X-Api-Key": api_key}
    r = retry_request("GET", url, params=params, headers=headers, timeout=10)
    if r.status_code == 401:
        params_with_key = params.copy()
        params_with_key["apiKey"] = api_key
        r2 = retry_request("GET", url, params=params_with_key, timeout=10)
        if r2.status_code == 200:
            data = r2.json()
            if data.get("status") == "ok":
                return data.get("articles", [])
            raise RuntimeError(f"News API error (fallback): {data}")
        raise RuntimeError(f"News API 401 Unauthorized. {r.text} / fallback {r2.status_code}")
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "ok":
        raise RuntimeError(f"News API error payload: {data}")
    return data.get("articles", [])

# ---------------- App start ----------------
st.set_page_config(layout="wide", page_title="News + Forecast", initial_sidebar_state="expanded")
USER_TZ = ZoneInfo("America/Denver") if ZoneInfo else timezone.utc

# Session
if "last_auto_refresh" not in st.session_state:
    st.session_state["last_auto_refresh"] = time.time()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("Settings")

    st.subheader("ZIP (location)")
    saved_zip = config.get("zip_code", DEFAULT_ZIP)
    zip_input = st.text_input("ZIP code (US)", value=saved_zip)
    if st.button("Save ZIP"):
        if zip_input and zip_input.strip():
            try:
                _ = geocode_zip_to_latlon_cached(zip_input.strip())
                config["zip_code"] = zip_input.strip()
                save_config(config)
                st.success(f"Saved ZIP {zip_input.strip()}")
                safe_rerun()
            except Exception as e:
                st.error(str(e))
        else:
            st.info("Enter a ZIP first.")

    st.markdown("---")
    st.subheader("News")
    all_topics = ["general", "business", "entertainment", "health", "science", "sports", "technology"]
    selected_topics = st.multiselect("Select categories", options=all_topics, default=["general"])
    if "general" in selected_topics and len(selected_topics) > 1:
        st.warning("'General' selected — other choices ignored.")
        selected_topics = ["general"]
    topic_list = selected_topics

    api_key_env = os.getenv("NEWS_API_KEY", DEFAULT_NEWS_API_KEY)
    api_key_input = st.text_input("News API key", value=(api_key_env or DEFAULT_NEWS_API_KEY), type="password")
    num_headlines = st.slider("Headlines (combined)", 5, 100, 20)

    st.markdown("---")
    st.subheader("Display")
    unit_choice = st.radio("Temp unit", ["°F", "°C"], index=0)
    use_celsius = unit_choice == "°C"
    auto_refresh_enabled = st.checkbox("Auto-refresh", value=False)
    refresh_minutes = st.number_input("Refresh interval (min)", min_value=1, max_value=120, value=10)
    if st.button("Refresh now"):
        st.session_state["last_auto_refresh"] = time.time()
        safe_rerun()

    st.markdown("---")
    if st.button("Clear stored ZIP"):
        if "zip_code" in config:
            del config["zip_code"]
            save_config(config)
        st.success("Cleared stored ZIP.")
        safe_rerun()

    if st.button("Clear caches"):
        st.cache_data.clear()
        st.success("Caches cleared.")
        safe_rerun()

    st.markdown("---")
    st.subheader("Enphase (external)")
    st.markdown(f'<a href="{ENPHASE_PUBLIC_URL}" target="_blank" rel="noopener noreferrer">Open Enphase (public)</a>', unsafe_allow_html=True)

# ---------------- Layout: immediate forecast + daily highs/lows + enphase + news ----------------

# Resolve zip -> coords
zip_to_use = config.get("zip_code", DEFAULT_ZIP)
try:
    lat, lon = geocode_zip_to_latlon_cached(zip_to_use)
except Exception as e:
    st.error(f"Could not geocode ZIP {zip_to_use}: {e}")
    lat = lon = None

# Auto-refresh handling
if auto_refresh_enabled:
    last = st.session_state.get("last_auto_refresh", 0)
    if time.time() - last >= refresh_minutes * 60:
        st.session_state["last_auto_refresh"] = time.time()
        safe_rerun()

# Fetch weather early
weather_obj = None
weather_error = None
if lat is not None and lon is not None:
    try:
        weather_obj = get_nws_forecast_cached(lat, lon)
    except Exception as e:
        weather_error = str(e)

# ---------------- Hourly helper (updated behaviors 1-5) ----------------
NUM_HOURLY_TO_SHOW = 6

def get_hourly_slice(hourly_periods: List[dict], now_user: datetime, user_tz) -> List[Tuple[datetime, dict]]:
    """
    Return a list of (dt_user, period) starting at the next full hour.
    We pick the first period whose dt_user > now_user (strictly greater) -> next full hour.
    If that first chosen period is within 60 minutes of now_user, label as 'Now' in display.
    """
    parsed = []
    for h in hourly_periods:
        dt = parse_iso_to_dt(h.get("startTime"))
        if dt:
            parsed.append((to_user_tz(dt, user_tz), h))
    parsed.sort(key=lambda x: x[0])
    # find first dt strictly > now_user (next full hour)
    idx = 0
    for i, (dt_user, h) in enumerate(parsed):
        if dt_user > now_user:
            idx = i
            break
    else:
        # fallback: use last available block start
        idx = 0
    # slice next N slots
    slice_items = parsed[idx: idx + NUM_HOURLY_TO_SHOW]
    # if not enough items and we have earlier entries, pad with earliest
    if len(slice_items) < NUM_HOURLY_TO_SHOW:
        slice_items = parsed[:NUM_HOURLY_TO_SHOW]
    return slice_items

def extract_pop(period: dict) -> Optional[int]:
    """Extract probability of precipitation. NWS may provide 'probabilityOfPrecipitation' nested or 'pop' keys."""
    # direct keys
    for key in ("probabilityOfPrecipitation", "pop", "probability"):
        v = period.get(key)
        if v is None:
            continue
        # sometimes it's a dict with 'value' or numeric
        if isinstance(v, dict):
            val = v.get("value") or v.get("unitCode") or None
            try:
                return int(float(val))
            except Exception:
                continue
        try:
            return int(float(v))
        except Exception:
            continue
    # some payloads place probabilities in period['probabilityOfPrecipitation']['value']
    try:
        v = period.get("probabilityOfPrecipitation", {}).get("value")
        if v is not None:
            return int(float(v))
    except Exception:
        pass
    return None

def compute_feels_like_for_period(period: dict, to_celsius_flag: bool) -> Optional[int]:
    """Compute feels-like temp (display units). Return integer or None."""
    # period typically contains: temperature, temperatureUnit, windSpeed (string), relativeHumidity or humidity maybe absent
    t = period.get("temperature")
    unit = period.get("temperatureUnit", "F")
    if t is None:
        return None
    try:
        temp_f = float(t) if unit.upper() == "F" else float(t)*9.0/5.0 + 32.0
    except Exception:
        return None

    # parse wind speed
    wind_str = period.get("windSpeed") or ""
    wind_mph = parse_wind_mph(wind_str) or 0.0

    # RH may be in period.get('relativeHumidity') as dict or numeric
    rh = None
    rh_raw = period.get("relativeHumidity") or period.get("humidity") or period.get("probabilityOfPrecipitation")
    if isinstance(rh_raw, dict):
        rh = rh_raw.get("value")
    else:
        rh = rh_raw
    try:
        rh_val = float(rh) if rh is not None else None
    except Exception:
        rh_val = None

    feels_f = None
    # wind chill: T <= 50F and wind >= 3 mph
    if temp_f <= 50 and wind_mph >= 3:
        try:
            feels_f = compute_wind_chill(temp_f, wind_mph)
        except Exception:
            feels_f = temp_f
    # heat index: T >= 80F and RH available
    elif temp_f >= 80 and rh_val is not None:
        try:
            feels_f = compute_heat_index(temp_f, rh_val)
        except Exception:
            feels_f = temp_f
    else:
        feels_f = temp_f

    # convert to display unit
    return convert_temp_for_display(feels_f, "F", to_celsius_flag)

# ---------------- Compact horizontal immediate forecast bar (top) ----------------
st.markdown("## Immediate forecast")
if weather_obj:
    now_user = to_user_tz(now_utc(), USER_TZ)
    hourly = weather_obj.get("forecastHourly", {}).get("properties", {}).get("periods", []) if weather_obj.get("forecastHourly") else []
    dayparts = weather_obj.get("forecast", {}).get("properties", {}).get("periods", []) if weather_obj.get("forecast") else []

    display_immediate = hourly if hourly else dayparts
    immediate_items = get_hourly_slice(display_immediate, now_user, USER_TZ)

    if immediate_items:
        cols = st.columns(len(immediate_items), gap="small")
        for i, (dt_user, it) in enumerate(immediate_items):
            with cols[i]:
                # label: "Now" if within 60 minutes of now_user
                label = format_time_short(dt_user)
                if (dt_user - now_user).total_seconds() < 3600 and (dt_user - now_user).total_seconds() >= -300:
                    label = "Now"
                # icon
                icon_local = get_cached_icon_path(it.get("icon")) or it.get("icon")
                if icon_local:
                    try:
                        st.image(icon_local, width='content')
                    except Exception:
                        pass
                # temp and feels-like
                temp = it.get("temperature")
                unit = it.get("temperatureUnit", "F")
                temp_disp = convert_temp_for_display(temp, unit, use_celsius)
                feels = compute_feels_like_for_period(it, use_celsius)
                pop = extract_pop(it)
                # color code: cold (<32F/0C) blue, hot (>85F/29C) red
                cold_threshold = 0 if use_celsius else 32
                hot_threshold = 29 if use_celsius else 85
                color = "black"
                if temp_disp is not None:
                    if temp_disp < cold_threshold:
                        color = "blue"
                    elif temp_disp >= hot_threshold:
                        color = "red"
                # build markup
                disp_unit = "C" if use_celsius else "F"
                temp_html = f"<span style='color:{color};font-weight:600'>{temp_disp}°{disp_unit}</span>" if temp_disp is not None else "N/A"
                st.markdown(f"**{label}**")
                st.markdown(temp_html, unsafe_allow_html=True)
                if feels is not None and feels != temp_disp:
                    st.caption(f"Feels like: {feels}°{disp_unit}")
                if pop is not None:
                    st.caption(f"Precip: {pop}%")
    else:
        st.info("No immediate forecast items available.")
else:
    st.info("No forecast available.")

# ---------- 5-day daily high/low forecast (with icons restored) ----------
st.markdown("## 5-day forecast (high / low)")
if weather_obj:
    periods = weather_obj.get("forecast", {}).get("properties", {}).get("periods", []) if weather_obj.get("forecast") else []
    if not periods:
        st.info("No daypart forecast data available.")
    else:
        parsed_periods = []
        for p in periods:
            dt = parse_iso_to_dt(p.get("startTime") or p.get("endTime") or "")
            if not dt:
                continue
            dt_local = to_user_tz(dt, USER_TZ)
            parsed_periods.append((dt_local, p))
        buckets: Dict[date, List[dict]] = {}
        for dt_local, p in parsed_periods:
            day = dt_local.date()
            buckets.setdefault(day, []).append({"period": p, "dt": dt_local})
        today_local = to_user_tz(now_utc(), USER_TZ).date()
        next_days = [d for d in sorted(buckets.keys()) if d >= today_local][:5]
        if len(next_days) < 5:
            cand = today_local
            added = set(next_days)
            while len(next_days) < 5:
                cand = cand + timedelta(days=1)
                if cand in buckets and cand not in added:
                    next_days.append(cand)
                    added.add(cand)
                elif cand not in added and len(next_days) < 5:
                    next_days.append(cand)
                    added.add(cand)
                if len(added) > 10:
                    break

        def choose_icon_for_bucket(bucket_items: List[dict]) -> Optional[str]:
            for it in bucket_items:
                p = it["period"]
                if p.get("isDaytime") and p.get("icon"):
                    return p.get("icon")
            icons = [it["period"].get("icon") for it in bucket_items if it["period"].get("icon")]
            if not icons:
                return None
            try:
                return max(set(icons), key=icons.count)
            except Exception:
                return icons[0]

        freezing_threshold = 0 if use_celsius else 32

        cols = st.columns(len(next_days), gap="small")
        for i, day in enumerate(next_days):
            with cols[i]:
                weekday = day.strftime("%a")
                st.markdown(f"**{weekday}**")
                bucket = buckets.get(day, [])
                if not bucket:
                    st.write("N/A")
                    continue
                icon_url = choose_icon_for_bucket(bucket)
                if icon_url:
                    icon_local = get_cached_icon_path(icon_url) or icon_url
                    if icon_local:
                        try:
                            st.image(icon_local, width='content')
                        except Exception:
                            pass
                temps_display = []
                for item in bucket:
                    p = item["period"]
                    t = p.get("temperature")
                    unit = p.get("temperatureUnit", "F")
                    td = convert_temp_for_display(t, unit, use_celsius)
                    if td is not None:
                        temps_display.append(td)
                if not temps_display:
                    st.write("N/A")
                    continue
                high = max(temps_display)
                low = min(temps_display)
                disp_unit = "C" if use_celsius else "F"
                low_html = f"<span style='color:red'>{low}°{disp_unit}</span>" if low < freezing_threshold else f"{low}°{disp_unit}"
                st.markdown(f"High: **{high}°{disp_unit}**  ")
                st.markdown(f"Low: {low_html}", unsafe_allow_html=True)
                short_texts = [p.get("shortForecast", "") for p in [it["period"] for it in bucket] if p.get("shortForecast")]
                if short_texts:
                    try:
                        common = max(set(short_texts), key=short_texts.count)
                    except Exception:
                        common = short_texts[0]
                    st.caption(common[:80])
else:
    st.info("No forecast available.")

# ---------- Enphase iframe (restored original size) ----------
st.markdown("---")
st.markdown("## Enphase (embedded if allowed)")
try:
    st.components.v1.iframe(ENPHASE_PUBLIC_URL, height=650)
except Exception:
    st.markdown(f'<a href="{ENPHASE_PUBLIC_URL}" target="_blank" rel="noopener noreferrer">Open Enphase hour graph (public)</a>', unsafe_allow_html=True)
st.markdown(f'If embedding is blocked, open in a new tab: <a href="{ENPHASE_PUBLIC_URL}" target="_blank" rel="noopener noreferrer">Open Enphase</a>', unsafe_allow_html=True)

# ---------- Dense news list (below enphase) ----------
st.markdown("---")
st.markdown("## Headlines")

# Retrieve News API key and settings from sidebar/session
news_api_key = None
try:
    news_api_key = st.session_state.get("News API key") or st.session_state.get("news_api_key")
except Exception:
    news_api_key = None
if not news_api_key:
    news_api_key = os.getenv("NEWS_API_KEY", DEFAULT_NEWS_API_KEY)

combined = []
errors = []
try:
    sidebar_topics = topic_list
except Exception:
    sidebar_topics = ["general"]
try:
    sidebar_num = num_headlines
except Exception:
    sidebar_num = 20

if news_api_key:
    for t in sidebar_topics:
        try:
            arts = fetch_news_cached(t, news_api_key, page_size=sidebar_num)
            combined.extend(arts)
        except Exception as e:
            errors.append(str(e))
else:
    errors.append("No News API key set.")

if errors:
    for e in errors:
        st.info(e)

def dedupe_and_sort(articles: Sequence[dict], limit: int) -> List[dict]:
    seen: Dict[str, dict] = {}
    for art in articles:
        key = (art.get("url") or art.get("title") or "").strip()
        pub = art.get("publishedAt")
        dt = parse_iso_to_dt(pub) if pub else datetime.fromtimestamp(0, tz=timezone.utc)
        art_copy = dict(art)
        art_copy["_parsed_pub"] = dt or datetime.fromtimestamp(0, tz=timezone.utc)
        if key not in seen or art_copy["_parsed_pub"] > seen[key]["_parsed_pub"]:
            seen[key] = art_copy
    items = list(seen.values())
    items.sort(key=lambda x: x.get("_parsed_pub", datetime.fromtimestamp(0, tz=timezone.utc)), reverse=True)
    for it in items:
        it.pop("_parsed_pub", None)
    return items[:limit]

articles = dedupe_and_sort(combined, sidebar_num)

# Render dense list
if not articles:
    st.info("No headlines to show.")
else:
    for art in articles:
        cols = st.columns([0.8, 9], gap="small")
        with cols[0]:
            if art.get("urlToImage"):
                img_local = get_cached_icon_path(art.get("urlToImage")) or art.get("urlToImage")
                try:
                    st.image(img_local, width='content')
                except Exception:
                    pass
        with cols[1]:
            title = art.get("title") or ""
            url = art.get("url") or ""
            src = art.get("source", {}).get("name", "")
            pub = art.get("publishedAt") or ""
            st.markdown(f"**[{title}]({url})**")
            meta = " • ".join([s for s in [src, pub[:10]] if s])
            if meta:
                st.caption(meta)
            desc = art.get("description") or art.get("content") or ""
            if desc:
                st.write(desc[:200])

# Footer debug
st.markdown("---")
st.caption("Dense layout: hourly starts at next full hour; first shown hour labeled 'Now' if within 60 minutes. Feels-like, POP, and color-coded temps included.")
if st.checkbox("Show debug"):
    st.write("config:", config)
    st.write("zip:", zip_to_use)
    st.write("last_auto_refresh:", st.session_state.get("last_auto_refresh"))
