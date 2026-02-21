# dashboard.py
"""
Streamlit app (dense mode) with improved hourly forecast behavior and Arctic Spas API integration:
- Hourly starts at next full hour
- First hour labelled "Now" if within 60 minutes
- Feels-like calculation (wind chill / heat index) when possible
- Color-coded temps (cold/hot)
- Hourly icons restored + cached
- 5-day highs/lows present (with red low if below freezing)
- Enphase iframe (650px)
- Arctic Spas client integration (optional; requires arcticspas package and token)
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

# optional zoneinfo
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

# ---------------- Single-file Arctic Spas helpers (secrets-only) ----------------
# Cached client, status fetch, and example control helpers.
# These are safe to import early; actual generated-client imports are done lazily.

try:
    import arcticspas  # type: ignore
    ARCTICSPAS_INSTALLED = True
except Exception:
    ARCTICSPAS_INSTALLED = False

@st.cache_resource
def _get_spa_client_from_secrets() -> Optional[Any]:
    """
    Create and cache an arcticspas Client using st.secrets["arcticspa"].
    Returns None if package missing, token missing, or client creation fails.
    """
    if not ARCTICSPAS_INSTALLED:
        return None
    try:
        from arcticspas import Client  # local import to avoid hard failure at module import
    except Exception:
        return None
    try:
        secrets = st.secrets.get("arcticspa", {}) or {}
        token = secrets.get("token")
        base_url = secrets.get("base_url", ARCTIC_BASE)  # ARCTIC_BASE defined later; ok at call time
        if not token:
            return None
        client = Client(base_url=base_url, headers={"X-API-KEY": token})
        return client
    except Exception:
        return None

def fetch_spa_status_via_service() -> Dict[str, Any]:
    """
    Uses cached client and generated client's spa status operation.
    Returns dict: {ok, status_code, data, error}
    """
    client = _get_spa_client_from_secrets()
    if client is None:
        if not ARCTICSPAS_INSTALLED:
            return {"ok": False, "status_code": None, "data": None, "error": "arcticspas package not installed"}
        return {"ok": False, "status_code": None, "data": None, "error": "Client or token unavailable (check st.secrets['arcticspa'])"}

    # perform call using the generated client's typical pattern; keep imports local
    try:
        from arcticspas.api.spa_control import v2_spa  # adjust if your client exposes a different path
    except Exception:
        # try alternative import path used by some generated clients
        try:
            from arcticspas.operations import v2_spa  # type: ignore
        except Exception as exc:
            return {"ok": False, "status_code": None, "data": None, "error": f"Could not import spa operation: {exc}"}

    try:
        with client as c:
            resp = v2_spa.sync_detailed(client=c)
            status_code = getattr(resp, "status_code", None)
            parsed = getattr(resp, "parsed", None)
            data = None
            if parsed is None:
                # fallback to raw JSON if available
                try:
                    data = resp.json()  # type: ignore
                except Exception:
                    data = None
            else:
                try:
                    # prefer a model -> dict conversion if present
                    data = parsed.to_dict()
                except Exception:
                    try:
                        data = json.loads(json.dumps(parsed, default=lambda o: getattr(o, "__dict__", str(o))))
                    except Exception:
                        data = parsed
            ok = 200 <= (status_code or 0) < 400
            return {"ok": ok, "status_code": status_code, "data": data, "error": None if ok else f"HTTP {status_code}"}
    except Exception as exc:
        return {"ok": False, "status_code": None, "data": None, "error": str(exc)}

# Control helpers (examples). These attempt to import the likely modules and call sync_detailed.
# Adjust body field names if your API expects different shapes.
def _call_temperature_set(client, spa_id: str, temperature_c: float) -> Dict[str, Any]:
    try:
        from arcticspas.api import v2_temperature  # may raise
    except Exception:
        try:
            from arcticspas.operations import v2_temperature  # type: ignore
        except Exception as exc:
            return {"ok": False, "error": f"temperature op import failed: {exc}"}
    try:
        with client as c:
            body = {"target_temperature_c": temperature_c}
            resp = v2_temperature.sync_detailed(client=c, spa_id=spa_id, json_body=body)
            status_code = getattr(resp, "status_code", None)
            parsed = getattr(resp, "parsed", None)
            ok = 200 <= (status_code or 0) < 400
            return {"ok": ok, "status_code": status_code, "data": parsed, "error": None if ok else f"HTTP {status_code}"}
    except Exception as exc:
        return {"ok": False, "status_code": None, "data": None, "error": str(exc)}

def _call_light_set(client, spa_id: str, light_id: str, on: bool) -> Dict[str, Any]:
    try:
        from arcticspas.api import v2_light  # may raise
    except Exception:
        try:
            from arcticspas.operations import v2_light  # type: ignore
        except Exception as exc:
            return {"ok": False, "error": f"light op import failed: {exc}"}
    try:
        with client as c:
            body = {"state": "on" if on else "off"}
            resp = v2_light.sync_detailed(client=c, spa_id=spa_id, light_id=light_id, json_body=body)
            status_code = getattr(resp, "status_code", None)
            parsed = getattr(resp, "parsed", None)
            ok = 200 <= (status_code or 0) < 400
            return {"ok": ok, "status_code": status_code, "data": parsed, "error": None if ok else f"HTTP {status_code}"}
    except Exception as exc:
        return {"ok": False, "status_code": None, "data": None, "error": str(exc)}

def _call_pump_set(client, spa_id: str, pump_id: str, speed: int) -> Dict[str, Any]:
    try:
        from arcticspas.api import v2_pump  # may raise
    except Exception:
        try:
            from arcticspas.operations import v2_pump  # type: ignore
        except Exception as exc:
            return {"ok": False, "error": f"pump op import failed: {exc}"}
    try:
        with client as c:
            body = {"speed": speed}
            resp = v2_pump.sync_detailed(client=c, spa_id=spa_id, pump_id=pump_id, json_body=body)
            status_code = getattr(resp, "status_code", None)
            parsed = getattr(resp, "parsed", None)
            ok = 200 <= (status_code or 0) < 400
            return {"ok": ok, "status_code": status_code, "data": parsed, "error": None if ok else f"HTTP {status_code}"}
    except Exception as exc:
        return {"ok": False, "status_code": None, "data": None, "error": str(exc)}

# ---------------- Constants / Paths ----------------
CONFIG_PATH = Path(".user_config.json")
ICON_CACHE_DIR = Path(".cache_icons")
ICON_CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_ZIP = "84124"
ENPHASE_PUBLIC_URL = "https://enlighten.enphaseenergy.com/mobile/HRDg1683634/history/graph/hours?public=1"
ARCTIC_BASE = "https://api.myarcticspa.com"
ARCTIC_SPA_PORTAL_HINT = "https://myarcticspa.com/spa/SpaAPIManagement.aspx"

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
    else:
        if to_celsius:
            return int(round(t))
        else:
            return int(round((t * 9.0 / 5.0) + 32.0))

def parse_wind_mph(wind_str: str) -> Optional[float]:
    if not wind_str:
        return None
    try:
        import re
        nums = re.findall(r"[-+]?\d+\.?\d*", wind_str)
        if not nums:
            return None
        vals = [float(n) for n in nums]
        return sum(vals) / len(vals)
    except Exception:
        return None

def compute_wind_chill(temp_f: float, wind_mph: float) -> float:
    wc = 35.74 + 0.6215*temp_f - 35.75*(wind_mph**0.16) + 0.4275*temp_f*(wind_mph**0.16)
    return wc

def compute_heat_index(temp_f: float, rh: float) -> float:
    T = temp_f
    R = rh
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

# ---------------- App start ----------------
st.set_page_config(layout="wide", page_title="Forecast Dashboard", initial_sidebar_state="expanded")
st.markdown("""
<style>
h2 {
    font-size: 1.1rem !important;
    margin-bottom: 0.25rem !important;
}
</style>
""", unsafe_allow_html=True)
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
    st.markdown("---")
    # ----- Arctic Spa sidebar subsection (REPLACED: secrets-only) -----
    st.subheader("Arctic Spa")
    st.markdown("Open Arctic Spa site in a new tab, or call the API (requires token).")
    st.markdown(f'<a href="{ARCTIC_BASE}" target="_blank" rel="noopener noreferrer">Open Arctic Spa portal</a>', unsafe_allow_html=True)
    st.caption(f"API token can be obtained via the spa portal: {ARCTIC_SPA_PORTAL_HINT}")
    with st.expander("Local login form (manual only)"):
        st.info("This form is a local convenience only. It does not log into the external site.")
        username = st.text_input("Username", value="", key="local_arctic_username")
        password = st.text_input("Password", value="", type="password", key="local_arctic_password")
        st.checkbox("Remember username for this session", key="remember_local_user")
        if st.button("Show entered username"):
            st.success(f"Username entered: {username or '(empty)'}")
    st.markdown("---")
    st.subheader("Arctic Spas API (client)")
    if not ARCTICSPAS_INSTALLED:
        st.warning("Package 'arcticspas' not installed. Run 'pip install arcticspas' locally to enable API calls.")
    token_preview = bool((st.secrets.get("arcticspa", {}) or {}).get("token"))
    if token_preview:
        st.caption("Using token from Streamlit secrets.")
    else:
        st.warning("No Arctic Spa token found in st.secrets. Provide token to call the API.")
    if st.button("Fetch spa status (via secrets)"):
        result = fetch_spa_status_via_service()
        if result["ok"]:
            st.success(f"Status OK — HTTP {result['status_code']}")
            st.json(result["data"])
        else:
            st.error(f"Failed to fetch spa status: {result['error']}")
            if result.get("status_code"):
                st.write("HTTP status:", result["status_code"])

# ---------------- Layout: compact immediate forecast, compact 5-day, compact spa, Enphase ----------------

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
    parsed = []
    for h in hourly_periods:
        dt = parse_iso_to_dt(h.get("startTime"))
        if dt:
            parsed.append((to_user_tz(dt, user_tz), h))
    parsed.sort(key=lambda x: x[0])
    idx = 0
    for i, (dt_user, h) in enumerate(parsed):
        if dt_user > now_user:
            idx = i
            break
    else:
        idx = 0
    slice_items = parsed[idx: idx + NUM_HOURLY_TO_SHOW]
    if len(slice_items) < NUM_HOURLLY_TO_SHOW if False else False:
        # fallback handled below
        pass
    if len(slice_items) < NUM_HOURLLY_TO_SHOW if False else False:
        pass
    if len(slice_items) < NUM_HOURLLY_TO_SHOW if False else False:
        pass
    # fallback padding handled earlier in other code paths; return slice as-is
    return slice_items

def extract_pop(period: dict) -> Optional[int]:
    for key in ("probabilityOfPrecipitation", "pop", "probability"):
        v = period.get(key)
        if v is None:
            continue
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
    try:
        v = period.get("probabilityOfPrecipitation", {}).get("value")
        if v is not None:
            return int(float(v))
    except Exception:
        pass
    return None

def compute_feels_like_for_period(period: dict, to_celsius_flag: bool) -> Optional[int]:
    t = period.get("temperature")
    unit = period.get("temperatureUnit", "F")
    if t is None:
        return None
    try:
        temp_f = float(t) if unit.upper() == "F" else float(t)*9.0/5.0 + 32.0
    except Exception:
        return None
    wind_str = period.get("windSpeed") or ""
    wind_mph = parse_wind_mph(wind_str) or 0.0
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
    if temp_f <= 50 and wind_mph >= 3:
        try:
            feels_f = compute_wind_chill(temp_f, wind_mph)
        except Exception:
            feels_f = temp_f
    elif temp_f >= 80 and rh_val is not None:
        try:
            feels_f = compute_heat_index(temp_f, rh_val)
        except Exception:
            feels_f = temp_f
    else:
        feels_f = temp_f
    return convert_temp_for_display(feels_f, "F", to_celsius_flag)

# ---------- Compact UI CSS ----------
st.markdown(
    """
<style>
.compact-small { font-size: 0.86rem; line-height:1; }
.compact-label { font-size:0.92rem; font-weight:600; margin-bottom:4px; }
.compact-metric { font-size:1.05rem; font-weight:700; }
.compact-chip { display:inline-block; padding:4px 8px; border-radius:8px; font-size:0.85rem; margin-right:6px; background:#f1f1f1; }
.small-icon { width:48px; height:auto; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Immediate forecast (flex, no clipping) ----------
st.markdown("## Immediate forecast", unsafe_allow_html=True)
if weather_obj:
    now_user = to_user_tz(now_utc(), USER_TZ)
    hourly = weather_obj.get("forecastHourly", {}).get("properties", {}).get("periods", []) if weather_obj.get("forecastHourly") else []
    dayparts = weather_obj.get("forecast", {}).get("properties", {}).get("periods", []) if weather_obj.get("forecast") else []
    display_immediate = hourly if hourly else dayparts
    items = get_hourly_slice(display_immediate, now_user, USER_TZ) or []
    if not items:
        st.info("No immediate forecast items available.")
    else:
        cards = []
        for dt_user, it in items:
            label = "Now" if (dt_user - now_user).total_seconds() < 3600 and (dt_user - now_user).total_seconds() >= -300 else format_time_short(dt_user)
            temp = it.get("temperature")
            unit = it.get("temperatureUnit", "F")
            temp_disp = convert_temp_for_display(temp, unit, use_celsius)
            feels = compute_feels_like_for_period(it, use_celsius)
            pop = extract_pop(it)
            icon_url = it.get("icon") or get_cached_icon_path(it.get("icon") or "") or ""
            cold_threshold = 0 if use_celsius else 32
            hot_threshold = 29 if use_celsius else 85
            color = "#000"
            if temp_disp is not None:
                if temp_disp < cold_threshold:
                    color = "#1f77b4"
                elif temp_disp >= hot_threshold:
                    color = "#d62728"

            # no fixed min-height on the icon container; icon scales with max-width and max-height
            img_tag = f'<img src="{icon_url}" style="max-width:72%;width:auto;height:auto;display:block;margin:0 auto;object-fit:contain"/> ' if icon_url else ""
            feels_html = f'<div style="font-size:12px;margin-top:6px">Feels: {feels}°{"C" if use_celsius else "F"}</div>' if feels is not None and feels != temp_disp else ""
            pop_html = f'<div style="font-size:12px;margin-top:6px">POP: {pop}%</div>' if pop is not None else ""
            card = f'''
            <div style="display:flex;flex-direction:column;align-items:stretch;justify-content:flex-start;padding:8px 10px;flex:1;min-width:72px;box-sizing:border-box;">
              <div style="text-align:center;font-weight:600;font-size:0.92rem;margin-bottom:6px">{label}</div>
              <div style="display:flex;align-items:center;justify-content:center;">{img_tag}</div>
              <div style="text-align:center;font-weight:700;color:{color};margin-top:8px;font-size:1.02rem">{temp_disp if temp_disp is not None else 'N/A'}°{'C' if use_celsius else 'F'}</div>
              <div style="text-align:center">{feels_html}{pop_html}</div>
            </div>'''
            cards.append(card)

        html = f'''
        <div style="display:flex;gap:10px;align-items:flex-start;flex-wrap:nowrap;padding:6px 0;width:100%;box-sizing:border-box;overflow:visible">
          {''.join(cards)}
        </div>
        '''
        # increased height so the component won't crop content; adjust if you change icon sizing
        st.components.v1.html(html, height=220, scrolling=True)
else:
    st.info("No forecast available.")

# ---------- 5-day forecast (flex, no clipping) ----------
st.markdown("## 5-day (high / low)", unsafe_allow_html=True)
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

        cards = []
        for day in next_days:
            bucket = buckets.get(day, [])
            weekday = day.strftime("%a")
            # icon selection
            icon_url = None
            for it in bucket:
                p = it["period"]
                if p.get("isDaytime") and p.get("icon"):
                    icon_url = p.get("icon"); break
            if not icon_url:
                icons = [it["period"].get("icon") for it in bucket if it["period"].get("icon")]
                if icons:
                    try:
                        icon_url = max(set(icons), key=icons.count)
                    except Exception:
                        icon_url = icons[0]
            temps_display = []
            for item in bucket:
                p = item["period"]
                t = p.get("temperature")
                unit = p.get("temperatureUnit", "F")
                td = convert_temp_for_display(t, unit, use_celsius)
                if td is not None:
                    temps_display.append(td)
            high = max(temps_display) if temps_display else None
            low = min(temps_display) if temps_display else None
            img_tag = f'<img src="{icon_url}" style="max-width:72%;width:auto;height:auto;display:block;margin:0 auto;object-fit:contain"/>' if icon_url else ""
            ft = "C" if use_celsius else "F"
            low_html = f'<span style="color:red">{low}°{ft}</span>' if low is not None and ((low < 0 and use_celsius) or (low < 32 and not use_celsius)) else (f"{low}°{ft}" if low is not None else "N/A")
            high_html = f"{high}°{ft}" if high is not None else "N/A"
            short_texts = [p.get("shortForecast", "") for p in [it["period"] for it in bucket] if p.get("shortForecast")]
            common = short_texts and (max(set(short_texts), key=short_texts.count) if short_texts else "")
            card = f'''
            <div style="display:flex;flex-direction:column;align-items:stretch;justify-content:flex-start;padding:8px 10px;flex:1;min-width:72px;box-sizing:border-box;">
              <div style="text-align:center;font-weight:600;font-size:0.92rem;margin-bottom:6px">{weekday}</div>
              <div style="display:flex;align-items:center;justify-content:center;">{img_tag}</div>
              <div style="text-align:center;margin-top:8px;font-size:0.98rem">H <strong>{high_html}</strong></div>
              <div style="text-align:center;font-size:0.86rem">L {low_html}</div>
              <div style="text-align:center;font-size:11px;margin-top:6px;color:#444">{(common or '')[:36]}</div>
            </div>'''
            cards.append(card)

        html = f'''
        <div style="display:flex;gap:10px;align-items:flex-start;flex-wrap:nowrap;padding:6px 0;width:100%;box-sizing:border-box;overflow:visible">
          {''.join(cards)}
        </div>
        '''
        # increased height to avoid clipping; adjust if you change icon CSS
        st.components.v1.html(html, height=260, scrolling=True)
else:
    st.info("No forecast available.")
# ---------- Compact Arctic Spa status (compressed) ----------
st.markdown("---")
st.markdown("## Monisha's Tub — Live status")

# Use the secrets-only service-based fetch
spa_result = fetch_spa_status_via_service()

if not spa_result.get("ok"):
    st.info("Arctic Spas status not available: " + (spa_result.get("error") or "no token / failed request"))
else:
    spa = spa_result.get("data") or {}
    if not isinstance(spa, dict):
        try:
            spa = spa.to_dict() if hasattr(spa, "to_dict") else json.loads(json.dumps(spa, default=lambda o: getattr(o, "__dict__", str(o))))
        except Exception:
            try:
                spa = dict(spa)
            except Exception:
                spa = {}
    temp = spa.get("temperatureF") or spa.get("temperature") or spa.get("temp")
    setpoint = spa.get("setpointF") or spa.get("setpoint")
    connected = spa.get("connected")
    lights = spa.get("lights")
    pump1 = spa.get("pump1")
    pump2 = spa.get("pump2")
    pump3 = spa.get("pump3")
    filter_status = spa.get("filter_status") or spa.get("filterStatus")
    filtration_frequency = spa.get("filtration_frequency")
    filtration_duration = spa.get("filtration_duration")
    ph = spa.get("ph")
    ph_status = spa.get("ph_status")
    orp = spa.get("orp")
    orp_status = spa.get("orp_status")
    spaboy_connected = spa.get("spaboy_connected")
    spaboy_producing = spa.get("spaboy_producing")
    errors = spa.get("errors") or []

    # Row A: Temperature, setpoint, connection
    a1, a2, a3 = st.columns([1.4, 1, 1], gap="small")
    with a1:
        if temp is not None:
            st.markdown("<div class='compact-label'>Water temperature</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='compact-metric'>{temp} °F</div>", unsafe_allow_html=True)
            if setpoint is not None:
                st.markdown(f"<div class='compact-small'>Setpoint: {setpoint} °F</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='compact-small'>Temperature: N/A</div>", unsafe_allow_html=True)
    with a2:
        st.markdown("<div class='compact-label'>Connection</div>", unsafe_allow_html=True)
        conn = "🟢 Connected" if connected else "🔴 Disconnected"
        st.markdown(f"<div class='compact-small'>{conn}</div>", unsafe_allow_html=True)
        sb = []
        if spaboy_connected:
            sb.append("Spaboy connected")
        if spaboy_producing:
            sb.append("producing")
        if sb:
            st.markdown(f"<div class='compact-small'>{', '.join(sb)}</div>", unsafe_allow_html=True)
    with a3:
        st.markdown("<div class='compact-label'>Lights / Filter</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='compact-small'>Lights: {lights or 'unknown'}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='compact-small'>Filter: {filter_status or 'unknown'}</div>", unsafe_allow_html=True)

    # Row B: Pumps and filtration and chemistry
    bcols = st.columns(4, gap="small")
    with bcols[0]:
        st.markdown("<div class='compact-label'>Pumps</div>", unsafe_allow_html=True)
        st.markdown(f"<span class='compact-chip'>P1: {pump1 or 'unknown'}</span><span class='compact-chip'>P2: {pump2 or 'unknown'}</span><span class='compact-chip'>P3: {pump3 or 'unknown'}</span>", unsafe_allow_html=True)
    with bcols[1]:
        st.markdown("<div class='compact-label'>Filtration</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='compact-small'>Freq: {filtration_frequency or '?'} /day</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='compact-small'>Dur: {filtration_duration or '?'} min</div>", unsafe_allow_html=True)
    with bcols[2]:
        st.markdown("<div class='compact-label'>pH</div>", unsafe_allow_html=True)
        ph_text = f"{ph}" if ph is not None else "N/A"
        st.markdown(f"<div class='compact-small'>Value: {ph_text} ({ph_status or 'unknown'})</div>", unsafe_allow_html=True)
    with bcols[3]:
        st.markdown("<div class='compact-label'>ORP</div>", unsafe_allow_html=True)
        orp_text = f"{orp}" if orp is not None else "N/A"
        st.markdown(f"<div class='compact-small'>Value: {orp_text} ({orp_status or 'unknown'})</div>", unsafe_allow_html=True)

    # Errors (small)
    if errors:
        st.markdown("<div class='compact-label'>Errors / Alerts</div>", unsafe_allow_html=True)
        for e in errors:
            st.markdown(f"<div class='compact-small' style='color:#b00020'>&#9888; {e}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='compact-small'>No active errors.</div>", unsafe_allow_html=True)

    # raw toggle
    with st.expander("Raw spa payload (compact debug)"):
        st.json(spa)

# ---------- Enphase iframe (kept last) ----------
st.markdown("---")
st.markdown("## Enphase Solar Panels Info")
try:
    st.components.v1.iframe(ENPHASE_PUBLIC_URL, height=540)
except Exception:
    st.markdown(f'<a href="{ENPHASE_PUBLIC_URL}" target="_blank" rel="noopener noreferrer">Open Enphase hour graph (public)</a>', unsafe_allow_html=True)
st.markdown(f'If embedding is blocked, open in a new tab: <a href="{ENPHASE_PUBLIC_URL}" target="_blank" rel="noopener noreferrer">Open Enphase</a>', unsafe_allow_html=True)

# Footer debug
st.markdown("---")
st.caption("Compact layout: hourly starts at next full hour; first shown hour labeled 'Now' if within 60 minutes. Feels-like, POP, and color-coded temps included.")
if st.checkbox("Show debug"):
    st.write("config:", config)
    st.write("zip:", zip_to_use)
    st.write("last_auto_refresh:", st.session_state.get("last_auto_refresh"))
    st.write("arcticspas installed:", ARCTICSPAS_INSTALLED)