# dashboard.py
"""
Streamlit app (dense mode) with improved hourly forecast behavior and Arctic Spas API integration.
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


# ---------- ntfy.sh alert config & helper ----------
NTFY_TOPIC  = st.secrets.get("NTFY_TOPIC",  None) or os.environ.get("NTFY_TOPIC")  or "monishas-tub-alerts"
NTFY_SERVER = st.secrets.get("NTFY_SERVER", None) or os.environ.get("NTFY_SERVER") or "https://ntfy.sh"

def _ascii_safe(s: str) -> str:
    return s.encode("latin-1", errors="ignore").decode("latin-1")

def send_ntfy_message(topic, title, message, server=NTFY_SERVER, priority="high", tags="warning,bathtub"):
    if not topic:
        raise ValueError("NTFY_TOPIC is not set.")
    url = f"{server.rstrip('/')}/{topic}"
    headers = {
        "Title": _ascii_safe(title), "Priority": priority,
        "Tags": tags, "Content-Type": "text/plain; charset=utf-8",
    }
    resp = requests.post(url, data=message.encode("utf-8"), headers=headers, timeout=15)
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"ntfy returned HTTP {resp.status_code}: {resp.text[:200]}")

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

try:
    import arcticspas
    ARCTICSPAS_INSTALLED = True
except Exception:
    ARCTICSPAS_INSTALLED = False

@st.cache_resource
def _get_spa_config_from_secrets():
    try:
        secrets = st.secrets.get("arcticspa", {}) or {}
        token = secrets.get("token")
        base_url = secrets.get("base_url", ARCTIC_BASE)
        if not token:
            return None
        return {"token": token, "base_url": base_url}
    except Exception:
        return None

def fetch_spa_status_via_service():
    cfg = _get_spa_config_from_secrets()
    if cfg is None:
        if not ARCTICSPAS_INSTALLED:
            return {"ok": False, "status_code": None, "data": None, "error": "arcticspas package not installed"}
        return {"ok": False, "status_code": None, "data": None, "error": "Client config or token unavailable"}
    try:
        from arcticspas.api.spa_control import v2_spa
    except Exception:
        try:
            from arcticspas.operations import v2_spa
        except Exception as exc:
            return {"ok": False, "status_code": None, "data": None, "error": f"Could not import spa operation: {exc}"}
    try:
        from arcticspas import Client
        client = Client(base_url=cfg["base_url"], headers={"X-API-KEY": cfg["token"]})
    except Exception as exc:
        return {"ok": False, "status_code": None, "data": None, "error": f"Failed to construct client: {exc}"}
    try:
        with client as c:
            resp = v2_spa.sync_detailed(client=c)
            status_code = getattr(resp, "status_code", None)
            parsed = getattr(resp, "parsed", None)
            data = None
            if parsed is None:
                try:    data = resp.json()
                except: data = None
            else:
                try:    data = parsed.to_dict()
                except:
                    try:    data = json.loads(json.dumps(parsed, default=lambda o: getattr(o, "__dict__", str(o))))
                    except: data = parsed
            ok = 200 <= (status_code or 0) < 400
            return {"ok": ok, "status_code": status_code, "data": data, "error": None if ok else f"HTTP {status_code}"}
    except Exception as exc:
        return {"ok": False, "status_code": None, "data": None, "error": str(exc)}


def _call_temperature_set(spa_id, temperature_c):
    cfg = _get_spa_config_from_secrets()
    if cfg is None:
        return {"ok": False, "error": "Client config/token missing"}
    try:
        from arcticspas.api import v2_temperature
    except Exception:
        try:
            from arcticspas.operations import v2_temperature
        except Exception as exc:
            return {"ok": False, "error": f"temperature op import failed: {exc}"}
    try:
        from arcticspas import Client
        client = Client(base_url=cfg["base_url"], headers={"X-API-KEY": cfg["token"]})
        with client as c:
            body = {"target_temperature_c": temperature_c}
            resp = v2_temperature.sync_detailed(client=c, spa_id=spa_id, json_body=body)
            status_code = getattr(resp, "status_code", None)
            ok = 200 <= (status_code or 0) < 400
            return {"ok": ok, "status_code": status_code, "data": getattr(resp, "parsed", None), "error": None if ok else f"HTTP {status_code}"}
    except Exception as exc:
        return {"ok": False, "status_code": None, "data": None, "error": str(exc)}


def _call_light_set(spa_id, light_id, on):
    cfg = _get_spa_config_from_secrets()
    if cfg is None:
        return {"ok": False, "error": "Client config/token missing"}
    try:
        from arcticspas.api import v2_light
    except Exception:
        try:
            from arcticspas.operations import v2_light
        except Exception as exc:
            return {"ok": False, "error": f"light op import failed: {exc}"}
    try:
        from arcticspas import Client
        client = Client(base_url=cfg["base_url"], headers={"X-API-KEY": cfg["token"]})
        with client as c:
            body = {"state": "on" if on else "off"}
            resp = v2_light.sync_detailed(client=c, spa_id=spa_id, light_id=light_id, json_body=body)
            status_code = getattr(resp, "status_code", None)
            ok = 200 <= (status_code or 0) < 400
            return {"ok": ok, "status_code": status_code, "data": getattr(resp, "parsed", None), "error": None if ok else f"HTTP {status_code}"}
    except Exception as exc:
        return {"ok": False, "status_code": None, "data": None, "error": str(exc)}


def _call_pump_set(spa_id, pump_id, speed):
    cfg = _get_spa_config_from_secrets()
    if cfg is None:
        return {"ok": False, "error": "Client config/token missing"}
    try:
        from arcticspas.api import v2_pump
    except Exception:
        try:
            from arcticspas.operations import v2_pump
        except Exception as exc:
            return {"ok": False, "error": f"pump op import failed: {exc}"}
    try:
        from arcticspas import Client
        client = Client(base_url=cfg["base_url"], headers={"X-API-KEY": cfg["token"]})
        with client as c:
            body = {"speed": speed}
            resp = v2_pump.sync_detailed(client=c, spa_id=spa_id, pump_id=pump_id, json_body=body)
            status_code = getattr(resp, "status_code", None)
            ok = 200 <= (status_code or 0) < 400
            return {"ok": ok, "status_code": status_code, "data": getattr(resp, "parsed", None), "error": None if ok else f"HTTP {status_code}"}
    except Exception as exc:
        return {"ok": False, "status_code": None, "data": None, "error": str(exc)}

# ---------------- Constants / Paths ----------------
CONFIG_PATH    = Path(".user_config.json")
ICON_CACHE_DIR = Path(".cache_icons")
ICON_CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_ZIP         = "84124"
ENPHASE_PUBLIC_URL  = "https://enlighten.enphaseenergy.com/mobile/HRDg1683634/history/graph/hours?public=1"
ARCTIC_BASE         = "https://api.myarcticspa.com"
ARCTIC_SPA_PORTAL_HINT = "https://myarcticspa.com/spa/SpaAPIManagement.aspx"

# Chemistry safe ranges
PH_MIN,   PH_MAX   = 7.2, 7.8
ORP_MIN,  ORP_MAX  = 600, 800
TEMP_MIN_F, TEMP_MAX_F = 95.0, 104.0

# ---------------- Utilities ----------------
def safe_rerun():
    try:
        if hasattr(st, "rerun"):            st.rerun()
        elif hasattr(st, "experimental_rerun"): st.experimental_rerun()
        else: raise AttributeError()
    except Exception:
        sys.exit(0)

def now_utc(): return datetime.now(timezone.utc)
def chunked_hash(s): return hashlib.sha1(s.encode("utf-8")).hexdigest()

def retry_request(method, url, **kwargs):
    max_attempts, backoff = 3, 0.8
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.request(method, url, **kwargs)
            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                if attempt == max_attempts: return resp
                time.sleep(backoff * attempt); continue
            return resp
        except requests.RequestException:
            if attempt == max_attempts: raise
            time.sleep(backoff * attempt)
    raise RuntimeError("Failed after retries")

@st.cache_data(ttl=86400)
def get_cached_icon_path(icon_url):
    if not icon_url: return None
    try:
        h = chunked_hash(icon_url)
        suffix = os.path.splitext(icon_url.split("?")[0])[-1] or ".png"
        fname = ICON_CACHE_DIR / f"{h}{suffix}"
        if fname.exists(): return str(fname)
        resp = retry_request("GET", icon_url, timeout=8, stream=True, headers={"User-Agent": "streamlit-icon-cache/1"})
        if resp.status_code == 200:
            with open(fname, "wb") as f:
                for chunk in resp.iter_content(1024):
                    if chunk: f.write(chunk)
            return str(fname)
        return None
    except Exception:
        return None

def load_config():
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_config(cfg):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        st.error(f"Failed to save config: {e}")

def clear_config():
    try:
        if CONFIG_PATH.exists(): CONFIG_PATH.unlink()
    except Exception as e:
        st.error(f"Failed to clear config: {e}")

config = load_config()

# ============================================================
# PERSISTENT CHEMISTRY RANGE — survives browser refresh
# ============================================================
# Stored in .user_config.json under key "chem_ranges":
#   { "date": "YYYY-MM-DD",
#     "temp_min": float, "temp_max": float,
#     "ph_min": float,   "ph_max": float,
#     "orp_min": float,  "orp_max": float }

def _load_chem_ranges(user_tz) -> Dict[str, Any]:
    """Load today's chem ranges from config, resetting if date has changed."""
    today = datetime.now(user_tz).strftime("%Y-%m-%d")
    stored = config.get("chem_ranges", {})
    if not isinstance(stored, dict) or stored.get("date") != today:
        return {"date": today}
    return stored

def _save_chem_ranges(ranges: Dict[str, Any]) -> None:
    config["chem_ranges"] = ranges
    save_config(config)

def update_chem_ranges_persistent(temp_f, ph, orp, user_tz) -> None:
    """Update persistent daily min/max for temp, ph, orp."""
    ranges = _load_chem_ranges(user_tz)
    changed = False
    for key, val in (("temp", temp_f), ("ph", ph), ("orp", orp)):
        if val is None:
            continue
        try:
            val = float(val)
        except Exception:
            continue
        lo_key, hi_key = f"{key}_min", f"{key}_max"
        cur_lo = ranges.get(lo_key)
        cur_hi = ranges.get(hi_key)
        new_lo = val if cur_lo is None else min(cur_lo, val)
        new_hi = val if cur_hi is None else max(cur_hi, val)
        if new_lo != cur_lo or new_hi != cur_hi:
            ranges[lo_key] = new_lo
            ranges[hi_key] = new_hi
            changed = True
    if changed:
        _save_chem_ranges(ranges)

# ---- Keep session_state in sync (for same-session display) ----
def _sync_chem_session(user_tz):
    """Copy persistent ranges into session_state for display functions."""
    ranges = _load_chem_ranges(user_tz)
    for key in ("temp", "ph", "orp"):
        st.session_state[f"chem_min_{key}"] = ranges.get(f"{key}_min")
        st.session_state[f"chem_max_{key}"] = ranges.get(f"{key}_max")


def parse_iso_to_dt(iso_ts):
    if not iso_ts: return None
    try:
        if iso_ts.endswith("Z"): iso_ts = iso_ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(iso_ts)
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
    except Exception:
        try:    return datetime.fromisoformat(iso_ts)
        except: return None

def to_user_tz(dt, user_tz):
    if dt is None: return dt
    try:    return dt.astimezone(user_tz)
    except: return dt.astimezone(timezone.utc)

def format_time_short(dt_user):
    try:    return dt_user.strftime("%-I %p")
    except: return dt_user.strftime("%I %p").lstrip("0")

def convert_temp_for_display(temp_value, from_unit, to_celsius):
    try:    t = float(temp_value)
    except: return None
    if from_unit.upper() == "F":
        return int(round((t - 32.0) * 5.0 / 9.0)) if to_celsius else int(round(t))
    else:
        return int(round(t)) if to_celsius else int(round((t * 9.0 / 5.0) + 32.0))

def parse_wind_mph(wind_str):
    if not wind_str: return None
    try:
        import re
        nums = re.findall(r"[-+]?\d+\.?\d*", wind_str)
        if not nums: return None
        vals = [float(n) for n in nums]
        return sum(vals) / len(vals)
    except: return None

def compute_wind_chill(temp_f, wind_mph):
    return 35.74 + 0.6215*temp_f - 35.75*(wind_mph**0.16) + 0.4275*temp_f*(wind_mph**0.16)

def compute_heat_index(temp_f, rh):
    T, R = temp_f, rh
    return (-42.379 + 2.04901523*T + 10.14333127*R - 0.22475541*T*R - 6.83783e-3*(T**2)
            - 5.481717e-2*(R**2) + 1.22874e-3*(T**2)*R + 8.5282e-4*T*(R**2) - 1.99e-6*(T**2)*(R**2))

@st.cache_data(ttl=86400)
def geocode_zip_to_latlon_cached(zip_code):
    resp = retry_request("GET", f"https://api.zippopotam.us/us/{zip_code}", timeout=8)
    if resp.status_code != 200:
        raise RuntimeError(f"Zippopotam failed for {zip_code}: {resp.status_code}")
    d = resp.json()
    place = d["places"][0]
    return float(place["latitude"]), float(place["longitude"])

@st.cache_data(ttl=600)
def get_nws_forecast_cached(lat, lon):
    base    = "https://api.weather.gov/points/{lat},{lon}"
    headers = {"User-Agent": "streamlit-nws-app (contact@example.com)"}
    r = retry_request("GET", base.format(lat=lat, lon=lon), headers=headers, timeout=10)
    r.raise_for_status()
    point = r.json()
    forecast_url        = point["properties"].get("forecast")
    forecast_hourly_url = point["properties"].get("forecastHourly")
    forecast = forecast_hourly = None
    if forecast_url:
        rf = retry_request("GET", forecast_url, headers=headers, timeout=10)
        rf.raise_for_status(); forecast = rf.json()
    if forecast_hourly_url:
        rh = retry_request("GET", forecast_hourly_url, headers=headers, timeout=10)
        rh.raise_for_status(); forecast_hourly = rh.json()
    return {"point": point, "forecast": forecast, "forecastHourly": forecast_hourly}


# ============================================================
# PUMP RUNTIME TRACKING — persisted to disk
# ============================================================
# Stored in .user_config.json under key "pump_runtimes":
#   {
#     "date": "YYYY-MM-DD",          # resets daily
#     "1": { "daily_seconds": float, "on_since": float|null, "last_state": str|null },
#     "2": { ... },
#     "3": { ... }
#   }
#
# "on_since" is a Unix timestamp written to disk when the pump turns on.
# On the next load (even after a browser refresh), we know the pump was
# already running since that timestamp, so we correctly accumulate elapsed
# time without losing anything between refreshes.
#
# State machine per pump (new_on vs prev_on from persisted last_state):
#   off→on  : record on_since = now
#   on→on   : daily_seconds += (now - on_since), reset on_since = now
#   on→off  : daily_seconds += (now - on_since), clear on_since
#   off→off : nothing
# ============================================================

def _is_pump_on(val):
    if val is None: return False
    s = str(val).strip().lower()
    if s in ("on","1","true","running","high","low","medium"): return True
    try:    return float(s) > 0
    except: return False

def _today_str(user_tz): return datetime.now(user_tz).strftime("%Y-%m-%d")

def _load_pump_runtimes(user_tz) -> Dict[str, Any]:
    """Load today's pump runtime state from config, resetting if date rolled over."""
    today  = _today_str(user_tz)
    stored = config.get("pump_runtimes", {})
    if not isinstance(stored, dict) or stored.get("date") != today:
        # New day — start fresh but keep the structure
        return {"date": today,
                "1": {"daily_seconds": 0.0, "on_since": None, "last_state": None},
                "2": {"daily_seconds": 0.0, "on_since": None, "last_state": None},
                "3": {"daily_seconds": 0.0, "on_since": None, "last_state": None}}
    # Ensure all pump keys exist (defensive)
    for n in ["1","2","3"]:
        stored.setdefault(n, {"daily_seconds": 0.0, "on_since": None, "last_state": None})
    return stored

def _save_pump_runtimes(runtimes: Dict[str, Any]) -> None:
    config["pump_runtimes"] = runtimes
    save_config(config)

def update_pump_runtimes(pump_vals: Dict[int, Any], user_tz) -> None:
    """
    Call once per app load. pump_vals: {1: pump1_val, 2: pump2_val, 3: pump3_val}
    Reads persisted state, applies transitions, writes back to disk.
    Also syncs session_state for get_pump_runtime_display().
    """
    now_ts   = time.time()
    runtimes = _load_pump_runtimes(user_tz)
    changed  = False

    for n in [1, 2, 3]:
        key      = str(n)
        entry    = runtimes[key]
        new_val  = pump_vals.get(n)
        new_on   = _is_pump_on(new_val)
        prev_on  = _is_pump_on(entry.get("last_state"))
        on_since = entry.get("on_since")          # persisted Unix timestamp or None
        daily    = float(entry.get("daily_seconds", 0.0))

        if new_on and prev_on:
            # Was on, still on — accumulate time since last checkpoint
            if on_since is not None:
                daily += now_ts - on_since
            entry["daily_seconds"] = daily
            entry["on_since"]      = now_ts       # reset checkpoint to now
            changed = True

        elif new_on and not prev_on:
            # Just turned on (or first load after refresh while pump is on)
            # If on_since is already set from a previous session, honour it
            # (pump was already running before refresh — don't double-count)
            if on_since is None:
                entry["on_since"] = now_ts
                changed = True
            # else: on_since already persisted, no change needed

        elif not new_on and prev_on:
            # Just turned off — finalise
            if on_since is not None:
                daily += now_ts - on_since
            entry["daily_seconds"] = daily
            entry["on_since"]      = None
            changed = True

        # off→off: nothing to update

        entry["last_state"] = str(new_val) if new_val is not None else None

        # Mirror into session_state for get_pump_runtime_display()
        st.session_state[f"pump_daily_seconds_{n}"] = entry["daily_seconds"]
        st.session_state[f"pump_on_since_{n}"]      = entry["on_since"]
        st.session_state[f"pump_state_{n}"]         = entry["last_state"]

    if changed:
        _save_pump_runtimes(runtimes)

def get_pump_runtime_display(n: int) -> str:
    """Return formatted runtime string, including live (un-checkpointed) time."""
    now_ts    = time.time()
    daily_sec = float(st.session_state.get(f"pump_daily_seconds_{n}", 0))
    on_since  = st.session_state.get(f"pump_on_since_{n}")
    if on_since is not None:
        daily_sec += now_ts - on_since      # live accumulation since last checkpoint
    hours   = int(daily_sec // 3600)
    minutes = int((daily_sec % 3600) // 60)
    return f"{hours}h {minutes:02d}m" if hours > 0 else f"{minutes}m"


def chem_inline_html(label, value, unit, safe_lo, safe_hi, day_min, day_max, fmt=".1f"):
    if value is None:
        return f'<span style="color:#6a8eaa;font-size:0.82rem">{label}: N/A</span>'
    in_range  = safe_lo <= value <= safe_hi
    val_color = "#6effa8" if in_range else "#ff6060"
    val_str   = f"{value:{fmt}}"
    if day_min is not None and day_max is not None:
        range_str = f"{day_min:{fmt}}–{day_max:{fmt}}"
    else:
        range_str = "—"
    return (f'<span style="font-size:0.82rem;color:#a0c8e8">{label}:</span> '
            f'<span style="font-weight:700;color:{val_color}">{val_str}{unit}</span> '
            f'<span style="font-size:0.74rem;color:#4a6a80">[{range_str}]</span>')


# ============================================================
# App start
# ============================================================
st.set_page_config(layout="wide", page_title="Forecast Dashboard", initial_sidebar_state="expanded")

st.markdown("""
<style>
h2 { font-size: 1.1rem !important; margin-top: 6px !important; margin-bottom: 6px !important; }
.stMarkdown p { margin: 6px 0 !important; }
.block-container > div { padding-top: 6px !important; padding-bottom: 6px !important; }
.responsive-grid { display: grid; gap: 10px;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  align-items: start; width: 100%; box-sizing: border-box; padding: 6px 0; margin: 0; }
.responsive-card { display:flex; flex-direction:column; align-items:stretch;
  justify-content:flex-start; padding:8px 10px; box-sizing:border-box;
  min-width: 120px; margin-bottom: 4px; }
.frame { border: 1px solid rgba(0,0,0,0.08); border-radius: 10px; padding: 12px;
  box-shadow: 0 1px 6px rgba(0,0,0,0.06); background: #e6f7ff; margin-bottom: 8px; }
.frame.compact { padding: 8px; border-radius: 8px; margin-bottom:6px; }
.spa-card { background: linear-gradient(135deg, #1a3a5c 0%, #0d2137 100%);
  border-radius: 14px; padding: 18px 20px; color: #fff;
  margin-bottom: 10px; box-shadow: 0 4px 18px rgba(0,0,0,0.18); }
.spa-card .spa-title { font-size: 1.05rem; font-weight: 700; color: #7ecbf7;
  letter-spacing: 0.03em; margin-bottom: 12px; }
.spa-temp-big { font-size: 2.6rem; font-weight: 800; color: #fff; line-height: 1; }
.spa-temp-sub { font-size: 0.88rem; color: #a0c8e8; margin-top: 4px; }
.spa-badge { display: inline-block; padding: 4px 10px; border-radius: 20px;
  font-size: 0.8rem; font-weight: 600; margin: 3px 3px 3px 0; }
.spa-badge-ok   { background: rgba(40,200,100,0.22);  color: #6effa8; border: 1px solid rgba(40,200,100,0.4); }
.spa-badge-warn { background: rgba(255,160,0,0.22);   color: #ffd060; border: 1px solid rgba(255,160,0,0.4); }
.spa-badge-err  { background: rgba(220,60,60,0.22);   color: #ff8080; border: 1px solid rgba(220,60,60,0.4); }
.spa-badge-off  { background: rgba(120,120,120,0.22); color: #ccc;    border: 1px solid rgba(120,120,120,0.3); }
.spa-section-label { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.07em;
  color: #7ecbf7; margin-bottom: 5px; margin-top: 12px; font-weight: 600; }
.spa-divider { border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 10px 0; }
.alert-banner { background: linear-gradient(90deg, #7b1a1a, #b02020);
  border-radius: 8px; padding: 10px 14px; color: #fff;
  font-weight: 600; margin-bottom: 10px; display: flex; align-items: center; gap: 8px; }
.signal-badge { display: inline-block; background: #2d8cff; color: #fff;
  border-radius: 6px; padding: 2px 8px; font-size: 0.8rem; font-weight: 700; letter-spacing: 0.04em; }
hr { margin: 6px 0 !important; height: 1px !important; border: none; background: #eee; }

/* Connection status dot */
.conn-dot { display:inline-block; width:11px; height:11px; border-radius:50%;
  vertical-align:middle; margin-right:5px; box-shadow: 0 0 6px currentColor; }
.conn-dot-ok  { background:#3ddc84; color:#3ddc84; }
.conn-dot-err { background:#ff5555; color:#ff5555; }
</style>
""", unsafe_allow_html=True)

USER_TZ = ZoneInfo("America/Denver") if ZoneInfo else timezone.utc

if "last_auto_refresh" not in st.session_state:
    st.session_state["last_auto_refresh"] = time.time()

# Initialise runtime session keys
for _n in [1, 2, 3]:
    for _k in [f"pump_daily_seconds_{_n}", f"pump_on_since_{_n}", f"pump_state_{_n}"]:
        if _k not in st.session_state:
            st.session_state[_k] = None if ("_state_" in _k or "_since_" in _k) else 0
if "pump_runtime_date" not in st.session_state:
    st.session_state["pump_runtime_date"] = ""
for _k in ("temp", "ph", "orp"):
    for _prefix in ("chem_min_", "chem_max_"):
        if _prefix + _k not in st.session_state:
            st.session_state[_prefix + _k] = None

# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("⚙️ Settings")
    st.subheader("📍 Location")
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
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.subheader("🌡️ Display")
    unit_choice = st.radio("Temp unit", ["°F", "°C"], index=0)
    use_celsius = unit_choice == "°C"
    auto_refresh_enabled = st.checkbox("Auto-refresh", value=False)
    refresh_minutes = st.number_input("Refresh interval (min)", min_value=1, max_value=120, value=10)
    if st.button("🔄 Refresh now"):
        st.session_state["last_auto_refresh"] = time.time()
        safe_rerun()
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    if st.button("Clear stored ZIP"):
        if "zip_code" in config:
            del config["zip_code"]
            save_config(config)
        st.success("Cleared stored ZIP."); safe_rerun()
    if st.button("Clear caches"):
        st.cache_data.clear(); st.success("Caches cleared."); safe_rerun()
    if st.button("Reset pump runtimes"):
        empty = {"date": "",
                 "1": {"daily_seconds": 0.0, "on_since": None, "last_state": None},
                 "2": {"daily_seconds": 0.0, "on_since": None, "last_state": None},
                 "3": {"daily_seconds": 0.0, "on_since": None, "last_state": None}}
        config["pump_runtimes"] = empty
        save_config(config)
        for _n in [1, 2, 3]:
            st.session_state[f"pump_daily_seconds_{_n}"] = 0
            st.session_state[f"pump_on_since_{_n}"]      = None
            st.session_state[f"pump_state_{_n}"]         = None
        st.success("Pump runtimes reset."); safe_rerun()
    if st.button("Reset chem ranges"):
        if "chem_ranges" in config:
            del config["chem_ranges"]
            save_config(config)
        for _k in ("temp", "ph", "orp"):
            st.session_state[f"chem_min_{_k}"] = None
            st.session_state[f"chem_max_{_k}"] = None
        st.success("Chemistry ranges reset."); safe_rerun()
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.subheader("☀️ Enphase")
    st.markdown(f'<a href="{ENPHASE_PUBLIC_URL}" target="_blank" rel="noopener noreferrer">Open Enphase (public)</a>', unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.subheader("📲 Push Alerts")
    st.markdown('<span class="signal-badge">via ntfy.sh</span>', unsafe_allow_html=True)
    if NTFY_TOPIC:
        st.caption(f"✅ Topic: `{NTFY_TOPIC}`  |  Server: `{NTFY_SERVER}`")
    else:
        st.warning("NTFY_TOPIC not set — using default.")
    if st.button("📨 Send test push notification"):
        try:
            send_ntfy_message(NTFY_TOPIC, "Test Alert", "✅ Test message from your Streamlit dashboard!", server=NTFY_SERVER, priority="default", tags="white_check_mark")
            st.success(f"Test notification sent to: `{NTFY_TOPIC}`")
        except Exception as exc:
            st.error(f"Failed: {exc}")
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.subheader("🛁 Arctic Spa")
    st.markdown(f'<a href="{ARCTIC_BASE}" target="_blank" rel="noopener noreferrer">Open Arctic Spa portal</a>', unsafe_allow_html=True)
    with st.expander("Local login form (manual only)"):
        st.info("This form is a local convenience only.")
        username = st.text_input("Username", value="", key="local_arctic_username")
        password = st.text_input("Password", value="", type="password", key="local_arctic_password")
        st.checkbox("Remember username for this session", key="remember_local_user")
        if st.button("Show entered username"):
            st.success(f"Username entered: {username or '(empty)'}")
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.subheader("Arctic Spas API (client)")
    if not ARCTICSPAS_INSTALLED:
        st.warning("Package 'arcticspas' not installed.")
    token_preview = bool((st.secrets.get("arcticspa", {}) or {}).get("token"))
    if token_preview:
        st.caption("Using token from Streamlit secrets.")
    else:
        st.warning("No Arctic Spa token found in st.secrets.")
    if st.button("Fetch spa status (via secrets)"):
        result = fetch_spa_status_via_service()
        if result["ok"]: st.success(f"Status OK — HTTP {result['status_code']}"); st.json(result["data"])
        else:             st.error(f"Failed: {result['error']}")

# ---------------- Layout ----------------
zip_to_use = config.get("zip_code", DEFAULT_ZIP)
try:
    lat, lon = geocode_zip_to_latlon_cached(zip_to_use)
except Exception as e:
    st.error(f"Could not geocode ZIP {zip_to_use}: {e}")
    lat = lon = None

if auto_refresh_enabled:
    last = st.session_state.get("last_auto_refresh", 0)
    if time.time() - last >= refresh_minutes * 60:
        st.session_state["last_auto_refresh"] = time.time()
        safe_rerun()

weather_obj   = None
weather_error = None
if lat is not None and lon is not None:
    try:
        weather_obj = get_nws_forecast_cached(lat, lon)
    except Exception as e:
        weather_error = str(e)

NUM_HOURLY_TO_SHOW = 6

def get_hourly_slice(hourly_periods, now_user, user_tz):
    parsed = []
    for h in hourly_periods:
        dt = parse_iso_to_dt(h.get("startTime"))
        if dt: parsed.append((to_user_tz(dt, user_tz), h))
    parsed.sort(key=lambda x: x[0])
    idx = 0
    for i, (dt_user, h) in enumerate(parsed):
        if dt_user > now_user:
            idx = i; break
    return parsed[idx: idx + NUM_HOURLY_TO_SHOW]

def extract_pop(period):
    for key in ("probabilityOfPrecipitation", "pop", "probability"):
        v = period.get(key)
        if v is None: continue
        if isinstance(v, dict):
            val = v.get("value") or v.get("unitCode") or None
            try:    return int(float(val))
            except: continue
        try:    return int(float(v))
        except: continue
    return None

def compute_feels_like_for_period(period, to_celsius_flag):
    t    = period.get("temperature")
    unit = period.get("temperatureUnit", "F")
    if t is None: return None
    try:    temp_f = float(t) if unit.upper() == "F" else float(t)*9.0/5.0+32.0
    except: return None
    wind_str = period.get("windSpeed") or ""
    wind_mph = parse_wind_mph(wind_str) or 0.0
    rh_raw   = period.get("relativeHumidity") or period.get("humidity") or period.get("probabilityOfPrecipitation")
    rh       = rh_raw.get("value") if isinstance(rh_raw, dict) else rh_raw
    try:    rh_val = float(rh) if rh is not None else None
    except: rh_val = None
    if temp_f <= 50 and wind_mph >= 3:
        feels_f = compute_wind_chill(temp_f, wind_mph)
    elif temp_f >= 80 and rh_val is not None:
        feels_f = compute_heat_index(temp_f, rh_val)
    else:
        feels_f = temp_f
    return convert_temp_for_display(feels_f, "F", to_celsius_flag)

st.markdown("""
<style>
.compact-small  { font-size: 0.86rem; line-height:1.3; }
.compact-label  { font-size:0.92rem; font-weight:600; margin-bottom:4px; }
.compact-metric { font-size:1.05rem; font-weight:700; }
.compact-chip   { display:inline-block; padding:4px 8px; border-radius:8px; font-size:0.85rem;
                  margin-right:6px; background:#f1f1f1; }
.small-icon     { width:48px; height:auto; }
</style>
""", unsafe_allow_html=True)

# ---------- Hourly forecast ----------
if weather_obj:
    now_user  = to_user_tz(now_utc(), USER_TZ)
    hourly    = weather_obj.get("forecastHourly", {}).get("properties", {}).get("periods", []) if weather_obj.get("forecastHourly") else []
    dayparts  = weather_obj.get("forecast", {}).get("properties", {}).get("periods", []) if weather_obj.get("forecast") else []
    display_immediate = hourly if hourly else dayparts
    items = get_hourly_slice(display_immediate, now_user, USER_TZ) or []
    if not items:
        st.info("No immediate forecast items available.")
    else:
        cards = []
        for dt_user, it in items:
            label      = "Now" if (dt_user - now_user).total_seconds() < 3600 and (dt_user - now_user).total_seconds() >= -300 else format_time_short(dt_user)
            temp       = it.get("temperature")
            unit       = it.get("temperatureUnit", "F")
            temp_disp  = convert_temp_for_display(temp, unit, use_celsius)
            feels      = compute_feels_like_for_period(it, use_celsius)
            pop        = extract_pop(it)
            icon_url   = it.get("icon") or ""
            cold_threshold = 0 if use_celsius else 32
            hot_threshold  = 29 if use_celsius else 85
            color = "#000"
            if temp_disp is not None:
                if temp_disp < cold_threshold: color = "#1f77b4"
                elif temp_disp >= hot_threshold: color = "#d62728"
            img_tag    = f'<img src="{icon_url}" style="max-width:72%;width:auto;height:auto;display:block;margin:0 auto;object-fit:contain"/>' if icon_url else ""
            feels_html = f'<div style="font-size:12px;margin-top:6px">Feels: {feels}°{"C" if use_celsius else "F"}</div>' if feels is not None and feels != temp_disp else ""
            pop_html   = f'<div style="font-size:12px;margin-top:6px">POP: {pop}%</div>' if pop is not None else ""
            card = f'''
            <div style="display:flex;flex-direction:column;align-items:stretch;justify-content:flex-start;padding:8px 10px;flex:1;min-width:72px;box-sizing:border-box;">
              <div style="text-align:center;font-weight:600;font-size:0.92rem;margin-bottom:6px">{label}</div>
              <div style="display:flex;align-items:center;justify-content:center;">{img_tag}</div>
              <div style="text-align:center;font-weight:700;color:{color};margin-top:8px;font-size:1.02rem">{temp_disp if temp_disp is not None else "N/A"}°{"C" if use_celsius else "F"}</div>
              <div style="text-align:center">{feels_html}{pop_html}</div>
            </div>'''
            cards.append(card)
        cards_html = ''.join(cards)
        st.components.v1.html(f"""
        <style>
        .responsive-grid {{ display: grid; gap: 10px;
          grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
          align-items: start; width: 100%; box-sizing: border-box; padding: 6px 0; }}
        @media (max-width: 600px) {{ .responsive-grid {{ grid-template-columns: repeat(3, 1fr); }} }}
        </style>
        <div style="font-weight:600;font-size:1.05rem;margin-bottom:6px">⏱️ Immediate forecast
          <span style="font-weight:400;font-size:0.88rem;color:#666">— next {NUM_HOURLY_TO_SHOW} hours</span></div>
        <div class="responsive-grid">{cards_html}</div>
        """, height=220, scrolling=True)
else:
    st.info("No forecast available.")

# ---------- 5-day forecast ----------
if weather_obj:
    periods = weather_obj.get("forecast", {}).get("properties", {}).get("periods", []) if weather_obj.get("forecast") else []
    if periods:
        parsed_periods = []
        for p in periods:
            dt = parse_iso_to_dt(p.get("startTime") or p.get("endTime") or "")
            if not dt: continue
            dt_local = to_user_tz(dt, USER_TZ)
            parsed_periods.append((dt_local, p))
        buckets: Dict[date, List[dict]] = {}
        for dt_local, p in parsed_periods:
            day = dt_local.date()
            buckets.setdefault(day, []).append({"period": p, "dt": dt_local})
        today_local = to_user_tz(now_utc(), USER_TZ).date()
        next_days   = [d for d in sorted(buckets.keys()) if d >= today_local][:5]
        cards = []
        for day in next_days:
            bucket  = buckets.get(day, [])
            weekday = day.strftime("%a")
            icon_url = None
            for it in bucket:
                p = it["period"]
                if p.get("isDaytime") and p.get("icon"):
                    icon_url = p.get("icon"); break
            if not icon_url:
                icons = [it["period"].get("icon") for it in bucket if it["period"].get("icon")]
                if icons: icon_url = max(set(icons), key=icons.count)
            temps_display = []
            for item in bucket:
                p  = item["period"]
                td = convert_temp_for_display(p.get("temperature"), p.get("temperatureUnit", "F"), use_celsius)
                if td is not None: temps_display.append(td)
            high = max(temps_display) if temps_display else None
            low  = min(temps_display) if temps_display else None
            img_tag   = f'<img src="{icon_url}" style="max-width:72%;width:auto;height:auto;display:block;margin:0 auto;object-fit:contain"/>' if icon_url else ""
            ft        = "C" if use_celsius else "F"
            low_html  = f'<span style="color:#e05050">{low}°{ft}</span>' if low is not None and ((low < 0 and use_celsius) or (low < 32 and not use_celsius)) else (f"{low}°{ft}" if low is not None else "N/A")
            high_html = f"{high}°{ft}" if high is not None else "N/A"
            short_texts = [p.get("shortForecast","") for p in [it["period"] for it in bucket] if p.get("shortForecast")]
            common    = max(set(short_texts), key=short_texts.count) if short_texts else ""
            card = f'''
            <div style="display:flex;flex-direction:column;align-items:stretch;justify-content:flex-start;padding:8px 10px;flex:1;min-width:72px;box-sizing:border-box;">
              <div style="text-align:center;font-weight:600;font-size:0.92rem;margin-bottom:6px">{weekday}</div>
              <div style="display:flex;align-items:center;justify-content:center;">{img_tag}</div>
              <div style="text-align:center;margin-top:8px;font-size:0.98rem">H <strong>{high_html}</strong></div>
              <div style="text-align:center;font-size:0.86rem">L {low_html}</div>
              <div style="text-align:center;font-size:11px;margin-top:6px;color:#444">{(common or '')[:36]}</div>
            </div>'''
            cards.append(card)
        cards_html = ''.join(cards)
        st.components.v1.html(f"""
        <style>
        .responsive-grid {{ display: grid; gap: 10px;
          grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
          align-items: start; width: 100%; box-sizing: border-box; padding: 6px 0; }}
        </style>
        <div style="font-weight:600;font-size:1.05rem;margin-bottom:6px">📅 5-day forecast
          <span style="font-weight:400;font-size:0.88rem;color:#666">— daily high / low</span></div>
        <div class="responsive-grid">{cards_html}</div>
        """, height=260, scrolling=True)

# ============================================================
# Arctic Spa Status Card
# ============================================================
spa_result = fetch_spa_status_via_service()

st.markdown("<div class='frame' style='background:#f0f4f8;'>", unsafe_allow_html=True)

if not spa_result.get("ok"):
    st.info("🛁 Arctic Spas status not available: " + (spa_result.get("error") or "no token / failed request"))
else:
    try:
        spa_test  # noqa: F821
    except NameError:
        spa_test = None

    spa = spa_test if spa_test is not None else (spa_result.get("data") or {})
    if not isinstance(spa, dict):
        try:    spa = spa.to_dict() if hasattr(spa, "to_dict") else json.loads(json.dumps(spa, default=lambda o: getattr(o, "__dict__", str(o))))
        except:
            try:    spa = dict(spa)
            except: spa = {}

    temp               = spa.get("temperatureF") or spa.get("temperature") or spa.get("temp")
    setpoint           = spa.get("setpointF") or spa.get("setpoint")
    connected          = spa.get("connected")
    lights             = spa.get("lights")
    pump1              = spa.get("pump1")
    pump2              = spa.get("pump2")
    pump3              = spa.get("pump3")
    filter_status      = spa.get("filter_status") or spa.get("filterStatus")
    filtration_frequency = spa.get("filtration_frequency")
    filtration_duration  = spa.get("filtration_duration")
    ph                 = spa.get("ph")
    ph_status          = spa.get("ph_status")
    orp                = spa.get("orp")
    orp_status         = spa.get("orp_status")
    spaboy_connected   = spa.get("spaboy_connected")
    spaboy_producing   = spa.get("spaboy_producing")
    errors             = spa.get("errors") or []

    # ---- Update pump runtimes ----
    update_pump_runtimes({1: pump1, 2: pump2, 3: pump3}, USER_TZ)

    # ---- Alert logic ----
    def _is_error_status(val):
        if val is None: return False
        return str(val).strip().lower() in ("error","critical","fault","failed","failure","alarm","danger",
                                            "high","low","too_high","too_low","toohigh","toolow",
                                            "out_of_range","outofrange","warning")

    temp_val = None
    try:
        if temp is not None: temp_val = float(temp)
    except: pass

    other_not_ok = False
    reasons: List[str] = []

    if connected is False:
        other_not_ok = True; reasons.append("spa disconnected")
    if _is_error_status(filter_status):
        other_not_ok = True; reasons.append(f"filter_status={filter_status}")
    if _is_error_status(ph_status):
        other_not_ok = True; reasons.append(f"ph_status={ph_status}")
    if _is_error_status(orp_status):
        other_not_ok = True; reasons.append(f"orp_status={orp_status}")

    ph_val = None
    try:
        if ph is not None: ph_val = float(ph)
    except: pass
    if ph_val is not None:
        if ph_val < PH_MIN:  other_not_ok = True; reasons.append(f"pH too low: {ph_val:.2f}")
        elif ph_val > PH_MAX: other_not_ok = True; reasons.append(f"pH too high: {ph_val:.2f}")

    orp_val = None
    try:
        if orp is not None: orp_val = float(orp)
    except: pass
    if orp_val is not None:
        if orp_val < ORP_MIN:  other_not_ok = True; reasons.append(f"ORP too low: {orp_val:.0f} mV")
        elif orp_val > ORP_MAX: other_not_ok = True; reasons.append(f"ORP too high: {orp_val:.0f} mV")

    # ---- FIX: Update persistent chem ranges (survives refresh) ----
    update_chem_ranges_persistent(temp_val, ph_val, orp_val, USER_TZ)
    _sync_chem_session(USER_TZ)  # copy into session_state for display

    if errors:
        other_not_ok = True; reasons.append(f"spa errors: {errors}")

    setpoint_val = None
    try:
        if setpoint is not None: setpoint_val = float(setpoint)
    except: pass

    temp_below_setpoint = (temp_val is not None and setpoint_val is not None
                           and temp_val < (setpoint_val - 4.0))
    trigger_alert = other_not_ok or temp_below_setpoint

    temp_reset_result = None
    if temp_below_setpoint:
        spa_id = spa.get("id") or spa.get("spa_id") or spa.get("spaId") or ""
        if spa_id and setpoint_val is not None:
            setpoint_c = (setpoint_val - 32.0) * 5.0 / 9.0
            try:    temp_reset_result = _call_temperature_set(spa_id, setpoint_c)
            except Exception as exc: temp_reset_result = {"ok": False, "error": str(exc)}
        else:
            temp_reset_result = {"ok": False, "error": "spa_id or setpoint not available"}

    alert_sent_recently = False
    if "last_spa_alert" in st.session_state:
        try:
            if time.time() - float(st.session_state["last_spa_alert"]) < 3600:
                alert_sent_recently = True
        except: pass

    if trigger_alert and not alert_sent_recently:
        ts = datetime.now(timezone.utc).astimezone(USER_TZ).strftime("%Y-%m-%d %H:%M %Z")
        alert_title  = "Spa Alert: Monisha's Tub"
        trigger_desc = f"temp {temp_val}F is >4F below setpoint {setpoint_val}F" if temp_below_setpoint else "health indicator"
        reset_line   = ""
        if temp_reset_result is not None:
            reset_line = f"Action: setpoint re-applied ({setpoint_val} F)" if temp_reset_result.get("ok") else f"Action: failed to re-apply — {temp_reset_result.get('error')}"
        p1_rt = get_pump_runtime_display(1)
        p2_rt = get_pump_runtime_display(2)
        p3_rt = get_pump_runtime_display(3)
        message_lines = [
            f"Time: {ts}",
            f"Temp: {temp_val if temp_val is not None else 'N/A'} F  |  Setpoint: {setpoint_val or 'N/A'} F",
            f"Trigger: {trigger_desc}",
            *(([reset_line]) if reset_line else []),
            f"Issues: {', '.join(reasons) if reasons else 'none'}",
            f"Conn: {'OK' if connected else 'DISCONNECTED'}  P1:{pump1} ({p1_rt}) P2:{pump2} ({p2_rt}) P3:{pump3} ({p3_rt})",
            f"Filter: {filter_status}  |  pH: {ph} ({ph_status})  |  ORP: {orp} ({orp_status})",
            f"Errors: {errors or 'none'}",
        ]
        try:
            send_ntfy_message(NTFY_TOPIC, alert_title, "\n".join(message_lines),
                              server=NTFY_SERVER, priority="urgent", tags="warning,bathtub")
            st.success(f"📲 Push alert sent to ntfy topic: `{NTFY_TOPIC}`")
            st.session_state["last_spa_alert"] = time.time()
        except Exception as exc:
            st.error(f"Failed to send ntfy alert: {exc}")
        if temp_reset_result is not None:
            if temp_reset_result.get("ok"): st.success(f"Setpoint re-applied: {setpoint_val} F")
            else:                            st.warning(f"Could not re-apply setpoint: {temp_reset_result.get('error')}")
    elif trigger_alert and alert_sent_recently:
        st.info("⚠️ Alert condition present but rate-limited (1/hr).")

    # ---- Build spa card HTML ----
    if trigger_alert:
        problems_str = " · ".join(reasons) if reasons else "temp below setpoint"
        alert_banner = f'<div class="alert-banner">⚠️ Alert condition detected — {problems_str}</div>'
    else:
        alert_banner = ""

    def badge_class(val, ok_vals=("on","ok","true","1","connected"), warn_vals=("low","high","slow")):
        if val is None: return "spa-badge-off"
        s = str(val).strip().lower()
        if s in ok_vals:   return "spa-badge-ok"
        if s in warn_vals: return "spa-badge-warn"
        if s in ("off","false","0","disabled","disconnected","error"): return "spa-badge-err"
        return "spa-badge-off"

    # ---- FIX: Connection status dot ----
    conn_dot_cls = "conn-dot-ok" if connected else "conn-dot-err"
    conn_label   = "Connected" if connected else "Disconnected"
    conn_badge_cls = "spa-badge-ok" if connected else "spa-badge-err"
    conn_html = (
        f'<span class="spa-badge {conn_badge_cls}" style="display:inline-flex;align-items:center;gap:6px;">'
        f'<span class="conn-dot {conn_dot_cls}"></span>{conn_label}</span>'
    )

    temp_color = "#ff6060" if (temp_val is not None and temp_val < 4) else "#7ef7c8"

    spaboy_html = ""
    if spaboy_connected is not None:
        sb_cls   = "spa-badge-ok" if spaboy_connected else "spa-badge-off"
        sp_label = "SpaBoy " + ("connected" if spaboy_connected else "disconnected")
        if spaboy_producing: sp_label += " · producing"
        spaboy_html = f'<span class="spa-badge {sb_cls}">{sp_label}</span>'

    lights_cls = badge_class(lights)
    filter_cls = badge_class(filter_status)

    errors_html = ""
    if errors:
        err_items   = "".join(f'<div style="color:#ff8080;font-size:0.85rem;margin-top:4px">⚠ {e}</div>' for e in errors)
        errors_html = f'<hr class="spa-divider"><div class="spa-section-label">Errors / Alerts</div>{err_items}'
    else:
        errors_html = '<hr class="spa-divider"><div style="color:#6effa8;font-size:0.85rem">✅ No active errors</div>'

    filtration_html = ""
    if filtration_frequency or filtration_duration:
        filtration_html = f'<span class="spa-badge spa-badge-off">Freq: {filtration_frequency or "?"}/day &nbsp;|&nbsp; Dur: {filtration_duration or "?"} min</span>'

    def pump_inline(n, pval):
        rt    = get_pump_runtime_display(n)
        is_on = _is_pump_on(pval)
        dot   = f'<span style="color:{"#6effa8" if is_on else "#666"};font-size:0.7rem">{"●" if is_on else "○"}</span>'
        st_lbl = str(pval or "off")
        return (f'<span style="color:#a0c8e8;font-size:0.82rem">P{n}</span> {dot} '
                f'<span style="font-size:0.82rem;color:#ccc">{st_lbl}</span> '
                f'<span style="font-size:0.82rem;color:#7ecbf7;font-weight:600">{rt}</span>')

    p1_html = pump_inline(1, pump1)
    p2_html = pump_inline(2, pump2)
    p3_html = pump_inline(3, pump3)

    temp_html = chem_inline_html("Temp", temp_val, "°F", TEMP_MIN_F, TEMP_MAX_F,
                                 st.session_state.get("chem_min_temp"),
                                 st.session_state.get("chem_max_temp"), fmt=".1f")
    ph_html   = chem_inline_html("pH", ph_val, "", PH_MIN, PH_MAX,
                                 st.session_state.get("chem_min_ph"),
                                 st.session_state.get("chem_max_ph"), fmt=".2f")
    orp_html  = chem_inline_html("ORP", orp_val, " mV", ORP_MIN, ORP_MAX,
                                 st.session_state.get("chem_min_orp"),
                                 st.session_state.get("chem_max_orp"), fmt=".0f")

    inline_styles = """
    <style>
    body { margin:0; font-family: sans-serif; }
    .spa-card { background: linear-gradient(135deg, #1a3a5c 0%, #0d2137 100%);
      border-radius: 14px; padding: 18px 20px; color: #fff; margin-bottom: 10px;
      box-shadow: 0 4px 18px rgba(0,0,0,0.18); }
    .spa-title { font-size: 1.05rem; font-weight: 700; color: #7ecbf7;
      letter-spacing: 0.03em; margin-bottom: 12px; }
    .spa-temp-big { font-size: 2.6rem; font-weight: 800; line-height: 1; }
    .spa-temp-sub { font-size: 0.88rem; color: #a0c8e8; margin-top: 4px; }
    .spa-badge { display:inline-flex; align-items:center; gap:6px; padding:4px 10px; border-radius:20px;
      font-size:0.8rem; font-weight:600; margin:3px 3px 3px 0; }
    .spa-badge-ok   { background:rgba(40,200,100,0.22);  color:#6effa8; border:1px solid rgba(40,200,100,0.4); }
    .spa-badge-warn { background:rgba(255,160,0,0.22);   color:#ffd060; border:1px solid rgba(255,160,0,0.4); }
    .spa-badge-err  { background:rgba(220,60,60,0.22);   color:#ff8080; border:1px solid rgba(220,60,60,0.4); }
    .spa-badge-off  { background:rgba(120,120,120,0.22); color:#ccc;    border:1px solid rgba(120,120,120,0.3); }
    .spa-section-label { font-size:0.78rem; text-transform:uppercase; letter-spacing:0.07em;
      color:#7ecbf7; margin-bottom:5px; margin-top:12px; font-weight:600; }
    .spa-divider { border:none; border-top:1px solid rgba(255,255,255,0.08); margin:10px 0; }
    .alert-banner { background:linear-gradient(90deg,#7b1a1a,#b02020); border-radius:8px;
      padding:10px 14px; color:#fff; font-weight:600; margin-bottom:10px; }
    /* Connection dot */
    .conn-dot { display:inline-block; width:10px; height:10px; border-radius:50%; flex-shrink:0; }
    .conn-dot-ok  { background:#3ddc84; box-shadow:0 0 6px #3ddc84; }
    .conn-dot-err { background:#ff4444; box-shadow:0 0 6px #ff4444; animation: pulse-red 1.5s infinite; }
    @keyframes pulse-red {
      0%,100% { box-shadow: 0 0 4px #ff4444; }
      50%      { box-shadow: 0 0 12px #ff4444; }
    }
    </style>
    """

    spa_card_html = f"""
    <div class="spa-card">
      {alert_banner}
      <div class="spa-title">🛁 Monisha's Tub — Live Status</div>

      <div style="display:flex;align-items:flex-end;gap:24px;flex-wrap:wrap;">
        <div>
          <div class="spa-temp-big" style="color:{temp_color}">{temp if temp is not None else "—"} °F</div>
          <div class="spa-temp-sub">Setpoint: {setpoint or "—"} °F</div>
        </div>
        <div style="padding-bottom:4px">
          {conn_html}
          {spaboy_html}
        </div>
      </div>

      <hr class="spa-divider">

      <div style="display:flex;gap:18px;flex-wrap:wrap;align-items:center;line-height:1.8">
        {temp_html} &nbsp;·&nbsp; {ph_html} &nbsp;·&nbsp; {orp_html}
      </div>

      <hr class="spa-divider">

      <div style="display:flex;gap:16px;flex-wrap:wrap;align-items:center;line-height:1.8">
        {p1_html} &nbsp;·&nbsp; {p2_html} &nbsp;·&nbsp; {p3_html}
      </div>
      <div style="font-size:0.7rem;color:#3a5a70;margin-top:2px">runtime today · resets midnight</div>

      <hr class="spa-divider">

      <span class="spa-badge {lights_cls}">Lights: {lights or "?"}</span>
      <span class="spa-badge {filter_cls}">Filter: {filter_status or "?"}</span>
      {filtration_html}

      {errors_html}
    </div>
    """

    st.components.v1.html(inline_styles + spa_card_html, height=360, scrolling=False)

    with st.expander("Raw spa payload (debug)"):
        st.json(spa)

st.markdown("</div>", unsafe_allow_html=True)

# ---------- Enphase iframe ----------
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
st.markdown("## ☀️ Enphase Solar Panels")
try:
    st.components.v1.iframe(ENPHASE_PUBLIC_URL, height=540)
except Exception:
    st.markdown(f'<a href="{ENPHASE_PUBLIC_URL}" target="_blank" rel="noopener noreferrer">Open Enphase hour graph (public)</a>', unsafe_allow_html=True)
st.markdown(f'If embedding is blocked: <a href="{ENPHASE_PUBLIC_URL}" target="_blank" rel="noopener noreferrer">Open in new tab</a>', unsafe_allow_html=True)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
st.caption("Compact layout · Hourly starts at next full hour · First shown hour labeled 'Now' · Feels-like, POP & color-coded temps · Pump runtime tracked in-session · Chemistry ranges persisted to disk · Alerts via ntfy.sh")
if st.checkbox("Show debug"):
    st.write("config:", config)
    st.write("zip:", zip_to_use)
    st.write("chem_ranges:", config.get("chem_ranges", {}))
    st.write("pump_runtimes (disk):", config.get("pump_runtimes", {}))
    st.write("last_auto_refresh:", st.session_state.get("last_auto_refresh"))
    st.write("arcticspas installed:", ARCTICSPAS_INSTALLED)
    st.write("ntfy topic:", NTFY_TOPIC)
    for _n in [1, 2, 3]:
        daily_s  = float(st.session_state.get(f"pump_daily_seconds_{_n}", 0))
        on_since = st.session_state.get(f"pump_on_since_{_n}")
        live     = (time.time() - on_since) if on_since else 0
        total    = daily_s + live
        st.write(f"pump_{_n}: disk_daily={daily_s:.0f}s  live={live:.0f}s  total={total:.0f}s  "
                 f"display={get_pump_runtime_display(_n)}  state={st.session_state.get(f'pump_state_{_n}')}")