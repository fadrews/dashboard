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
import logging
import threading
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
RUNTIME_PATH   = Path(".dashboard_runtime.json")
EVENT_LOG_PATH = Path(".dashboard_events.jsonl")
ICON_CACHE_DIR = Path(".cache_icons")
ICON_CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_ZIP         = "84124"
ENPHASE_PUBLIC_URL  = "https://enlighten.enphaseenergy.com/mobile/HRDg1683634/history/graph/hours?public=1"
ARCTIC_BASE         = "https://api.myarcticspa.com"
ARCTIC_SPA_PORTAL_HINT = "https://myarcticspa.com/spa/SpaAPIManagement.aspx"

# Chemistry safe ranges
PH_MIN,   PH_MAX   = 7.1, 7.9 #.2 .8
ORP_MIN,  ORP_MAX  = 520, 700 #600 800
TEMP_MIN_F, TEMP_MAX_F = 95.0, 104.0
SPA_POLL_SECONDS = 3600              # One API check per hour, per running server
SPA_RETRY_SECONDS = 600             # Back off 10 minutes after a failed check
ALERT_COOLDOWN_SECONDS = 14400      # Four hours, shared by browser sessions
MAX_PUMP_INTERVAL_SECONDS = 5400    # Longer gaps cannot be estimated responsibly

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
        # Avoid leaving truncated JSON if a page reloads during a write.
        with _ops_lock():
            tmp = CONFIG_PATH.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as file:
                json.dump(cfg, file, indent=2)
            os.replace(tmp, CONFIG_PATH)
    except Exception as e:
        st.error(f"Failed to save config: {e}")

def clear_config():
    try:
        if CONFIG_PATH.exists(): CONFIG_PATH.unlink()
    except Exception as e:
        st.error(f"Failed to clear config: {e}")

config = load_config()


# ---------- Server-shared operational state and audit events ----------
# These files persist across browser sessions on ONE Streamlit server instance.
# Streamlit Community Cloud may delete them on restart/redeployment. No API
# tokens, login credentials or raw spa payloads are written to event logs.
LOGGER = logging.getLogger("home_dashboard")
if not LOGGER.handlers:
    LOGGER.addHandler(logging.StreamHandler())
LOGGER.setLevel(logging.INFO)

@st.cache_resource
def _ops_lock():
    return threading.RLock()

def _read_ops():
    try:
        with RUNTIME_PATH.open("r", encoding="utf-8") as file:
            obj = json.load(file)
            return obj if isinstance(obj, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}

def _write_ops(state):
    # Atomic replacement prevents partial JSON files on routine interruptions.
    temporary = RUNTIME_PATH.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(state, file, ensure_ascii=False, default=str)
    os.replace(temporary, RUNTIME_PATH)

def _event(event_type, **details):
    record = {"at": datetime.now(timezone.utc).isoformat(), "type": event_type, **details}
    line = json.dumps(record, ensure_ascii=False, default=str)
    LOGGER.info("dashboard_event %s", line)
    try:
        with EVENT_LOG_PATH.open("a", encoding="utf-8") as file:
            file.write(line + "\n")
    except OSError as exc:
        LOGGER.warning("Could not append dashboard event: %s", exc)

def _last_events(count=30):
    try:
        with EVENT_LOG_PATH.open("r", encoding="utf-8") as file:
            return file.readlines()[-count:]
    except OSError:
        return []

def _spa_result_from_ops(ops):
    snapshot = ops.get("spa_snapshot")
    if isinstance(snapshot, dict):
        return {"ok": True, "data": snapshot,
                "status_code": ops.get("spa_status_code"),
                "checked_at": ops.get("spa_checked_at"),
                "stale": not ops.get("spa_attempt_ok", False),
                "error": ops.get("spa_last_error")}
    return {"ok": False, "data": None, "checked_at": None,
            "stale": True, "error": ops.get("spa_last_error") or "No successful spa reading yet"}

def get_spa_status_throttled():
    # Unlike Session State this gate is shared by all open browser sessions.
    # There is no background polling: a new check requires a script execution.
    with _ops_lock():
        state = _read_ops()
        now = time.time()
        prior = float(state.get("spa_attempt_ts") or 0)
        interval = SPA_POLL_SECONDS if state.get("spa_attempt_ok") else SPA_RETRY_SECONDS
        if prior and now - prior < interval:
            return _spa_result_from_ops(state)
        result = fetch_spa_status_via_service()
        state["spa_attempt_ts"] = now
        state["spa_attempt_ok"] = bool(result.get("ok"))
        state["spa_last_error"] = result.get("error")
        if result.get("ok"):
            data = result.get("data")
            if not isinstance(data, dict):
                try:
                    data = data.to_dict() if hasattr(data, "to_dict") else dict(data)
                except Exception:
                    data = {}
            state["spa_snapshot"] = data
            state["spa_status_code"] = result.get("status_code")
            state["spa_checked_at"] = now
        _write_ops(state)
        _event("SPA_POLL", success=bool(result.get("ok")),
               status_code=result.get("status_code"), error=result.get("error"))
        return _spa_result_from_ops(state)

def _local_timestamp(unix_ts):
    if not unix_ts:
        return "Never"
    return datetime.fromtimestamp(float(unix_ts), USER_TZ).strftime("%b %d, %-I:%M %p")

def process_new_spa_observation(spa, checked_at, *, trigger_alert, reasons,
                                temp_below_setpoint, temp_val, setpoint_val,
                                ph_val, orp_val, connected, filter_status,
                                ph_status, orp_status, errors):
    """At most once for each real API sample, across refrigerator sessions."""
    if not checked_at:
        return {}
    with _ops_lock():
        ops = _read_ops()
        if ops.get("spa_processed_at") == checked_at:
            return {}
        ops["spa_processed_at"] = checked_at  # claim first: avoid duplicate commands
        _write_ops(ops)

        # Reload config before writes so stale browser sessions do not overwrite
        # the latest stored chemistry/runtime records.
        config.clear()
        config.update(load_config())
        update_pump_runtimes({1: spa.get("pump1"), 2: spa.get("pump2"),
                              3: spa.get("pump3")}, USER_TZ, sampled_at=checked_at)
        update_chem_ranges_persistent(temp_val, ph_val, orp_val, USER_TZ)
        _event("SPA_STATUS", sampled_at=checked_at, temp_f=temp_val,
               target_f=setpoint_val, connected=connected, ph=ph_val,
               orp_mv=orp_val, pumps=[spa.get("pump1"), spa.get("pump2"), spa.get("pump3")],
               alert=trigger_alert)

        outcome = {}
        if not temp_below_setpoint:
            ops["setpoint_incident_active"] = False
        elif not ops.get("setpoint_incident_active"):
            # One command per continuous low-temperature incident, even if
            # the app is reloaded/reconnected. Never re-send due to reruns.
            ops["setpoint_incident_active"] = True
            ops["setpoint_attempt_ts"] = time.time()
            _write_ops(ops)
            spa_id = spa.get("id") or spa.get("spa_id") or spa.get("spaId")
            if spa_id and setpoint_val is not None and connected is not False:
                result = _call_temperature_set(spa_id, (setpoint_val - 32.0) * 5.0 / 9.0)
            else:
                result = {"ok": False, "error": "spa id/setpoint missing or disconnected"}
            outcome["setpoint"] = result
            _event("SPA_SETPOINT", success=bool(result.get("ok")),
                   error=result.get("error"), incident="low_temperature")

        if trigger_alert:
            last_sent_or_attempted = float(ops.get("alert_attempt_ts") or 0)
            if time.time() - last_sent_or_attempted >= ALERT_COOLDOWN_SECONDS:
                # Reserve cooldown before network request (ambiguous failures
                # must not trigger a burst of duplicate notifications).
                ops["alert_attempt_ts"] = time.time()
                _write_ops(ops)
                message = [
                    "Time: " + _local_timestamp(checked_at),
                    f"Temperature: {temp_val if temp_val is not None else 'N/A'} F  |  Setpoint: {setpoint_val if setpoint_val is not None else 'N/A'} F",
                    "Issues: " + (", ".join(reasons) if reasons else "temperature below setpoint"),
                    f"Connected: {connected}; pumps: {spa.get('pump1')}, {spa.get('pump2')}, {spa.get('pump3')}",
                    f"Filter: {filter_status}; pH: {ph_val} ({ph_status}); ORP: {orp_val} ({orp_status})",
                    f"Spa errors: {errors or 'none'}",
                ]
                if "setpoint" in outcome:
                    message.append("Setpoint reapplication: " +
                                   ("success" if outcome["setpoint"].get("ok") else "failed"))
                try:
                    send_ntfy_message(NTFY_TOPIC, "Spa Alert: Monisha's Tub", "\n".join(message),
                                      server=NTFY_SERVER, priority="urgent", tags="warning,bathtub")
                    outcome["notification"] = "sent"
                    _event("SPA_ALERT", success=True, reasons=reasons)
                except Exception as exc:
                    outcome["notification"] = "failed"
                    _event("SPA_ALERT", success=False, error=str(exc), reasons=reasons)
            else:
                _event("SPA_ALERT_SUPPRESSED", reason="four_hour_cooldown")
        ops["alert_condition_active"] = bool(trigger_alert)
        _write_ops(ops)
        return outcome


# ============================================================
# PERSISTENT CHEMISTRY RANGE â€” survives browser refresh
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


def ensure_chem_ranges_from_cached_sample(spa_result, user_tz):
    """Repair absent daily min/max from a successful *today* spa observation.

    The last successful sample and the daily ranges live in separate files.
    Losing/resetting only the range file must not display missing min/max
    while a same-day measurement is available. This is display-state repair:
    it never calls the spa API, sends alerts, or modifies pump run times.
    """
    if not spa_result.get("ok") or not isinstance(spa_result.get("data"), dict):
        return
    checked_at = spa_result.get("checked_at")
    if checked_at is None:
        return
    try:
        sampled_day = datetime.fromtimestamp(float(checked_at), user_tz).date()
        if sampled_day != datetime.now(user_tz).date():
            return  # Yesterday's cached sample is not today's first reading.
    except (ValueError, TypeError, OverflowError, OSError):
        return

    sample = spa_result["data"]
    raw_temp = next((sample[key] for key in ("temperatureF", "temperature", "temp")
                     if sample.get(key) is not None), None)
    observations = {"temp": raw_temp, "ph": sample.get("ph"), "orp": sample.get("orp")}
    with _ops_lock():
        # Always start with the latest persisted record, including changes
        # made by another browser session rather than its stale in-memory copy.
        config.clear()
        config.update(load_config())
        ranges = _load_chem_ranges(user_tz)
        initialized = []
        for name, raw in observations.items():
            try:
                value = float(raw)
                if not math.isfinite(value):
                    continue
            except (TypeError, ValueError):
                continue
            lo_key, hi_key = f"{name}_min", f"{name}_max"
            lo, hi = ranges.get(lo_key), ranges.get(hi_key)
            if lo is not None and hi is not None:
                continue  # Never replace an existing daily range.
            if lo is None:
                ranges[lo_key] = min(float(hi), value) if hi is not None else value
            if hi is None:
                ranges[hi_key] = max(float(lo), value) if lo is not None else value
            initialized.append(name)
        if initialized:
            _save_chem_ranges(ranges)
            _event("CHEM_RANGE_INITIALIZED", sampled_at=checked_at,
                   fields=initialized, source="cached_current_day_spa_observation")


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

@st.cache_data(ttl=900)
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
    fetched_at = time.time()
    _event("WEATHER_UPDATE", success=True, fetched_at=fetched_at)
    return {"point": point, "forecast": forecast, "forecastHourly": forecast_hourly, "fetched_at": fetched_at}



# ============================================================
# PUMP RUNTIME ESTIMATE: two hourly state observations, not exact telemetry.
# Assumes unchanged ON means running throughout the interval, a state change
# means running half the interval, and unchanged OFF means none. Long gaps
# (>90 minutes) are excluded instead of inventing unknown hours of runtime.
# Update ONLY when a new successful spa API sample is obtained.
# ============================================================
def _is_pump_on(val):
    if val is None:
        return None  # Unknown is not the same as OFF
    s = str(val).strip().lower()
    if s in ("on", "1", "true", "running", "high", "low", "medium"):
        return True
    if s in ("off", "0", "false", "stopped", "disabled", ""):
        return False
    try:
        return float(s) > 0
    except (TypeError, ValueError):
        return None

def _today_str(user_tz):
    return datetime.now(user_tz).strftime("%Y-%m-%d")

def _load_pump_runtimes(user_tz):
    today = _today_str(user_tz)
    stored = config.get("pump_runtimes") or {}
    if not isinstance(stored, dict):
        stored = {}
    if stored.get("date") != today:
        # Keep prior observed state/time across midnight, but reset totals.
        stored = {"date": today, "last_sample_ts": stored.get("last_sample_ts")}
        for n in (1, 2, 3):
            old = (config.get("pump_runtimes") or {}).get(str(n), {})
            stored[str(n)] = {"daily_seconds": 0.0,
                              "last_state": old.get("last_state")}
    for n in (1, 2, 3):
        stored.setdefault(str(n), {"daily_seconds": 0.0, "last_state": None})
    return stored

def update_pump_runtimes(pump_vals, user_tz, sampled_at=None):
    sample_ts = float(sampled_at if sampled_at is not None else time.time())
    runtimes = _load_pump_runtimes(user_tz)
    prior_ts = runtimes.get("last_sample_ts")
    today_start = datetime.combine(datetime.fromtimestamp(sample_ts, user_tz).date(),
                                   datetime.min.time(), tzinfo=user_tz).timestamp()
    elapsed = sample_ts - float(prior_ts) if prior_ts is not None else None
    eligible = elapsed is not None and 0 < elapsed <= MAX_PUMP_INTERVAL_SECONDS
    span = max(0.0, sample_ts - max(float(prior_ts), today_start)) if eligible else 0.0
    for n in (1, 2, 3):
        key = str(n)
        entry = runtimes[key]
        old_state = _is_pump_on(entry.get("last_state"))
        new_state = _is_pump_on(pump_vals.get(n))
        increment = 0.0
        if eligible and old_state is not None and new_state is not None:
            if old_state and new_state:
                increment = span
            elif old_state != new_state:
                increment = span / 2.0
        entry["daily_seconds"] = float(entry.get("daily_seconds") or 0) + increment
        entry["last_state"] = pump_vals.get(n)
        st.session_state[f"pump_daily_seconds_{n}"] = entry["daily_seconds"]
        st.session_state[f"pump_state_{n}"] = entry["last_state"]
    runtimes["last_sample_ts"] = sample_ts
    config["pump_runtimes"] = runtimes
    save_config(config)
    _event("PUMP_OBSERVATION", sample_ts=sample_ts,
           interval_seconds=round(elapsed) if elapsed is not None else None,
           interval_estimated=eligible,
           states={str(n): pump_vals.get(n) for n in (1, 2, 3)},
           minutes={str(n): round(runtimes[str(n)]["daily_seconds"] / 60)
                    for n in (1, 2, 3)})

def _sync_pump_session(user_tz):
    runtimes = _load_pump_runtimes(user_tz)
    for n in (1, 2, 3):
        entry = runtimes[str(n)]
        st.session_state[f"pump_daily_seconds_{n}"] = float(entry.get("daily_seconds") or 0)
        st.session_state[f"pump_state_{n}"] = entry.get("last_state")

def get_pump_runtime_display(n):
    # Show time accumulated up to last observation. No guessed live minutes.
    seconds = float(st.session_state.get(f"pump_daily_seconds_{n}") or 0)
    hours, minutes = divmod(int(seconds // 60), 60)
    return f"~{hours}h {minutes:02d}m" if hours else f"~{minutes}m"

def chem_inline_html(label, value, unit, safe_lo, safe_hi, day_min, day_max, fmt=".1f"):
    if value is None:
        return f'<span style="color:#64748b;font-size:0.82rem">{label}: N/A</span>'
    in_range  = safe_lo <= value <= safe_hi
    val_color = "#14753b" if in_range else "#b42318"
    val_str   = f"{value:{fmt}}"
    if day_min is not None and day_max is not None:
        range_str = f"{day_min:{fmt}}â€“{day_max:{fmt}}"
    else:
        range_str = "â€”"
    return (f'<span style="font-size:0.82rem;color:#475569">{label}:</span> '
            f'<span style="font-weight:700;color:{val_color}">{val_str}{unit}</span> '
            f'<span style="font-size:0.74rem;color:#64748b">[{range_str}]</span>')


# ============================================================
# App start
# ============================================================
st.set_page_config(layout="wide", page_title="Home Dashboard", initial_sidebar_state="collapsed")

st.markdown("""<style>
/* Compact, iframe-free sections for Streamlit 1.65 and the Samsung browser. */
[data-testid="stAppViewContainer"] {background:#f7f9fc;}
[data-testid="stMainBlockContainer"], section.main .block-container {max-width:100%; padding:2.35rem 1rem .65rem !important;}
[data-testid="stVerticalBlock"] {gap:.35rem !important;}
[data-testid="stHeader"] {height:2rem !important;background:#f7f9fc;}
.section-heading {font:700 1.1rem/1.25 Arial,sans-serif;padding:8px 12px;
  margin:6px 0 4px;border:1px solid #c8dfed;border-radius:8px;
  color:#173f5c;background:#dceefa;}
.section-heading.spa {background:#ffead5;border-color:#f1d6b8;color:#633b23;}
.section-heading.solar {background:#e2f3df;border-color:#cce3c8;color:#24552b;}
.weather-panel,.spa-panel,.solar-panel {box-sizing:border-box;background:#fff;border:1px solid #d7e5ef;
  border-radius:8px;padding:10px 12px 0;clear:both;overflow:visible;}
.spa-panel {border-color:#efd8bf;}
.solar-panel {border-color:#cce3c8;}
.panel-footer {box-sizing:border-box;position:static;clear:both;display:flex;align-items:center;
  justify-content:flex-end;gap:6px;flex-wrap:wrap;border-top:1px solid #e4e9ee;
  margin-top:9px;padding:7px 1px 8px;min-height:28px;
  color:#536678;font:11px/1.35 Arial,sans-serif;overflow-wrap:anywhere;}
.panel-footer .footer-label {font-weight:700;text-transform:uppercase;letter-spacing:.03em;color:#52657a;}
.panel-footer .footer-alert {color:#a62319;}
.weather-panel .forecast-title {font:700 13px Arial,sans-serif;color:#395b72;margin:0 0 5px;}
.weather-two {display:grid;grid-template-columns:minmax(0,1.2fr) minmax(0,1fr);gap:12px;}
.weather-two>div {min-width:0;}
.weather-two>div+div {border-left:1px solid #dbe4ed;padding-left:12px;}
.weather-grid {display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:4px;}
.weather-grid.daily {grid-template-columns:repeat(5,minmax(0,1fr));}
.weather-card {text-align:center;min-width:0;font:12px/1.18 Arial,sans-serif;color:#273747;}
.weather-card .when {font-size:13px;font-weight:700;margin-bottom:3px;}
.weather-card img {width:46px;height:46px;object-fit:contain;display:block;margin:0 auto;}
.weather-card .temp {font-size:16px;font-weight:700;margin-top:2px;}
.weather-card .sub {font-size:11px;color:#506275;margin-top:2px;overflow-wrap:anywhere;}
/* Spa: fully native HTML, no fixed-height embedded iframe and no iframe scrollbar. */
.spa-three {display:grid;grid-template-columns:minmax(0,1.05fr) minmax(0,1fr) minmax(0,1.45fr);gap:0;}
.spa-col {padding:3px 12px;min-width:0;overflow-wrap:anywhere;}
.spa-col:first-child {padding-left:2px;}
.spa-col+.spa-col {border-left:1px solid #dce2e8;}
.spa-subhead {font:700 12px/1.3 Arial,sans-serif;color:#64748b;margin:0 0 8px;}
.spa-temperature {font:800 32px/1.1 Arial,sans-serif;color:#273d4d;}
.spa-temperature.bad {color:#b42318;}
.spa-setpoint {font:14px/1.3 Arial,sans-serif;margin:5px 0 9px;}
.spa-status {display:inline-block;padding:2px 7px;border-radius:12px;font:700 11px/1.3 Arial,sans-serif;white-space:nowrap;}
.spa-status.ok {background:#def4e4;color:#146c38;}
.spa-status.bad {background:#ffe0dc;color:#a62319;}
.spa-status.neutral {background:#edf0f3;color:#53616d;}
.spa-detail {font:12px/1.4 Arial,sans-serif;margin:5px 0;color:#263749;}
.spa-pump-row {display:grid;grid-template-columns:30px minmax(0,1fr) auto;gap:5px;align-items:center;
  font:13px/1.35 Arial,sans-serif;padding:3px 0;border-bottom:1px solid #edf0f3;}
.spa-pump-row small {font-size:11px;color:#637383;}
.spa-chem-row {display:grid;grid-template-columns:40px minmax(68px,1fr) 48px minmax(78px,1fr);
  gap:5px;align-items:center;padding:5px 0;border-bottom:1px solid #edf0f3;font:13px/1.35 Arial,sans-serif;}
.spa-chem-row small {font-size:10px;color:#637383;white-space:normal;}
.spa-issues {background:#fff1ed;color:#a62319;padding:5px 7px;margin-top:7px;
  border-radius:5px;font:12px/1.4 Arial,sans-serif;}
.spa-no-issues {color:#217447;font:11px/1.4 Arial,sans-serif;margin-top:6px;}
.solar-panel .solar-intro {font:14px/1.5 Arial,sans-serif;color:#394b59;padding:0 0 7px;}
.solar-panel a {color:#155b30;font-weight:700;}
@media(max-width:850px) {
  .weather-two {grid-template-columns:1fr;gap:9px;}
  .weather-two>div+div {border-left:0;border-top:1px solid #dbe4ed;padding:9px 0 0;}
  .spa-three {grid-template-columns:1fr 1fr;}
  .spa-col:last-child {grid-column:1/-1;border-left:0;border-top:1px solid #dce2e8;
    padding-top:8px;margin-top:7px;}
}
@media(max-width:520px) {
  .weather-card img {width:32px;height:32px;}
  .weather-card .temp {font-size:13px;}
  .weather-card .sub {font-size:9px;}
  .spa-pump-row {grid-template-columns:24px minmax(0,1fr);}
  .spa-pump-row small {grid-column:2;}
  .spa-chem-row {grid-template-columns:34px minmax(58px,1fr) 43px minmax(58px,1fr);}
}
</style>""", unsafe_allow_html=True)



USER_TZ = ZoneInfo("America/Denver") if ZoneInfo else timezone.utc

# Initialise runtime session keys
for _n in [1, 2, 3]:
    for _k in [f"pump_daily_seconds_{_n}", f"pump_state_{_n}"]:
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
    st.title("Settings")
    st.subheader("Location")
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
    st.subheader("Display")
    unit_choice = st.radio("Temp unit", ["Fahrenheit (F)", "Celsius (C)"], index=0)
    use_celsius = unit_choice == "Celsius (C)"
    st.caption("Weather updates while this browser session is active; spa checks no more than hourly.")
    if st.button("Refresh display now"):
        safe_rerun()
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    if st.button("Clear stored ZIP"):
        if "zip_code" in config:
            del config["zip_code"]
            save_config(config)
        st.success("Cleared stored ZIP."); safe_rerun()
    if st.button("Clear weather/geocode caches"):
        st.cache_data.clear(); st.success("Weather/geocode caches cleared."); safe_rerun()
    if st.button("Reset pump runtimes"):
        empty = {"date": "", "last_sample_ts": None,
                 **{str(n): {"daily_seconds": 0.0, "last_state": None} for n in (1, 2, 3)}}
        config["pump_runtimes"] = empty
        save_config(config)
        for _n in [1, 2, 3]:
            st.session_state[f"pump_daily_seconds_{_n}"] = 0
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
    st.subheader("Enphase")
    st.markdown(f'<a href="{ENPHASE_PUBLIC_URL}" target="_blank" rel="noopener noreferrer">Open Enphase (public)</a>', unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.subheader("Push Alerts")
    st.markdown('<span class="signal-badge">via ntfy.sh</span>', unsafe_allow_html=True)
    if NTFY_TOPIC:
        st.caption(f"Topic: `{NTFY_TOPIC}`  |  Server: `{NTFY_SERVER}`")
    else:
        st.warning("NTFY_TOPIC not set - using default.")
    if st.button("Send test push notification"):
        try:
            send_ntfy_message(NTFY_TOPIC, "Test Alert", "âœ… Test message from your Streamlit dashboard!", server=NTFY_SERVER, priority="default", tags="white_check_mark")
            st.success(f"Test notification sent to: `{NTFY_TOPIC}`")
        except Exception as exc:
            st.error(f"Failed: {exc}")
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.subheader("Arctic Spa")
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
    if st.button("Show latest stored spa status"):
        result = _spa_result_from_ops(_read_ops())
        if result["ok"]: st.success(f"Status OK - HTTP {result['status_code']}"); st.json(result["data"])
        else:             st.error(f"Failed: {result['error']}")

# ---------------- Layout ----------------
zip_to_use = config.get("zip_code", DEFAULT_ZIP)
try:
    lat, lon = geocode_zip_to_latlon_cached(zip_to_use)
except Exception as e:
    st.error(f"Could not geocode ZIP {zip_to_use}: {e}")
    lat = lon = None

NUM_HOURLY_TO_SHOW = 6


def get_hourly_slice(hourly_periods, now_user, user_tz):
    parsed = []
    for h in hourly_periods:
        dt = parse_iso_to_dt(h.get("startTime"))
        if dt:
            parsed.append((to_user_tz(dt, user_tz), h))
    parsed.sort(key=lambda x: x[0])
    if not parsed:
        return []
    # Most recently STARTED forecast period is the current period.
    idx = next((i for i in range(len(parsed)-1, -1, -1)
                if parsed[i][0] <= now_user), 0)
    return parsed[idx:idx+NUM_HOURLY_TO_SHOW]

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



# Shared HTML footers. All visible checked times are inside their panels.
# Use ASCII-only markup plus HTML numeric references for old Samsung/Tizen HTML parsers.
def _html_text(value):
    from html import escape
    return escape(str(value), quote=True).encode("ascii", "xmlcharrefreplace").decode("ascii")

def _spa_flag(value):
    """Interpret optional API status without treating string 'false' as True."""
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in ("true", "1", "yes", "on", "connected", "producing", "active"):
        return True
    if normalized in ("false", "0", "no", "off", "disconnected", "not producing", "inactive"):
        return False
    return None


def _spa_indicator(value, positive_label, negative_label, *, negative_style="bad"):
    """Generate the same compact status badge used by the spa connection."""
    status = _spa_flag(value)
    if status is True:
        label, style = positive_label, "ok"
    elif status is False:
        label, style = negative_label, negative_style
    else:
        label, style = "Unknown", "neutral"
    return f'<span class="spa-status {style}">{_html_text(label)}</span>'

def _checked_footer(timestamp, *, note=None):
    value = _html_text(_local_timestamp(timestamp)) if timestamp else "Not available"
    note_html = (f'<span class="footer-alert">{_html_text(note)}</span>' if note else "")
    return (f'<div class="panel-footer">{note_html}'
            f'<span class="footer-label">Last checked:</span><span>{value}</span></div>')

# Browser-active auto-refresh changes only the weather section, not spa API activity.
@st.fragment(run_every="5m")
def render_weather_section():
    # ---------- Unified weather panel (native Streamlit HTML, no iframe) ----------
    from html import escape as _html_escape
    weather_obj = None
    if lat is not None and lon is not None:
        try:
            weather_obj = get_nws_forecast_cached(lat, lon)
        except Exception as exc:
            _event("WEATHER_UPDATE", success=False, error=str(exc))
    if weather_obj:
        now_user = to_user_tz(now_utc(), USER_TZ)
        hourly = (weather_obj.get("forecastHourly") or {}).get("properties", {}).get("periods", [])
        dayparts = (weather_obj.get("forecast") or {}).get("properties", {}).get("periods", [])
        items = get_hourly_slice(hourly or dayparts, now_user, USER_TZ)
        hourly_cards = []
        for dt_user, it in items:
            label = "Now" if dt_user <= now_user else format_time_short(dt_user)
            temp_disp = convert_temp_for_display(it.get("temperature"), it.get("temperatureUnit", "F"), use_celsius)
            feels = compute_feels_like_for_period(it, use_celsius)
            pop = extract_pop(it)
            icon_url = _html_escape(it.get("icon") or "", quote=True)
            icon = '<img src="{}" alt="Weather">'.format(icon_url) if icon_url else ""
            unit = "C" if use_celsius else "F"
            extras = []
            if feels is not None and feels != temp_disp: extras.append("Feels {}&deg;".format(feels))
            if pop is not None: extras.append("POP {}%".format(pop))
            hourly_cards.append('<div class="weather-card"><div class="when">{}</div>{}<div class="temp">{}&deg;{}</div><div class="sub">{}</div></div>'.format(_html_escape(label),icon,temp_disp if temp_disp is not None else "--",unit,"<br>".join(extras)))
        buckets = {}
        for period in dayparts:
            dt = parse_iso_to_dt(period.get("startTime") or period.get("endTime") or "")
            if dt:
                day = to_user_tz(dt, USER_TZ).date()
                buckets.setdefault(day, []).append(period)
        today_local = now_user.date()
        daily_cards = []
        for day in [d for d in sorted(buckets) if d >= today_local][:5]:
            group = buckets[day]
            icon_url = next((p.get("icon") for p in group if p.get("isDaytime") and p.get("icon")), None)
            if not icon_url: icon_url = next((p.get("icon") for p in group if p.get("icon")), "")
            icon = '<img src="{}" alt="Forecast">'.format(_html_escape(icon_url, quote=True)) if icon_url else ""
            values = [convert_temp_for_display(p.get("temperature"),p.get("temperatureUnit","F"),use_celsius) for p in group]
            values = [v for v in values if v is not None]
            high, low = (max(values),min(values)) if values else ("--","--")
            common = next((p.get("shortForecast") for p in group if p.get("isDaytime") and p.get("shortForecast")), "")
            if not common: common = next((p.get("shortForecast") for p in group if p.get("shortForecast")), "")
            unit = "C" if use_celsius else "F"
            daily_cards.append('<div class="weather-card"><div class="when">{}</div>{}<div class="temp">{}&deg; / {}&deg;{}</div><div class="sub">{}</div></div>'.format(day.strftime("%a"),icon,high,low,unit,_html_escape(common[:32])))
        weather_html = (
            '<div class="section-heading">Weather Forecast - ZIP {}</div>'
            '<div class="weather-panel"><div class="weather-two">'
            '<div><div class="forecast-title">Next 6 hours</div><div class="weather-grid">{}</div></div>'
            '<div><div class="forecast-title">5-day outlook</div><div class="weather-grid daily">{}</div></div>'
            '</div>{}</div>'
        ).format(_html_text(zip_to_use), "".join(hourly_cards), "".join(daily_cards),
                 _checked_footer(weather_obj.get("fetched_at")))
        st.markdown(weather_html, unsafe_allow_html=True)
    else:
        weather_html = ('<div class="section-heading">Weather Forecast - ZIP {}</div>'
                        '<div class="weather-panel"><div class="spa-detail">No forecast available.</div>{}</div>')
        st.markdown(weather_html.format(_html_text(zip_to_use), _checked_footer(None)),
                    unsafe_allow_html=True)


render_weather_section()

# ============================================================
# Arctic Spa Status Card
# ============================================================
spa_result = get_spa_status_throttled()
# Initialize or repair missing daily ranges using today's cached spa reading.
# No new API request is made and the hourly polling gate is unaffected.
ensure_chem_ranges_from_cached_sample(spa_result, USER_TZ)
_sync_pump_session(USER_TZ)
_sync_chem_session(USER_TZ)
spa_footer = _checked_footer(
    spa_result.get("checked_at"),
    note="Cached reading; latest spa check failed" if spa_result.get("stale") else None,
)


if not spa_result.get("ok"):
    unavailable = _html_text(spa_result.get("error") or "no token / failed request")
    st.markdown(
        '<div class="section-heading spa">Monisha&#39;s Bath Tub</div>'
        f'<div class="spa-panel"><div class="spa-detail">Spa status unavailable: {unavailable}</div>'
        f'{spa_footer}</div>', unsafe_allow_html=True)
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


    if errors:
        other_not_ok = True; reasons.append(f"spa errors: {errors}")

    setpoint_val = None
    try:
        if setpoint is not None: setpoint_val = float(setpoint)
    except: pass

    temp_below_setpoint = (temp_val is not None and setpoint_val is not None
                           and temp_val < (setpoint_val - 4.0))
    trigger_alert = other_not_ok or temp_below_setpoint

    # Side effects are processed once per new successful sample, not once per
    # browser visit or Streamlit rerun. Four-hour cooldown is server-shared.
    effects = process_new_spa_observation(
        spa, spa_result.get("checked_at"), trigger_alert=trigger_alert,
        reasons=reasons, temp_below_setpoint=temp_below_setpoint,
        temp_val=temp_val, setpoint_val=setpoint_val,
        ph_val=ph_val, orp_val=orp_val, connected=connected,
        filter_status=filter_status, ph_status=ph_status,
        orp_status=orp_status, errors=errors)
    _sync_pump_session(USER_TZ)
    _sync_chem_session(USER_TZ)

    # ---- Build spa card HTML ----
    if trigger_alert:
        problems_str = " Â· ".join(reasons) if reasons else "temp below setpoint"
        alert_banner = f'<div class="alert-banner">âš ï¸ Alert condition detected â€” {problems_str}</div>'
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

    lights_cls = badge_class(lights)
    filter_cls = badge_class(filter_status)

    errors_html = ""
    if errors:
        err_items   = "".join(f'<div style="color:#ff8080;font-size:0.85rem;margin-top:4px">âš  {e}</div>' for e in errors)
        errors_html = f'<hr class="spa-divider"><div class="spa-section-label">Errors / Alerts</div>{err_items}'
    else:
        errors_html = '<hr class="spa-divider"><div style="color:#6effa8;font-size:0.85rem">âœ… No active errors</div>'

    filtration_html = ""
    if filtration_frequency or filtration_duration:
        filtration_html = f'<span class="spa-badge spa-badge-off">Freq: {filtration_frequency or "?"}/day &nbsp;|&nbsp; Dur: {filtration_duration or "?"} min</span>'

    def pump_inline(n, pval):
        rt    = get_pump_runtime_display(n)
        is_on = _is_pump_on(pval)
        dot   = f'<span style="color:{"#6effa8" if is_on else "#666"};font-size:0.7rem">{"â—" if is_on else "â—‹"}</span>'
        st_lbl = str(pval or "off")
        return (f'<span style="color:#a0c8e8;font-size:0.82rem">P{n}</span> {dot} '
                f'<span style="font-size:0.82rem;color:#ccc">{st_lbl}</span> '
                f'<span style="font-size:0.82rem;color:#7ecbf7;font-weight:600">{rt}</span>')

    p1_html = pump_inline(1, pump1)
    p2_html = pump_inline(2, pump2)
    p3_html = pump_inline(3, pump3)

    temp_html = chem_inline_html("Temp", temp_val, "Â°F", TEMP_MIN_F, TEMP_MAX_F,
                                 st.session_state.get("chem_min_temp"),
                                 st.session_state.get("chem_max_temp"), fmt=".1f")
    ph_html   = chem_inline_html("pH", ph_val, "", PH_MIN, PH_MAX,
                                 st.session_state.get("chem_min_ph"),
                                 st.session_state.get("chem_max_ph"), fmt=".2f")
    orp_html  = chem_inline_html("ORP", orp_val, " mV", ORP_MIN, ORP_MAX,
                                 st.session_state.get("chem_min_orp"),
                                 st.session_state.get("chem_max_orp"), fmt=".0f")

    # Native Streamlit HTML (no iframe) eliminates fixed-height clipping and
    # lets the status timestamp occupy the same bottom row as Weather.
    # Sanitize all provider strings before inserting them into HTML.
    def _display(value, suffix=""):
        return _html_text(value) + suffix if value is not None else "&mdash;"

    def _chem_row(label, value, lo, hi, unit, minimum, maximum, fmt):
        if value is None:
            shown, status, status_cls = "&mdash;", "Unavailable", "neutral"
        else:
            status_cls = "ok" if lo <= value <= hi else "bad"
            status = "OK" if status_cls == "ok" else "Check"
            shown = f"{value:{fmt}}{unit}"
        day = (f"{minimum:{fmt}}&ndash;{maximum:{fmt}}"
               if minimum is not None and maximum is not None else "&mdash;")
        return (f'<div class="spa-chem-row"><span>{label}</span><strong>{shown}</strong>'
                f'<span class="spa-status {status_cls}">{status}</span>'
                f'<small>Today {day}</small></div>')

    chem_rows = "".join([
        _chem_row("Temp", temp_val, TEMP_MIN_F, TEMP_MAX_F, "&deg;F",
                  st.session_state.get("chem_min_temp"), st.session_state.get("chem_max_temp"), ".1f"),
        _chem_row("pH", ph_val, PH_MIN, PH_MAX, "",
                  st.session_state.get("chem_min_ph"), st.session_state.get("chem_max_ph"), ".2f"),
        _chem_row("ORP", orp_val, ORP_MIN, ORP_MAX, " mV",
                  st.session_state.get("chem_min_orp"), st.session_state.get("chem_max_orp"), ".0f"),
    ])
    conn_status = "Connected" if connected else "Disconnected" if connected is False else "Unknown"
    conn_class = "ok" if connected else "bad" if connected is False else "neutral"
    temp_class = "bad" if temp_below_setpoint else "ok" if temp_val is not None else "neutral"
    # Connection and sanitizer production are independent observations.
    # Green mirrors the spa connection badge; gray means idle or unknown.
    spaboy_connection_html = _spa_indicator(
        spaboy_connected, "Connected", "Disconnected")
    spaboy_production_html = _spa_indicator(
        spaboy_producing, "Producing", "Not producing", negative_style="neutral")
    pumps_html = "".join(
        f'<div class="spa-pump-row"><strong>P{n}</strong>'
        f'<span>{_display(v)}</span>'
        f'<small>{_html_text(get_pump_runtime_display(n))} today (est.)</small></div>'
        for n, v in ((1, pump1), (2, pump2), (3, pump3))
    )
    filtration_detail = (
        f" &middot; {_display(filtration_frequency)}/day &middot; {_display(filtration_duration)} min"
        if filtration_frequency or filtration_duration else ""
    )
    issue_html = (
        '<div class="spa-issues">Alert: ' + _html_text("; ".join(reasons) if reasons else "temperature below setpoint") + '</div>'
        if trigger_alert else '<div class="spa-no-issues">No active alerts</div>'
    )
    if errors:
        issue_html += '<div class="spa-issues">' + _html_text("; ".join(str(e) for e in errors)) + '</div>'
    if effects.get("notification") == "failed":
        issue_html += '<div class="spa-issues">Notification failed. See Operations in Settings.</div>'
    if effects.get("setpoint") and not effects["setpoint"].get("ok"):
        issue_html += '<div class="spa-issues">Setpoint update failed. See Operations in Settings.</div>'

    spa_panel_html = (
        '<div class="section-heading spa">Monisha&#39;s Bath Tub</div>'
        '<div class="spa-panel">'
          '<div class="spa-three">'
            '<div class="spa-col"><div class="spa-subhead">WATER TEMPERATURE</div>'
              f'<div class="spa-temperature {temp_class}">{_display(temp, " &deg;F")}</div>'
              f'<div class="spa-setpoint">Setpoint: <strong>{_display(setpoint, " &deg;F")}</strong></div>'
              f'<div class="spa-detail">Spa connection: <span class="spa-status {conn_class}">{conn_status}</span></div>'
              f'<div class="spa-detail">SpaBoy connection: {spaboy_connection_html}</div>'
              f'<div class="spa-detail">SpaBoy production: {spaboy_production_html}</div></div>'
            '<div class="spa-col"><div class="spa-subhead">PUMPS &amp; EQUIPMENT</div>'
              f'{pumps_html}'
              f'<div class="spa-detail">Lights: <strong>{_display(lights)}</strong></div>'
              f'<div class="spa-detail">Filter: <strong>{_display(filter_status)}</strong>{filtration_detail}</div></div>'
            '<div class="spa-col"><div class="spa-subhead">WATER CHEMISTRY / DAILY MIN-MAX</div>'
              f'{chem_rows}'
              f'<div class="spa-detail" style="font-size:10px;color:#64748b">'
              f'Safe pH {PH_MIN}&ndash;{PH_MAX} &middot; ORP {ORP_MIN}&ndash;{ORP_MAX} mV'
              '</div></div>'
          '</div>'
          f'{issue_html}'
          f'{spa_footer}'
        '</div>'
    )
    st.markdown(spa_panel_html, unsafe_allow_html=True)

    with st.sidebar.expander("Raw spa payload (debug)"):
        st.json(spa)




# ---------- Solar: link only, no fabricated production data or background fetch. ----------
st.markdown(
    '<div class="section-heading solar">Solar System (Enphase)</div>'
    '<div class="solar-panel">'
    '<div class="solar-intro">'
    'Solar production and system status are available directly from Enphase. '
    f'<a href="{ENPHASE_PUBLIC_URL}" target="_blank" rel="noopener noreferrer">'
    'Open Solar System (new tab)</a>'
    '</div>'
    '<div class="panel-footer"><span>External Enphase page; no solar data is polled here.</span></div>'
    '</div>', unsafe_allow_html=True,
)

with st.sidebar.expander("Operations - last 30 events"):
    st.caption("Server-local diagnostic history; may reset after deployment/restart.")
    for line in reversed(_last_events(30)):
        st.code(line.strip(), language="json")

if st.sidebar.checkbox("Show debug", value=False):
    st.write("config:", config)
    st.write("zip:", zip_to_use)
    st.write("chem_ranges:", config.get("chem_ranges", {}))
    st.write("pump_runtimes (disk):", config.get("pump_runtimes", {}))
    st.write("last spa poll:", _read_ops().get("spa_checked_at"))
    st.caption("Pump runtimes are sample-based estimates. Short cycles and long offline gaps are not captured.")
    st.write("arcticspas installed:", ARCTICSPAS_INSTALLED)
    st.write("ntfy topic:", NTFY_TOPIC)
    for _n in [1, 2, 3]:
        daily_s  = float(st.session_state.get(f"pump_daily_seconds_{_n}", 0))
        live = 0
        total = daily_s
        st.write(f"pump_{_n}: sampled_daily={daily_s:.0f}s  "
                 f"display={get_pump_runtime_display(_n)}  state={st.session_state.get(f'pump_state_{_n}')}")
