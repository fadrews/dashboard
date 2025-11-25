# dashboard.py
"""
Daily headlines + NWS forecast dashboard (updated: multi-category + dedupe)
Features:
- Compact thumbnail layout for headlines (thumbnail left, text right) with denser spacing
- Multi-select news categories; selecting 'general' excludes other selections
- Deduplicates combined articles by URL (falls back to title)
- Browser timezone & locale detection (JS -> server)
- Relative hourly times and friendly formatting (uses client tz & locale when available)
- Caching for successful API calls only
- Icon caching to .cache_icons/ to reduce remote fetches
- Unit toggle (F/C)
- Auto-refresh (user-configurable interval) + manual refresh
- Graceful retries/backoff for remote calls (NewsAPI, NWS, zippopotam)
- Uses new Streamlit st.image(width='content' / 'stretch') API
Run:
    streamlit run dashboard.py
"""

from typing import Optional, Tuple, Dict, Any, List
import streamlit as st
import requests
import os
import json
import sys
import time
import hashlib
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Try zoneinfo for timezone conversions (Python 3.9+)
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

# ---------------- Constants / Paths ----------------
CONFIG_PATH = Path(".user_config.json")
ICON_CACHE_DIR = Path(".cache_icons")
ICON_CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_ZIP_IF_NO_JS = "84124"
DEFAULT_NEWS_API_KEY = "79c6795338c44f249006e46e2ab64456"

# ---------------- Helpers ----------------
def safe_rerun():
    try:
        if hasattr(st, "rerun"):
            st.rerun()
        elif hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
        else:
            raise AttributeError("st.rerun not available in this Streamlit version.")
    except Exception as e:
        print("Could not call Streamlit rerun():", e)
        print("Make sure you run the app with: streamlit run dashboard.py")
        sys.exit(0)

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def chunked_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

# ---------------- Network helper with retry/backoff ----------------
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
        else:
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

# ---------------- Networked APIs (cached on success only) ----------------
@st.cache_data(ttl=86400)
def geocode_zip_to_latlon_cached(zip_code: str) -> Tuple[float, float]:
    try:
        resp = retry_request("GET", f"https://api.zippopotam.us/us/{zip_code}", timeout=8)
        if resp.status_code != 200:
            raise RuntimeError(f"Zippopotam failed for {zip_code}: {resp.status_code}")
        d = resp.json()
        place = d["places"][0]
        return float(place["latitude"]), float(place["longitude"])
    except Exception as e:
        raise RuntimeError(f"Geocode error: {e}")

@st.cache_data(ttl=600)
def get_nws_forecast_cached(lat: float, lon: float) -> dict:
    base = "https://api.weather.gov/points/{lat},{lon}"
    headers = {"User-Agent": "streamlit-nws-app (contact@example.com)"}
    try:
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
    except Exception as e:
        raise RuntimeError(f"NWS API error: {e}")

@st.cache_data(ttl=300)
def fetch_news_cached(topic: str, api_key: str, page_size: int = 8) -> List[dict]:
    url = "https://newsapi.org/v2/top-headlines"
    params = {"category": topic if topic != "general" else None, "country": "us", "pageSize": page_size}
    params = {k: v for k, v in params.items() if v is not None}
    headers = {"X-Api-Key": api_key}
    try:
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
    except Exception as e:
        raise RuntimeError(f"News API fetch failed: {e}")

# ---------------- Parsing / formatting helpers ----------------
def parse_iso_to_dt(iso_ts: str) -> Optional[datetime]:
    if not iso_ts:
        return None
    try:
        dt = datetime.fromisoformat(iso_ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        try:
            if iso_ts.endswith("Z"):
                return datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
        except Exception:
            return None

def to_user_tz(dt: datetime, user_tz) -> datetime:
    if dt is None:
        return dt
    try:
        return dt.astimezone(user_tz)
    except Exception:
        return dt.astimezone(timezone.utc)

def use_12_hour_format(locale_str: Optional[str]) -> bool:
    if not locale_str:
        return True
    loc = locale_str.lower()
    if "en_us" in loc or "en-us" in loc or loc.startswith("en"):
        return True
    if loc.startswith(("de", "fr", "es", "it", "nl", "pt", "ru", "zh", "jp", "ko")):
        return False
    return True

def format_time_for_display(dt_user: datetime, locale_str: Optional[str]) -> str:
    try:
        if use_12_hour_format(locale_str):
            try:
                return dt_user.strftime("%-I %p")
            except Exception:
                return dt_user.strftime("%I %p").lstrip("0")
        else:
            return dt_user.strftime("%H:%M")
    except Exception:
        return dt_user.isoformat()

def format_relative(dt_user: datetime, now_user: datetime) -> str:
    delta = dt_user - now_user
    total_seconds = int(delta.total_seconds())
    if abs(total_seconds) < 60:
        return "now"
    if total_seconds > 0:
        mins = (total_seconds + 59) // 60
        if mins < 60:
            return f"in {mins}m"
        hours = mins // 60
        rem_m = mins % 60
        return f"in {hours}h{(' ' + str(rem_m) + 'm') if rem_m else ''}"
    else:
        mins = (-total_seconds) // 60
        if mins < 60:
            return f"{mins}m ago"
        hours = mins // 60
        rem_m = mins % 60
        return f"{hours}h{(' ' + str(rem_m) + 'm') if rem_m else ''}"

# ---------------- App start ----------------
st.set_page_config(layout="wide", page_title="News + NWS Forecast", initial_sidebar_state="expanded")

# Load config
config = load_config()
saved_zip = config.get("zip_code")

# Session defaults
if "use_defaults_no_js" not in st.session_state:
    st.session_state["use_defaults_no_js"] = False
if "last_auto_refresh" not in st.session_state:
    st.session_state["last_auto_refresh"] = time.time()

# Read query params from JS (ua, sw, lat, lon, geo_acquired, tz, locale)
query_params = st.query_params
ua_param = query_params.get("ua", [None])[0]
screen_width_param = query_params.get("sw", [None])[0]
lat_param = query_params.get("lat", [None])[0]
lon_param = query_params.get("lon", [None])[0]
geo_acquired = query_params.get("geo_acquired", [None])[0]
tz_param = query_params.get("tz", [None])[0]
locale_param = query_params.get("locale", [None])[0]

# Decide USER_TZ (use browser tz if provided and valid)
if tz_param and ZoneInfo:
    try:
        USER_TZ = ZoneInfo(tz_param)
    except Exception:
        USER_TZ = ZoneInfo("America/Denver") if ZoneInfo else timezone.utc
else:
    USER_TZ = ZoneInfo("America/Denver") if ZoneInfo else timezone.utc

# Inject JS to collect UA, screen width, geolocation, tz, and locale when missing
if (not ua_param or not screen_width_param) and not st.session_state["use_defaults_no_js"]:
    st.markdown(
        """
        <div style="border:1px solid #ddd;padding:12px;border-radius:8px;">
        <strong>Device detection</strong><br>
        The app will detect your device, timezone & locale with a small JavaScript snippet. If JavaScript is disabled, click the button to proceed with defaults.
        </div>
        """,
        unsafe_allow_html=True,
    )

    js = """
    <script>
    (function() {
      function encode(s) { return encodeURIComponent(s); }
      function setParamsAndReload(params) {
        var url = window.location.href.split('?')[0];
        var qp = [];
        for (var k in params) {
          if (params[k]!==null && params[k]!==undefined) qp.push(k + '=' + encode(params[k]));
        }
        var newurl = url + '?' + qp.join('&');
        window.history.replaceState({}, '', newurl);
        setTimeout(function(){ window.location.reload(); }, 130);
      }

      var ua = navigator.userAgent || '';
      var sw = window.innerWidth || screen.width || 0;
      var tz = null;
      var locale = null;
      try {
        tz = Intl.DateTimeFormat().resolvedOptions().timeZone || null;
        locale = Intl.DateTimeFormat().resolvedOptions().locale || null;
      } catch(e) {
        tz = null;
        locale = null;
      }

      if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(function(pos) {
          setParamsAndReload({ua: ua, sw: sw, lat: pos.coords.latitude, lon: pos.coords.longitude, geo_acquired: 1, tz: tz, locale: locale});
        }, function(err) {
          setParamsAndReload({ua: ua, sw: sw, geo_acquired: 0, tz: tz, locale: locale});
        }, {timeout:5000});
      } else {
        setParamsAndReload({ua: ua, sw: sw, geo_acquired: 0, tz: tz, locale: locale});
      }
    })();
    </script>
    """
    st.components.v1.html(js, height=1)

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Use defaults (no JS): ZIP=84124, Desktop"):
            st.session_state["use_defaults_no_js"] = True
            st.session_state["fallback_zip"] = DEFAULT_ZIP_IF_NO_JS
            st.session_state["fallback_layout"] = "Desktop"
            safe_rerun()
    with c2:
        st.markdown("If JavaScript is enabled but the app doesn't detect it, try reloading the page or disabling strict content blockers for this site.")
    st.stop()

# Effective UA/screen/coords
use_defaults_no_js = st.session_state.get("use_defaults_no_js", False)
if use_defaults_no_js:
    ua_info = {"raw": "", "os": "Unknown"}
    screen_width = 1200
    chosen_zip = st.session_state.get("fallback_zip", DEFAULT_ZIP_IF_NO_JS)
else:
    try:
        screen_width = int(screen_width_param) if screen_width_param is not None else None
    except (ValueError, TypeError):
        screen_width = None
    ua_info = {"raw": ua_param or "", "os": "Unknown"}
    chosen_zip = None

# ---------------- Sidebar UI ----------------
with st.sidebar:
    st.title("Settings / Configuration")
    st.write(f"Detected browser OS: **{ua_info.get('os','Unknown')}**")
    if screen_width:
        st.write(f"Screen width: **{screen_width}px**")
    st.write(f"Detected timezone: **{tz_param or 'unknown'}**")
    st.write(f"Detected locale: **{locale_param or 'unknown'}**")
    st.markdown("---")

    st.subheader("News options")
    # Multi-category selector with general-exclusion logic
    all_topics = ["general", "business", "entertainment", "health", "science", "sports", "technology"]
    selected_topics = st.multiselect(
        "Select news categories",
        options=all_topics,
        default=["general"],
        help="If 'general' is selected, other categories will be ignored."
    )
    if "general" in selected_topics and len(selected_topics) > 1:
        st.warning("'General' includes all categories, so other selections are ignored.")
        selected_topics = ["general"]
    # topic_list to use later
    topic_list = selected_topics

    api_key_env = os.getenv("NEWS_API_KEY", DEFAULT_NEWS_API_KEY)
    api_key_input = st.text_input("News API key (session/paste)", value=(api_key_env or DEFAULT_NEWS_API_KEY), type="password", help="Default key is built in; you can override it here.")
    st.caption("Using default NewsAPI key unless overridden.")
    num_headlines = st.slider("Number of headlines (final combined)", 5, 100, 20)

    st.markdown("---")
    st.subheader("Forecast / location")
    unit_choice = st.radio("Temperature unit", options=["°F", "°C"], index=0)
    use_celsius = unit_choice == "°C"

    st.markdown("**Auto-refresh**")
    auto_refresh_enabled = st.checkbox("Enable auto-refresh", value=False)
    refresh_minutes = st.number_input("Refresh interval (minutes)", min_value=1, max_value=120, value=10, step=1)
    if st.button("Refresh now"):
        st.session_state["last_auto_refresh"] = time.time()
        safe_rerun()

    st.markdown("---")
    st.subheader("Forecast location method")
    if use_defaults_no_js:
        location_method_default_index = 2
    else:
        location_method_default_index = 0 if (geo_acquired == "1") else (1 if saved_zip else 2)

    location_method = st.radio(
        "Use",
        options=["geolocation (browser)", "saved ZIP", "enter ZIP", "enter lat/lon"],
        index=location_method_default_index,
    )

    st.markdown("**Change saved ZIP**")
    change_zip = st.text_input("New ZIP to save (leave blank to keep current)", value="")
    if st.button("Save new ZIP"):
        if change_zip.strip():
            try:
                latlon = geocode_zip_to_latlon_cached(change_zip.strip())
                config["zip_code"] = change_zip.strip()
                save_config(config)
                saved_zip = change_zip.strip()
                st.success(f"Saved ZIP {saved_zip}")
                safe_rerun()
            except Exception as e:
                st.error(str(e))
        else:
            st.info("No ZIP entered; nothing changed.")

    st.markdown("---")
    if st.button("Clear stored ZIP"):
        clear_config()
        config.clear()
        saved_zip = None
        st.success("Cleared stored ZIP.")
        safe_rerun()

    st.markdown("---")
    if st.button("Clear all caches"):
        st.cache_data.clear()
        st.success("Caches cleared.")
        safe_rerun()

    compact = st.checkbox("Compact news cards", value=True)

# ---------------- Main UI header ----------------
st.title("Daily Headlines & NWS Forecast")
st.write("Headlines (NewsAPI) + location-specific NWS forecast. Uses browser timezone & locale when available.")

if use_defaults_no_js:
    chosen_zip = chosen_zip or DEFAULT_ZIP_IF_NO_JS
    st.info(f"JavaScript unavailable — using defaults: ZIP {chosen_zip}, Desktop layout.")

# Determine coordinates
chosen_lat = None
chosen_lon = None

if location_method == "geolocation (browser)" and not use_defaults_no_js:
    if geo_acquired == "1" and lat_param and lon_param:
        try:
            chosen_lat = float(lat_param)
            chosen_lon = float(lon_param)
            st.info("Using browser geolocation for forecast.")
        except Exception:
            chosen_lat = None
            chosen_lon = None
    else:
        st.warning("Browser geolocation not available or not allowed. Choose another method.")

elif location_method == "saved ZIP":
    if saved_zip:
        try:
            latlon = geocode_zip_to_latlon_cached(saved_zip)
            chosen_lat, chosen_lon = latlon
            st.info(f"Using saved ZIP {saved_zip} for forecast.")
        except Exception as e:
            st.error(str(e))
    else:
        st.warning("No saved ZIP; choose a different method or save a ZIP in Configuration.")

elif location_method == "enter ZIP":
    initial_zip = chosen_zip or saved_zip or ""
    zip_code = st.text_input("ZIP code (US)", value=initial_zip)
    if zip_code:
        try:
            latlon = geocode_zip_to_latlon_cached(zip_code.strip())
            chosen_lat, chosen_lon = latlon
            if st.checkbox("Save this ZIP as default"):
                config["zip_code"] = zip_code.strip()
                save_config(config)
                saved_zip = zip_code.strip()
                st.success(f"Saved ZIP {saved_zip}")
        except Exception as e:
            st.error(str(e))

elif location_method == "enter lat/lon":
    manual_lat = st.text_input("Latitude", value=(lat_param or ""))
    manual_lon = st.text_input("Longitude", value=(lon_param or ""))
    if manual_lat and manual_lon:
        try:
            chosen_lat = float(manual_lat)
            chosen_lon = float(manual_lon)
        except Exception:
            st.error("Invalid numeric lat/lon.")

if chosen_lat is None and chosen_lon is None and (chosen_zip or saved_zip):
    z_to_use = chosen_zip or saved_zip
    if z_to_use:
        try:
            chosen_lat, chosen_lon = geocode_zip_to_latlon_cached(z_to_use)
            if chosen_zip:
                st.info(f"Using ZIP {z_to_use} for forecast (fallback/default).")
        except Exception as e:
            st.error(str(e))

# Auto-refresh
if auto_refresh_enabled:
    last = st.session_state.get("last_auto_refresh", 0)
    elapsed = time.time() - last
    if elapsed >= (refresh_minutes * 60):
        st.session_state["last_auto_refresh"] = time.time()
        safe_rerun()

# Fetch NWS forecast
weather_obj = None
weather_error = None
if chosen_lat is not None and chosen_lon is not None:
    with st.spinner("Fetching NWS forecast..."):
        try:
            weather_obj = get_nws_forecast_cached(chosen_lat, chosen_lon)
        except Exception as e:
            weather_error = str(e)
else:
    st.info("No coordinates chosen yet. Select a location method in the sidebar or enter a ZIP/lat-lon.")

# ---------------- Fetch news for multiple categories + dedupe ----------------
news_api_key = api_key_input.strip() or api_key_env or None

combined_articles: List[dict] = []
fetch_errors: List[str] = []
if news_api_key:
    for t in topic_list:
        try:
            # 'general' means no category param (handled inside fetch_news_cached)
            arts = fetch_news_cached(t, news_api_key, page_size=num_headlines)
            combined_articles.extend(arts)
        except Exception as e:
            fetch_errors.append(f"{t}: {e}")
else:
    fetch_errors.append("No News API key provided; set one in the sidebar or env var.")

if fetch_errors:
    for fe in fetch_errors:
        st.info(fe)

# Deduplicate by URL (fallback to title), keep newest by publishedAt
def dedupe_and_sort(articles: List[dict], limit: int) -> List[dict]:
    seen = {}
    for art in articles:
        key = art.get("url") or art.get("title") or ""
        # Normalize key
        key = (key or "").strip()
        # Parse date
        pub = art.get("publishedAt")
        try:
            dt = datetime.fromisoformat(pub.replace("Z", "+00:00")) if pub else datetime.min
        except Exception:
            dt = datetime.min
        # If unseen or newer, keep
        if key not in seen or dt > seen[key]["_parsed_pub"]:
            art_copy = art.copy()
            art_copy["_parsed_pub"] = dt
            seen[key] = art_copy
    # produce list sorted by date desc
    items = list(seen.values())
    items.sort(key=lambda x: x.get("_parsed_pub", datetime.min), reverse=True)
    # remove helper field before returning
    for it in items:
        if "_parsed_pub" in it:
            del it["_parsed_pub"]
    return items[:limit]

articles = dedupe_and_sort(combined_articles, num_headlines)

# Mobile/desktop detection
is_mobile = (screen_width is not None and screen_width < 700) if not use_defaults_no_js else False

# ---------------- Rendering helpers (denser layout) ----------------
def render_article_compact(art: dict, compact_mode: bool):
    col_img, col_text = st.columns([0.9, 9], gap="small")
    with col_img:
        if art.get("urlToImage"):
            try:
                img_local = get_cached_icon_path(art.get("urlToImage")) or art.get("urlToImage")
                st.image(img_local, width='content')
            except Exception:
                pass
    with col_text:
        title = art.get("title") or ""
        url = art.get("url") or ""
        src = art.get("source", {}).get("name", "")
        time_str = (art.get("publishedAt") or "")[:10]
        st.markdown(f"**[{title}]({url})**  ")
        if src:
            st.caption(f"{src} — {time_str}")

# Forecast helpers
NUM_HOURLY_TO_SHOW = 6

def display_forecast_periods(periods: List[dict], user_tz, locale_str: Optional[str], now_user: datetime):
    for per in periods:
        c_icon, c_text = st.columns([0.6, 5], gap="small")
        with c_icon:
            icon_local = get_cached_icon_path(per.get("icon")) or per.get("icon")
            if icon_local:
                try:
                    st.image(icon_local, width='content')
                except Exception:
                    pass
        with c_text:
            st.markdown(f"**{per.get('name')}** — {per.get('temperature')}°{'C' if use_celsius else 'F'}")
            st.write(per.get('shortForecast') or "")

def display_hourly_next(hourly_periods: List[dict], user_tz, locale_str: Optional[str], now_user: datetime):
    parsed: List[Tuple[datetime, dict]] = []
    for h in hourly_periods:
        dt = parse_iso_to_dt(h.get("startTime"))
        if dt:
            parsed.append((dt, h))
    if not parsed:
        slice_items = [(parse_iso_to_dt(h.get("startTime")) or now_user, h) for h in hourly_periods[:NUM_HOURLY_TO_SHOW]]
    else:
        parsed_user = [(to_user_tz(dt, user_tz), h) for dt, h in parsed]
        idx = 0
        for i, (dt_user, h) in enumerate(parsed_user):
            if dt_user >= (now_user - timedelta(minutes=30)):
                idx = i
                break
        slice_items = parsed_user[idx: idx + NUM_HOURLY_TO_SHOW]
        if len(slice_items) < NUM_HOURLY_TO_SHOW:
            slice_items = parsed_user[:NUM_HOURLY_TO_SHOW]

    for dt_user, h in slice_items:
        rel = format_relative(dt_user, now_user)
        time_str = format_time_for_display(dt_user, locale_param)
        display_label = f"{rel} ({time_str})"
        c_icon, c_text = st.columns([0.6, 5], gap="small")
        with c_icon:
            icon_local = get_cached_icon_path(h.get("icon")) or h.get("icon")
            if icon_local:
                try:
                    st.image(icon_local, width='content')
                except Exception:
                    pass
        with c_text:
            temp_c = None
            try:
                temp = h.get("temperature")
                unit = h.get("temperatureUnit", "F")
                if use_celsius:
                    if unit.upper() == "F":
                        temp_c = round((float(temp) - 32) * 5.0 / 9.0)
                    else:
                        temp_c = int(temp)
                else:
                    if unit.upper() == "F":
                        temp_c = int(temp)
                    else:
                        temp_c = round((float(temp) * 9.0 / 5.0) + 32)
            except Exception:
                temp_c = h.get("temperature")
            st.write(f"{display_label} — {temp_c}°{'C' if use_celsius else 'F'}, {h.get('shortForecast')}")

# ---------------- Render UI ----------------
if is_mobile:
    st.subheader("Mobile view")
    st.markdown("### Headlines")
    if articles:
        for art in articles:
            render_article_compact(art, compact)
    else:
        st.info("No headlines to show.")

    st.markdown("### Forecast")
    if weather_obj:
        p = weather_obj["point"]["properties"]
        city = p.get('relativeLocation', {}).get('properties', {}).get('city', '')
        state = p.get('relativeLocation', {}).get('properties', {}).get('state', '')
        st.write(f"Location: **{city}, {state}**")
        if weather_obj.get("forecast") and weather_obj["forecast"].get("properties"):
            periods = weather_obj["forecast"]["properties"].get("periods", [])[:6]
            display_forecast_periods(periods, USER_TZ, locale_param, to_user_tz(now_utc(), USER_TZ))
        if weather_obj.get("forecastHourly") and weather_obj["forecastHourly"].get("properties"):
            st.markdown("**Hourly (next few hours)**")
            hourly_periods = weather_obj["forecastHourly"]["properties"].get("periods", [])
            display_hourly_next(hourly_periods, USER_TZ, locale_param, to_user_tz(now_utc(), USER_TZ))
    else:
        if weather_error:
            st.warning(weather_error)
else:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### Headlines")
        if articles:
            for art in articles:
                render_article_compact(art, compact)
        else:
            st.info("No headlines to show.")
    with col2:
        st.markdown("### Forecast (NWS)")
        if weather_obj:
            p = weather_obj["point"]["properties"]
            rel_loc_props = p.get('relativeLocation', {}).get('properties', {})
            city = rel_loc_props.get("city", "")
            state = rel_loc_props.get("state", "")
            st.write(f"Location: **{city}, {state}**")
            st.write(f"Grid: {p.get('gridId','')} {p.get('gridX','')},{p.get('gridY','')}")
            st.write("---")
            if weather_obj.get("forecast") and weather_obj["forecast"].get("properties"):
                periods = weather_obj["forecast"]["properties"].get("periods", [])[:6]
                display_forecast_periods(periods, USER_TZ, locale_param, to_user_tz(now_utc(), USER_TZ))
            else:
                st.info("No forecast data available.")
            st.write("---")
            if weather_obj.get("forecastHourly") and weather_obj["forecastHourly"].get("properties"):
                st.markdown("**Hourly (next few hours)**")
                hourly_periods = weather_obj["forecastHourly"]["properties"].get("periods", [])
                display_hourly_next(hourly_periods, USER_TZ, locale_param, to_user_tz(now_utc(), USER_TZ))
            st.write("---")
        else:
            if weather_error:
                st.warning(weather_error)

# Footer / debug
st.markdown("---")
st.caption("Notes: Images/icons are cached locally in .cache_icons/; API results are cached on success only.")
if st.checkbox("Show debug info (query params / config)"):
    st.write("query params:", dict(query_params))
    st.write("loaded config:", config)
    st.write("detected UA raw:", ua_info.get("raw", ""))
    st.write("detected tz param:", tz_param)
    st.write("detected locale param:", locale_param)
    st.write("USER_TZ:", getattr(USER_TZ, 'key', str(USER_TZ)))
    st.write("last_auto_refresh:", st.session_state.get("last_auto_refresh"))
