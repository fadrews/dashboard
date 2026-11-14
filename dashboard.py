
    
  
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

    # Three-column light panel. All spa measurements remain on the main page.
    # Use the existing HTML component on Streamlit 1.65 for older Tizen compatibility.
    def _display(value, suffix=""):
        return f"{value}{suffix}" if value is not None else "—"

    def _chem_row(label, value, lo, hi, unit, minimum, maximum, fmt):
        if value is None:
            status, shown = "Unavailable", "—"
            status_cls = "neutral"
        else:
            status_cls = "ok" if lo <= value <= hi else "bad"
            status = "OK" if status_cls == "ok" else "Check"
            shown = f"{value:{fmt}}{unit}"
        day = (f"{minimum:{fmt}}–{maximum:{fmt}}" if minimum is not None and maximum is not None else "—")
        return (f'<div class="chem-row"><span>{label}</span><strong>{shown}</strong>'
                f'<span class="status {status_cls}">{status}</span>'
                f'<small>Today {day}</small></div>')

    chem_rows = "".join([
        _chem_row("Temp", temp_val, TEMP_MIN_F, TEMP_MAX_F, "°F",
                  st.session_state.get("chem_min_temp"), st.session_state.get("chem_max_temp"), ".1f"),
        _chem_row("pH", ph_val, PH_MIN, PH_MAX, "",
                  st.session_state.get("chem_min_ph"), st.session_state.get("chem_max_ph"), ".2f"),
        _chem_row("ORP", orp_val, ORP_MIN, ORP_MAX, " mV",
                  st.session_state.get("chem_min_orp"), st.session_state.get("chem_max_orp"), ".0f"),
    ])
    conn_status = "Connected" if connected else "Disconnected" if connected is False else "Unknown"
    conn_class = "ok" if connected else "bad" if connected is False else "neutral"
    temp_class = "bad" if temp_below_setpoint else "ok" if temp_val is not None else "neutral"
    spa_boy_status = ("Producing" if spaboy_producing else "Connected" if spaboy_connected else "Disconnected") if spaboy_connected is not None else "Unknown"
    pumps_html = "".join(
        f'<div class="pump-row"><strong>P{n}</strong><span>{_display(v)}</span><small>{get_pump_runtime_display(n)} today</small></div>'
        for n, v in ((1, pump1), (2, pump2), (3, pump3))
    )
    filtration_detail = (f" · {filtration_frequency or '?'}×/day · {filtration_duration or '?'} min"
                         if filtration_frequency or filtration_duration else "")
    issue_html = (f'<div class="issues">⚠ {problems_str}</div>' if trigger_alert else
                  '<div class="no-issues">No active alerts</div>')
    if errors:
        issue_html += '<div class="issues">' + ' · '.join(str(e) for e in errors) + '</div>'

    spa_card_html = f"""
    <!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1">
    <style>
    *{{box-sizing:border-box}} body{{margin:0;font-family:Arial,sans-serif;color:#243447;background:#fff}}
    .spa-card{{border:1px solid #efd8bf;border-radius:9px;padding:10px 12px;background:#fff;}}
    .three{{display:grid;grid-template-columns:1.05fr 1fr 1.45fr;gap:0;}}
    .col{{padding:3px 12px;min-width:0;}}
    .col:first-child{{padding-left:2px}} .col+.col{{border-left:1px solid #dce2e8}}
    .subhead{{font-size:12px;font-weight:700;color:#64748b;margin:0 0 8px}}
    .temperature{{font-size:32px;font-weight:800;color:#273d4d;line-height:1.1}}
    .temperature.bad{{color:#b42318}} .setpoint{{font-size:14px;margin:5px 0 9px}}
    .status{{display:inline-block;padding:2px 7px;border-radius:12px;font-size:11px;font-weight:700;white-space:nowrap}}
    .status.ok{{background:#def4e4;color:#146c38}} .status.bad{{background:#ffe0dc;color:#a62319}}
    .status.neutral{{background:#edf0f3;color:#53616d}}
    .detail{{font-size:12px;margin:5px 0;line-height:1.35}}
    .pump-row{{display:grid;grid-template-columns:30px 1fr auto;gap:5px;align-items:center;font-size:13px;
      padding:3px 0;border-bottom:1px solid #edf0f3}}
    .pump-row small{{font-size:11px;color:#637383}}
    .chem-row{{display:grid;grid-template-columns:40px minmax(68px,1fr) 48px minmax(78px,1fr);
      gap:5px;align-items:center;padding:5px 0;border-bottom:1px solid #edf0f3;font-size:13px}}
    .chem-row small{{font-size:10px;color:#637383;white-space:nowrap}}
    .issues{{background:#fff1ed;color:#a62319;padding:5px 7px;margin-top:7px;border-radius:5px;font-size:12px}}
    .no-issues{{color:#217447;font-size:11px;margin-top:6px}}
    @media(max-width:650px){{.three{{grid-template-columns:1fr 1fr}} .col:last-child{{grid-column:1/-1;
      border-left:0;border-top:1px solid #dce2e8;padding-top:8px;margin-top:7px}}}}
    </style></head><body><div class="spa-card"><div class="three">
      <div class="col"><div class="subhead">WATER TEMPERATURE</div>
        <div class="temperature {temp_class}">{_display(temp, ' °F')}</div>
        <div class="setpoint">Setpoint: <strong>{_display(setpoint, ' °F')}</strong></div>
        <div class="detail">Connection: <span class="status {conn_class}">{conn_status}</span></div>
        <div class="detail">SpaBoy: {spa_boy_status}</div></div>
      <div class="col"><div class="subhead">PUMPS &amp; EQUIPMENT</div>
        {pumps_html}
        <div class="detail">Lights: <strong>{_display(lights)}</strong></div>
        <div class="detail">Filter: <strong>{_display(filter_status)}</strong>{filtration_detail}</div></div>
      <div class="col"><div class="subhead">WATER CHEMISTRY · DAILY MIN–MAX</div>
        {chem_rows}
        <div class="detail" style="font-size:10px;color:#64748b">Safe pH {PH_MIN}–{PH_MAX} · ORP {ORP_MIN}–{ORP_MAX} mV</div></div>
    </div>{issue_html}</div></body></html>
    """
    st.components.v1.html(spa_card_html.encode("ascii", "xmlcharrefreplace").decode("ascii"), height=205 if not trigger_alert else 275, scrolling=False)

    with st.sidebar.expander("Raw spa payload (debug)"):
        st.json(spa)



# ---------- Solar: link only. No unsupported real-time solar measurements. ----------
st.markdown('<div class="section-heading solar">Solar System (Enphase)</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="section-note">Solar production and system status are available on the Enphase website. '
    f'<a href="{ENPHASE_PUBLIC_URL}" target="_blank" rel="noopener noreferrer">'
    f'Open Solar System (new tab)</a></div>', unsafe_allow_html=True)

if st.sidebar.checkbox("Show debug", value=False):
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
