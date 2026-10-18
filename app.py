import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trafficflow.detector import (analyse_road, congestion_level,
                                  forecast_speed, isolation_forest_flags)
from trafficflow.simulator import ROADS, TrafficSimulator

st.set_page_config(page_title="TrafficFlow AI", page_icon="🚦", layout="wide")
st.title("🚦 TrafficFlow AI")
st.caption("Live traffic monitoring, anomaly detection and explainable alerts (simulated feed)")

if "sim" not in st.session_state:
    st.session_state.sim = TrafficSimulator()
    st.session_state.history = st.session_state.sim.warmup(120)
    st.session_state.alerts = []

with st.sidebar:
    st.header("Controls")
    paused = st.toggle("Pause feed", False)
    window = st.slider("Baseline window (min)", 10, 60, 30)
    z_thresh = st.slider("Sensitivity (z-threshold)", 1.5, 5.0, 3.0, 0.5,
                         help="Lower = more alerts")
    persist = st.slider("Persistence (intervals)", 1, 6, 3)
    show_forecast = st.checkbox("Show 10-min speed forecast", True)
    use_ml = st.checkbox("Compare with Isolation Forest", False)
    st.divider()
    road_pick = st.selectbox("Simulate incident on", list(ROADS))
    if st.button("Inject incident"):
        st.session_state.sim.inject_incident(road_pick)


@st.fragment(run_every=2)
def live():
    ss = st.session_state
    if not paused:
        ss.history = pd.concat([ss.history, ss.sim.step()], ignore_index=True).tail(2000)
    hist = ss.history
    latest_alerts = {}
    for road, g in hist.groupby("road"):
        a = analyse_road(g, window, z_thresh, persist)
        if a:
            latest_alerts[road] = a
            if not ss.alerts or ss.alerts[-1]["road"] != road or ss.alerts[-1]["time"] != a["time"]:
                recent = [x for x in ss.alerts[-20:] if x["road"] == road]
                if not recent or (a["time"] - recent[-1]["time"]).total_seconds() > 600:
                    ss.alerts.append(a)

    cols = st.columns(len(ROADS))
    for col, (road, g) in zip(cols, hist.groupby("road")):
        cur, ref = g.iloc[-1], g["speed"].tail(60).quantile(0.9)
        level = congestion_level(cur["speed"], ref)
        icon = {"Free": "🟢", "Moderate": "🟡", "Heavy": "🟠", "Severe": "🔴"}[level]
        col.metric(f"{icon} {road}", f"{cur['speed']:.0f} km/h", f"{level}", delta_color="off")
        col.caption(f"{cur['volume']:.0f} vehicles/min")

    for a in latest_alerts.values():
        (st.error if a["severity"] == "Critical" else st.warning)(f"**{a['road']}** — {a['message']}")
    if not latest_alerts:
        st.success("No active anomalies.")

    c1, c2 = st.columns(2)
    for chart_col, metric, title in ((c1, "speed", "Average speed (km/h)"), (c2, "volume", "Vehicles per minute")):
        fig = go.Figure()
        for road, g in hist.groupby("road"):
            g = g.tail(120)
            fig.add_scatter(x=g["time"], y=g[metric], name=road, mode="lines")
            if metric == "speed" and show_forecast:
                fc = forecast_speed(g)
                if fc:
                    ft = [g["time"].iloc[-1] + pd.Timedelta(minutes=i + 1) for i in range(len(fc))]
                    fig.add_scatter(x=ft, y=fc, name=f"{road} forecast", mode="lines",
                                    line=dict(dash="dot"), showlegend=False)
            if metric == "speed" and use_ml:
                flags = isolation_forest_flags(g)
                fig.add_scatter(x=g.loc[flags, "time"], y=g.loc[flags, "speed"], mode="markers",
                                marker=dict(symbol="x", size=7), name=f"{road} IF", showlegend=False)
        fig.update_layout(title=title, height=330, margin=dict(l=10, r=10, t=40, b=10),
                          legend=dict(orientation="h"))
        chart_col.plotly_chart(fig, use_container_width=True)

    st.subheader("Alert history")
    if ss.alerts:
        st.dataframe(pd.DataFrame(ss.alerts[::-1])[["time", "road", "severity", "message"]],
                     use_container_width=True, hide_index=True)
    else:
        st.caption("No alerts yet. Use 'Inject incident' in the sidebar.")


live()
