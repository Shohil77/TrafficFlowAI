"""Baseline-deviation anomaly detection, forecasting and explainable alerts."""
import numpy as np
import pandas as pd

FREE_FLOW_REF = 0.5  # congestion = 1 - speed / recent 90th-percentile speed


def congestion_level(speed, ref_speed):
    ratio = 1 - speed / max(ref_speed, 1e-6)
    return "Severe" if ratio > 0.5 else "Heavy" if ratio > 0.3 else "Moderate" if ratio > 0.15 else "Free"


def analyse_road(df, window=30, z_thresh=3.0, persist=3):
    """df: one road's history sorted by time. Returns an alert dict or None."""
    if len(df) < window + persist:
        return None
    base = df.iloc[-(window + persist):-persist]
    recent = df.iloc[-persist:]
    s_mu, s_sd = base["speed"].mean(), max(base["speed"].std(), 1e-6)
    v_mu, v_sd = base["volume"].mean(), max(base["volume"].std(), 1e-6)
    z_speed = (recent["speed"] - s_mu) / s_sd
    z_vol = (recent["volume"] - v_mu) / v_sd
    if not (z_speed < -z_thresh).all():
        return None
    speed_now, vol_now = recent["speed"].mean(), recent["volume"].mean()
    drop = (s_mu - speed_now) / s_mu * 100
    vol_up = (vol_now - v_mu) / v_mu * 100
    severity = "Critical" if drop > 50 else "Warning"
    msg = (f"{'High' if severity == 'Critical' else 'Elevated'} congestion detected: "
           f"average speed is {speed_now:.0f} km/h, {drop:.0f}% below its recent baseline "
           f"of {s_mu:.0f} km/h (z = {z_speed.mean():.1f}), sustained for {persist} intervals.")
    if z_vol.mean() > 1.5:
        msg += f" Traffic volume is also up {vol_up:.0f}% vs baseline."
    return {"time": df["time"].iloc[-1], "road": df["road"].iloc[-1], "severity": severity,
            "speed": speed_now, "baseline": s_mu, "drop_pct": drop, "message": msg}


def forecast_speed(df, horizon=10, alpha=0.4):
    """Holt linear exponential smoothing on recent speeds; returns list of forecasts."""
    y = df["speed"].tail(40).to_numpy()
    if len(y) < 10:
        return []
    level, trend = y[0], y[1] - y[0]
    for v in y[1:]:
        prev = level
        level = alpha * v + (1 - alpha) * (level + trend)
        trend = 0.2 * (level - prev) + 0.8 * trend
    cap = float(np.quantile(df["speed"].tail(120), 0.98))
    return [min(max(level + trend * (h + 1), 0), cap) for h in range(horizon)]


def isolation_forest_flags(df, contamination=0.02):
    """Optional ML detector: flags anomalous (speed, volume) points for one road."""
    from sklearn.ensemble import IsolationForest
    if len(df) < 50:
        return pd.Series(False, index=df.index)
    model = IsolationForest(contamination=contamination, random_state=0)
    return pd.Series(model.fit_predict(df[["speed", "volume"]]) == -1, index=df.index)
