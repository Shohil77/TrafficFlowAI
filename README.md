# TrafficFlow AI
Dashboard that monitors traffic data, detects unusual congestion and explains alerts.

## Run
    pip install -r requirements.txt
    streamlit run app.py

## How it works
- `trafficflow/simulator.py` – simulated feed (rush-hour pattern, noise, random/injected incidents)
- `trafficflow/detector.py` – rolling-baseline z-score detection with persistence, evidence-based alert text,
  Holt exponential-smoothing forecast, optional Isolation Forest
- `app.py` – Streamlit + Plotly live dashboard (refreshes every 2s)

To use real data, replace `TrafficSimulator.step()` with a function returning rows of `time, road, speed, volume`.
