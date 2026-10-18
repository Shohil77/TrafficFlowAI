"""Simulated live traffic feed with injectable incidents."""
import numpy as np
import pandas as pd

ROADS = {  # name: (free-flow speed km/h, typical vehicles per interval)
    "Road A": (60, 120), "Road B": (50, 90), "Road C": (80, 150), "Road D": (40, 70),
}


class TrafficSimulator:
    def __init__(self, seed=42, incident_prob=0.01):
        self.rng = np.random.default_rng(seed)
        self.t = pd.Timestamp.now().floor("min") - pd.Timedelta(minutes=120)
        self.incident_prob = incident_prob
        self.incidents = {}  # road -> remaining ticks

    def inject_incident(self, road, duration=12):
        self.incidents[road] = duration

    def step(self):
        """Advance one 1-minute interval; return one row per road."""
        self.t += pd.Timedelta(minutes=1)
        hour = self.t.hour + self.t.minute / 60
        rush = np.exp(-((hour - 8.5) ** 2) / 2) + np.exp(-((hour - 17.5) ** 2) / 2)
        rows = []
        for road, (free, vol) in ROADS.items():
            if road not in self.incidents and self.rng.random() < self.incident_prob:
                self.incidents[road] = int(self.rng.integers(8, 20))
            volume = vol * (0.6 + 0.5 * rush) + self.rng.normal(0, vol * 0.05)
            speed = free * (1 - 0.15 * rush) + self.rng.normal(0, free * 0.03)
            if road in self.incidents:
                speed *= 0.4
                volume *= 1.3
                self.incidents[road] -= 1
                if self.incidents[road] <= 0:
                    del self.incidents[road]
            rows.append({"time": self.t, "road": road,
                         "speed": max(speed, 3), "volume": max(volume, 0)})
        return pd.DataFrame(rows)

    def warmup(self, n=120):
        saved, self.incident_prob = self.incident_prob, 0
        df = pd.concat([self.step() for _ in range(n)], ignore_index=True)
        self.incident_prob = saved
        return df
