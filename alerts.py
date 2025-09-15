from dataclasses import dataclass
import pandas as pd

@dataclass
class Alert:
    kind: str
    date: str
    message: str
    severity: str

def generate_alerts(daily: pd.DataFrame, lat: float, lon: float, crop: str | None = None) -> list[Alert]:
    alerts = []

    # Heavy rain
    heavy = daily[daily["precipitation_sum"] >= 50]
    for _, r in heavy.iterrows():
        alerts.append(Alert(
            kind="heavy_rain",
            date=str(r["date"]),
            message=f"Heavy rain {r['precipitation_sum']:.0f} mm expected. Protect seedlings, delay spraying.",
            severity="high"
        ))

    # Dry spell 7-day
    dry7 = (daily["precipitation_sum"] <= 1.0).astype(int).rolling(7).sum()
    idx = dry7[dry7 == 7].index
    for i in idx:
        d = daily.loc[i, "date"]
        alerts.append(Alert(
            kind="dry_spell",
            date=str(d),
            message="7-day dry spell detected. Schedule irrigation if possible.",
            severity="medium"
        ))

    # Heat stress (generic; tune per crop)
    heat = daily[daily["temperature_2m_max"] >= 38]
    for _, r in heat.iterrows():
        alerts.append(Alert(
            kind="heat_stress",
            date=str(r["date"]),
            message=f"Heat stress day (Tmax {r['temperature_2m_max']:.1f}°C). Mulch/irrigate to reduce canopy stress.",
            severity="medium"
        ))
    return alerts
