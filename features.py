import math
import pandas as pd

def degree_days(daily: pd.DataFrame, base_temp: float) -> pd.Series:
    tmean = (daily["temperature_2m_max"] + daily["temperature_2m_min"]) / 2.0
    dd = (tmean - base_temp).clip(lower=0)
    return dd.cumsum()

def _ra_extraterrestrial_radiation(lat_deg: float, doy: int) -> float:
    # FAO-56 extraterrestrial radiation Ra (MJ m-2 d-1)
    lat = math.radians(lat_deg)
    dr = 1 + 0.033 * math.cos(2*math.pi/365 * doy)
    delta = 0.409 * math.sin(2*math.pi/365 * doy - 1.39)
    ws = math.acos(-math.tan(lat)*math.tan(delta))
    Gsc = 0.0820  # MJ m-2 min-1
    Ra = (24*60/math.pi) * Gsc * dr * (
        ws*math.sin(lat)*math.sin(delta) + math.cos(lat)*math.cos(delta)*math.sin(ws)
    )
    return Ra  # MJ m-2 d-1

def eto_hargreaves(daily: pd.DataFrame, lat_deg: float) -> pd.Series:
    # Hargreaves-Samani (FAO-56): ET0 = 0.0023*(Tmean+17.8)*(Tmax-Tmin)^0.5*Ra
    out = []
    for _, row in daily.iterrows():
        tmax = row["temperature_2m_max"]
        tmin = row["temperature_2m_min"]
        tmean = (tmax + tmin)/2.0
        doy = pd.to_datetime(str(row["date"])).dayofyear
        Ra = _ra_extraterrestrial_radiation(lat_deg, doy)
        et0 = 0.0023 * (tmean + 17.8) * math.sqrt(max(tmax - tmin, 0)) * Ra
        # Convert MJ/m2 to mm: 1 mm ~ 2.45 MJ/m2 latent heat, but Hargreaves already yields mm/day, keep as-is
        out.append(et0)
    return pd.Series(out, index=daily.index, name="et0_mm")

def rolling_dry_spell(daily: pd.DataFrame, k: int = 7) -> pd.Series:
    dry = (daily["precipitation_sum"] <= 1.0).astype(int)
    return dry.rolling(k).sum()

def humidity_pressure_proxy(hourly_rh_nights_high: int) -> bool:
    return hourly_rh_nights_high >= 2  # ≥2 nights with RH≥85% (computed outside if you ingest hourly)
