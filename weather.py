import httpx
import pandas as pd
from datetime import date
import logging

logger = logging.getLogger(__name__)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

def fetch_forecast(lat: float, lon: float, days: int = 14, tz: str = "Asia/Kolkata") -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join([
            "temperature_2m","relative_humidity_2m","precipitation","rain","surface_pressure",
            "wind_speed_10m","wind_gusts_10m","wind_direction_10m","shortwave_radiation"
        ]),
        "daily": ",".join([
            "temperature_2m_max","temperature_2m_min","precipitation_sum",
            "sunrise","sunset","wind_speed_10m_max","wind_gusts_10m_max","shortwave_radiation_sum"
        ]),
        "forecast_days": days,
        "timezone": tz,
    }
    try:
        r = httpx.get(OPEN_METEO_URL, params=params, timeout=30)
        r.raise_for_status()
        js = r.json()
        daily = pd.DataFrame(js["daily"])
        daily["date"] = pd.to_datetime(daily["time"]).dt.date
        daily = daily.drop(columns=["time"])
        return daily
    except httpx.HTTPStatusError as e:
        logger.error(f"Weather API returned error {e.response.status_code} for lat={lat}, lon={lon}")
        raise Exception(f"Weather service unavailable: {e.response.status_code}")
    except httpx.RequestError as e:
        logger.error(f"Network error when fetching weather data for lat={lat}, lon={lon}: {e}")
        raise Exception(f"Failed to connect to weather service: {e}")
    except Exception as e:
        logger.error(f"Unexpected error when fetching weather data for lat={lat}, lon={lon}: {e}")
        raise Exception(f"Weather data processing failed: {e}")

def fetch_reanalysis(lat: float, lon: float, start: str, end: str, tz: str = "Asia/Kolkata") -> pd.DataFrame:
    url = "https://archive-api.open-meteo.com/v1/era5"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join([
            "temperature_2m_max","temperature_2m_min","precipitation_sum",
            "shortwave_radiation_sum","windspeed_10m_max"
        ]),
        "start_date": start, "end_date": end, "timezone": tz
    }
    try:
        r = httpx.get(url, params=params, timeout=60)
        r.raise_for_status()
        js = r.json()
        daily = pd.DataFrame(js["daily"])
        daily["date"] = pd.to_datetime(daily["time"]).dt.date
        daily = daily.drop(columns=["time"])
        return daily
    except httpx.HTTPStatusError as e:
        logger.error(f"Reanalysis API returned error {e.response.status_code} for lat={lat}, lon={lon}")
        raise Exception(f"Reanalysis service unavailable: {e.response.status_code}")
    except httpx.RequestError as e:
        logger.error(f"Network error when fetching reanalysis data for lat={lat}, lon={lon}: {e}")
        raise Exception(f"Failed to connect to reanalysis service: {e}")
    except Exception as e:
        logger.error(f"Unexpected error when fetching reanalysis data for lat={lat}, lon={lon}: {e}")
        raise Exception(f"Reanalysis data processing failed: {e}")
