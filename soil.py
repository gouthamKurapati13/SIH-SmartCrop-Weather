import httpx
import numpy as np
import logging

# ISRIC SoilGrids: returns many layers. We'll compute a simple topsoil available water capacity (AWC) proxy.

logger = logging.getLogger(__name__)

def get_default_soil_values() -> dict:
    """
    Return default soil values when the API is unavailable.
    These are typical values for moderate agricultural soil.
    """
    return {
        "clay_pct": 25.0,  # Medium clay content
        "sand_pct": 40.0,  # Medium sand content
        "silt_pct": 35.0,  # Medium silt content
        "bulk_density": 1.3,  # g/cm3 - typical for agricultural soil
        "soc": 15.0,  # Moderate soil organic carbon
        "awc_mm_per_m": 150.0,  # Available water capacity mm/m
        "awc_0_30_mm": 45.0  # Available water in 0-30cm (150 * 0.30)
    }

def fetch_soil(lat: float, lon: float) -> dict: httpx
import numpy as np

# ISRIC SoilGrids: returns many layers. We’ll compute a simple topsoil available water capacity (AWC) proxy.

def fetch_soil(lat: float, lon: float) -> dict:
    # Depth 0-30 cm layers for clay, sand, silt, bulk density, organic carbon
    base = "https://rest.isric.org/soilgrids/v2.0/properties/query"
    params = {
        "lat": lat, "lon": lon,
        "property": ",".join(["clay","sand","silt","bdod","soc","awch1"]),
        "depth": "0-5cm,5-15cm,15-30cm",
        "value": "mean"
    }
    try:
        r = httpx.get(base, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"Soil API returned error {e.response.status_code} for lat={lat}, lon={lon}")
        # Return default soil values for fallback
        return get_default_soil_values()
    except httpx.RequestError as e:
        logger.error(f"Network error when fetching soil data for lat={lat}, lon={lon}: {e}")
        return get_default_soil_values()
    except Exception as e:
        logger.error(f"Unexpected error when fetching soil data for lat={lat}, lon={lon}: {e}")
        return get_default_soil_values()
    props = {p["name"]: p for p in data["properties"]["layers"]}

    def mean_prop(name):
        vs = []
        for d in props[name]["depths"]:
            vs.append(d["values"]["mean"])
        return float(np.mean(vs))

    # awch1 is available water content (mm/m); take mean of 0-30cm
    out = {
        "clay_pct": mean_prop("clay"),
        "sand_pct": mean_prop("sand"),
        "silt_pct": mean_prop("silt"),
        "bulk_density": mean_prop("bdod")/100,  # bdod in kg/m3 *10? SoilGrids bdod is in cg/cm3; divide by 100 to get g/cm3
        "soc": mean_prop("soc"),
        "awc_mm_per_m": mean_prop("awch1")
    }
    # Approximate plant-available water in 0–30 cm
    out["awc_0_30_mm"] = out["awc_mm_per_m"] * 0.30
    return out
