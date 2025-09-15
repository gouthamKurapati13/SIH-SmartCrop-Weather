from dataclasses import dataclass
import pandas as pd
from config import BASE_TEMP_BY_CROP

# Minimal rules to start; later replace with ML ranking

@dataclass
class CropOption:
    crop: str
    reason: str
    score: float

def season_from_month(m: int) -> str:
    if m in [6,7,8,9,10]: return "kharif"
    if m in [11,12,1,2,3]: return "rabi"
    return "zaid"

def recommend_crops(daily: pd.DataFrame, soil: dict, state_hint: str | None = None) -> list[CropOption]:
    month = pd.to_datetime(str(daily.iloc[0]["date"])).month
    season = season_from_month(month)
    awc = soil.get("awc_0_30_mm", 30)
    sand = soil.get("sand_pct", 40)

    candidates = []
    if season == "kharif":
        candidates = ["rice","maize","soybean","cotton","pigeonpea","groundnut","sorghum","millet"]
    elif season == "rabi":
        candidates = ["wheat","chickpea","mustard","barley","linseed","peas","onion","garlic"]
    else:
        candidates = ["mungbean","vegetables","fodder","short-duration maize"]

    # Simple scoring: penalize high-sand + low AWC for water-demanding crops
    water_hungry = {"rice": 1.0, "cotton": 0.7, "wheat": 0.6}
    for c in candidates:
        score = 0.6
        if c in water_hungry:
            score -= max(0, (sand - 45)/100) * water_hungry[c]
            score -= max(0, (50 - awc)/100) * water_hungry[c]
        # Boost if forecast has regular rainfall
        rain7 = daily["precipitation_sum"].head(7).sum()
        if rain7 >= 70: score += 0.2
        # Degree days progress (early sowing viability)
        base = BASE_TEMP_BY_CROP.get(c, 10.0)
        tmean = (daily["temperature_2m_max"] + daily["temperature_2m_min"]) / 2.0
        dd7 = (tmean - base).clip(lower=0).head(7).sum()
        if dd7 >= 25: score += 0.1
        score = max(0, min(1, score))
        reason = f"{season.title()} season; AWC≈{awc:.0f} mm (0–30 cm); 7-day rain {rain7:.0f} mm; sand {sand:.0f}%."
        yield CropOption(c, reason, score)
