from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import logging

from weather import fetch_forecast
from soil import fetch_soil
from alerts import generate_alerts
from recommender import recommend_crops, CropOption

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Smart Crop Reco (India)")

class AlertOut(BaseModel):
    kind: str
    date: str
    message: str
    severity: str

class RecoOut(BaseModel):
    crop: str
    reason: str
    score: float

@app.get("/features/forecast")
def features(lat: float, lon: float, days: int = 14):
    try:
        daily = fetch_forecast(lat, lon, days=days)
        return daily.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Error fetching forecast for lat={lat}, lon={lon}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch weather data: {str(e)}")

@app.get("/alerts", response_model=List[AlertOut])
def alerts(lat: float, lon: float, crop: Optional[str] = None):
    try:
        daily = fetch_forecast(lat, lon, days=14)
        res = generate_alerts(daily, lat, lon, crop)
        return [a.__dict__ for a in res]
    except Exception as e:
        logger.error(f"Error generating alerts for lat={lat}, lon={lon}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate alerts: {str(e)}")

@app.get("/recommendations", response_model=List[RecoOut])
def recommendations(lat: float, lon: float, state: Optional[str] = None):
    try:
        logger.info(f"Fetching recommendations for lat={lat}, lon={lon}, state={state}")
        
        # Fetch weather data
        daily = fetch_forecast(lat, lon, days=14)
        
        # Fetch soil data (with fallback to defaults if API fails)
        soil = fetch_soil(lat, lon)
        
        # Generate recommendations
        options = list(recommend_crops(daily, soil, state_hint=state))
        options.sort(key=lambda x: x.score, reverse=True)
        
        logger.info(f"Successfully generated {len(options)} recommendations")
        return [RecoOut(crop=o.crop, reason=o.reason, score=o.score) for o in options]
        
    except Exception as e:
        logger.error(f"Error generating recommendations for lat={lat}, lon={lon}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {str(e)}")
