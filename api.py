from fastapi import FastAPI, Query, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import logging
import cv2 as cv
import numpy as np
import io

from weather import fetch_forecast
from soil import fetch_soil
from alerts import generate_alerts
from recommender import recommend_crops, CropOption
from ph_analysis import analyze_ph_from_image

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Smart Crop Reco (India)")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins - for development
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def root():
    return {"message": "Smart Crop Recommendation API", "endpoints": ["/features/forecast", "/alerts", "/recommendations", "/analyze-ph"]}

@app.get("/health")
def health_check():
    return {"status": "healthy", "features": ["weather", "soil", "recommendations", "ph_analysis"]}

class AlertOut(BaseModel):
    kind: str
    date: str
    message: str
    severity: str

class RecoOut(BaseModel):
    crop: str
    reason: str
    score: float

class PhAnalysisOut(BaseModel):
    median_ph: Optional[float]
    confidence: str
    single_pad_result: Optional[dict]
    multi_pad_result: Optional[dict]
    error: Optional[str] = None

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

@app.post("/analyze-ph", response_model=PhAnalysisOut)
async def analyze_ph(
    file: UploadFile = File(..., description="Image file containing pH strip"),
    gray_roi_x: Optional[int] = Query(default=20, description="Gray card ROI x coordinate"),
    gray_roi_y: Optional[int] = Query(default=20, description="Gray card ROI y coordinate"),
    gray_roi_w: Optional[int] = Query(default=60, description="Gray card ROI width"),
    gray_roi_h: Optional[int] = Query(default=60, description="Gray card ROI height")
):
    """
    Analyze pH from an uploaded image containing pH strips.
    
    The image should contain:
    - A pH strip (single-pad or multi-pad)
    - A gray card for white balance (optional, defaults to top-left corner)
    
    Returns the median pH value from both single-pad and multi-pad analysis methods.
    """
    try:
        logger.info(f"Processing pH analysis for file: {file.filename}")
        
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read the uploaded image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv.imdecode(nparr, cv.IMREAD_COLOR)
        
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        # Set up gray card ROI
        gray_roi = (gray_roi_x, gray_roi_y, gray_roi_w, gray_roi_h)
        
        # Validate gray ROI is within image bounds
        h, w = img_bgr.shape[:2]
        if (gray_roi_x + gray_roi_w > w) or (gray_roi_y + gray_roi_h > h):
            logger.warning(f"Gray ROI {gray_roi} exceeds image bounds {w}x{h}, using defaults")
            gray_roi = (20, 20, min(60, w-20), min(60, h-20))
        
        # Perform pH analysis
        results = analyze_ph_from_image(img_bgr, gray_roi)
        
        logger.info(f"pH analysis completed with median pH: {results.get('median_ph')}")
        
        return PhAnalysisOut(
            median_ph=results.get('median_ph'),
            confidence=results.get('confidence', 'low'),
            single_pad_result=results.get('single_pad'),
            multi_pad_result=results.get('multi_pad'),
            error=results.get('error')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in pH analysis: {e}")
        raise HTTPException(status_code=500, detail=f"pH analysis failed: {str(e)}")
