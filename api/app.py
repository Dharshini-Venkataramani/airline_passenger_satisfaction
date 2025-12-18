# api/app.py
"""
FastAPI service for Airline Passenger Satisfaction prediction.
"""

from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import shared pipeline components so unpickling works
from airline_pipeline import build_preprocessing, make_estimator_for_name

MODEL_PATH = Path("/app/models/global_best_airline_classifier_optuna.pkl")

app = FastAPI(
    title="Airline Passenger Satisfaction API",
    description="Predict passenger satisfaction",
    version="1.0.0",
)

def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)

model = load_model(MODEL_PATH)

UI_TO_MODEL_COLS = {
    "Gender": "gender",
    "Age": "age",
    "Customer Type": "customer_type",
    "Type of Travel": "travel_type",
    "Class": "class",

    "Flight Distance": "flight_distance",
    "Departure Delay in Minutes": "departure_delay",
    "Arrival Delay in Minutes": "arrival_delay",

    "Inflight wifi service": "inflight_wifi",
    "Departure/Arrival time convenient": "time_convenient",
    "Ease of Online booking": "online_booking",
    "Gate location": "gate_location",
    "Food and drink": "food_drink",
    "Online boarding": "online_boarding",
    "Seat comfort": "seat_comfort",
    "Inflight entertainment": "inflight_entertainment",
    "On-board service": "onboard_service",
    "Leg room service": "legroom",
    "Baggage handling": "baggage",
    "Checkin service": "checkin",
    "Inflight service": "inflight_service",
    "Cleanliness": "cleanliness",
}

REQUIRED_MODEL_COLUMNS = list(UI_TO_MODEL_COLS.values())

class PredictRequest(BaseModel):
    instances: List[Dict[str, Any]]

class PredictResponse(BaseModel):
    predictions: List[str]
    count: int

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not request.instances:
        raise HTTPException(status_code=400, detail="No instances provided")

    X_ui = pd.DataFrame(request.instances)
    X = X_ui.rename(columns=UI_TO_MODEL_COLS)

    missing = set(REQUIRED_MODEL_COLUMNS) - set(X.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {sorted(missing)}",
        )

    try:
        preds = model.predict(X)
        preds = ["satisfied" if p == 1 else "dissatisfied" for p in preds]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return PredictResponse(predictions=preds, count=len(preds))
