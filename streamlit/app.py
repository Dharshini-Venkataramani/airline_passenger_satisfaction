import json
import os
from pathlib import Path
from typing import Any, Dict

import requests
import streamlit as st

# -----------------------------------------------------------------------------
# MUST be the first Streamlit command
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Airline Customer Satisfaction Prediction", page_icon="✈️")

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
SCHEMA_PATH = Path("/app/data/data_schema.json")
API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")
PREDICT_ENDPOINT = f"{API_BASE_URL}/predict"

# -----------------------------------------------------------------------------
# Load schema from JSON file
# -----------------------------------------------------------------------------
@st.cache_resource
@st.cache_resource
def load_schema(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)

schema = load_schema(SCHEMA_PATH)

numerical = schema["numerical"]
categorical = schema["categorical"]
ordinal_names = ["Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking", "Gate location",
           "Food and drink", "Online boarding", "Seat comfort", "Inflight entertainment", "On-board service",
           "Leg room service", "Baggage handling", "Checkin service", "Inflight service", "Cleanliness"]

# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
st.title("✈️ Airline Passenger Satisfaction Prediction")
st.write(
    f"This app sends your inputs to the FastAPI backend at **{API_BASE_URL}** for prediction."
)

st.header("Input Features")

user_input: Dict[str, Any] = {}

st.header("Categorical Features")
for name, info in categorical.items():
    user_input[name] = st.selectbox(
        name,
        info["unique_values"],
    )

st.header("Numerical Features")
for name, stats in numerical.items():
    if name not in ordinal_names:
        user_input[name] = st.number_input(
            name,
            min_value=int(stats["min"]),
            max_value=int(stats["max"]),
            value=int(stats["median"]),
            step=1,)

st.header("Service Satisfaction rating")
for name,stats in numerical.items():
    if name in ordinal_names:
        user_input[name] = st.slider(
            name,
            min_value=int(stats["min"]),
            max_value=int(stats["max"]),
            value=int(stats["median"]),
            step=1,)


# -----------------------------------------------------------------------------
# Predict
# -----------------------------------------------------------------------------
if st.button("🔮 Predict"):
    payload = {"instances": [user_input]}

    resp = requests.post(PREDICT_ENDPOINT, json=payload)

    if resp.status_code != 200:
        st.error(resp.text)
    else:
        pred = resp.json()["predictions"][0]
        st.success(f"Prediction: **{pred.upper()}**")
