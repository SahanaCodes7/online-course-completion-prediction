# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from typing import List, Dict
import os
import pandas as pd

from .inference import InferenceModel

# locate models folder and instantiate inference
APP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(APP_DIR)
MODELS_DIR = os.path.join(REPO_ROOT, "models")

# instantiate (auto-picks the first model .pkl it finds)
inference = InferenceModel(models_dir=MODELS_DIR)

app = FastAPI(
    title="Online Course Completion Prediction API",
    description="Predict whether a student will complete an online course.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
      <head>
        <title>Online Course Completion Prediction</title>
      </head>
      <body style="font-family: Arial, sans-serif; padding: 2rem;">
        <h1>Online Course Completion Prediction API</h1>
        <p>Use the interactive docs at <a href="/docs">/docs</a> (Swagger) to test the API.</p>
        <p>Endpoints:</p>
        <ul>
          <li><b>POST /predict</b> - single JSON object -> returns {"completed_course": 0|1}</li>
          <li><b>POST /predict/batch</b> - list of JSON objects -> returns list of predictions</li>
          <li><b>GET /fields</b> - returns feature names expected (helpful for user input)</li>
        </ul>
      </body>
    </html>
    """

@app.get("/fields")
def fields():
    return {"expected_features": inference.get_feature_names(), "model": inference.get_model_name()}

@app.post("/predict")
def predict_course_completion(data: Dict):
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="Input must be a JSON object (single record).")
    df = pd.DataFrame([data])
    preds = inference.predict(df)
    return {"completed_course": preds[0], "model": inference.get_model_name()}

@app.post("/predict/batch")
def predict_batch(data: List[Dict]):
    if not isinstance(data, list):
        raise HTTPException(status_code=400, detail="Input must be a JSON array of objects.")
    df = pd.DataFrame(data)
    preds = inference.predict(df)
    return {"predictions": preds, "model": inference.get_model_name()}
