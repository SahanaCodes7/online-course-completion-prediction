from fastapi import FastAPI
import pandas as pd
import os
from fastapi.responses import HTMLResponse
from .inference import InferenceModel

# Paths to the saved model files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(os.path.dirname(BASE_DIR), "models")

# Find the model file automatically
model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pkl") and f not in ("scaler.pkl", "features.pkl")]
if not model_files:
    raise FileNotFoundError("No model file found in models directory.")
MODEL_PATH = os.path.join(MODELS_DIR, model_files[0])

SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "features.pkl")

# Load inference model
inference = InferenceModel(MODEL_PATH, SCALER_PATH, FEATURES_PATH)

# Create FastAPI app
app = FastAPI(
    title="Online Course Completion Prediction API",
    description="API for predicting online course completion using ML models.",
    version="1.0.0",
    docs_url="/docs",       # Swagger UI
    redoc_url="/redoc",     # ReDoc
    openapi_url="/openapi.json"
)

@app.get("/", response_class=HTMLResponse)
def home():
    """
    Simple homepage for API.
    """
    return """
    <html>
        <head>
            <title>Course Completion Predictor</title>
        </head>
        <body style="font-family:Arial; margin:40px;">
            <h2>🚀 Online Course Completion Prediction API</h2>
            <p>Use <a href='/docs'>/docs</a> to test the API interactively.</p>
            <p>Available endpoint:</p>
            <ul>
                <li><b>POST /predict</b> → Predict if a student will complete the course.</li>
            </ul>
        </body>
    </html>
    """

@app.post("/predict")
def predict_course_completion(data: dict):
    """
    Predict course completion.
    """
    df = pd.DataFrame([data])
    prediction = inference.predict(df)
    return {"completed_course": int(prediction[0])}

