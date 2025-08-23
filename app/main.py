from fastapi import FastAPI
from pydantic import BaseModel
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
async def home():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <title>Online Course Completion Prediction API</title>
      <style>
        body {
          margin: 0;
          font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
          background: #0f0f1a;
          color: #eaeaea;
          display: flex;
          flex-direction: column;
          min-height: 100vh;
        }
        header {
          padding: 2rem;
          text-align: center;
          background: linear-gradient(90deg, #3a0ca3, #7209b7, #4361ee);
          color: white;
        }
        header h1 {
          margin: 0;
          font-size: 2rem;
          font-weight: 700;
        }
        header p {
          margin: 0.5rem 0 0;
          font-size: 1rem;
          opacity: 0.9;
        }
        main {
          flex: 1;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          gap: 2rem;
          padding: 2rem;
        }
        .card {
          background: linear-gradient(145deg, #4361ee, #3a0ca3);
          border-radius: 16px;
          padding: 2rem;
          text-align: center;
          box-shadow: 0 6px 20px rgba(0,0,0,0.5);
          width: 280px;
          transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .card:hover {
          transform: translateY(-6px);
          box-shadow: 0 10px 25px rgba(0,0,0,0.6);
        }
        .card img {
          height: 60px;
          margin-bottom: 1rem;
        }
        .card h3 {
          margin-bottom: 0.5rem;
          color: #fff;
        }
        .card p a {
          color: #ffb703;
          font-weight: 600;
        }
        .callout {
          background: #1a1a2e;
          border: 1px solid #333;
          padding: 1.2rem 1.5rem;
          border-radius: 10px;
          max-width: 600px;
          text-align: center;
          box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        }
        .callout h2 {
          margin: 0 0 0.5rem;
          font-size: 1.2rem;
          font-weight: 600;
          color: #eaeaea;
        }
        .callout code {
          background: #222;
          padding: 0.25rem 0.5rem;
          border-radius: 6px;
          font-size: 0.9rem;
          color: #80b3ff;
        }
        footer {
          text-align: center;
          padding: 1rem;
          font-size: 0.9rem;
          background: #111122;
          color: #999;
        }
      </style>
    </head>
    <body>
      <header>
        <h1>Online Course Completion Prediction API</h1>
        <p>A modern ML-powered API for predicting student course completion</p>
      </header>

      <main>
        <div style="display:flex; gap:2rem; flex-wrap:wrap; justify-content:center;">
          <div class="card">
            <img src="https://via.placeholder.com/60.png?text=API" alt="API Docs Icon" />
            <h3>Interactive API Docs</h3>
            <p><a href="/docs" target="_blank">View Swagger UI</a></p>
          </div>
          <div class="card">
            <img src="https://via.placeholder.com/60.png?text=Doc" alt="ReDoc Icon" />
            <h3>ReDoc Documentation</h3>
            <p><a href="/redoc" target="_blank">View ReDoc</a></p>
          </div>
        </div>

        <!-- Prediction endpoint as a subtle info box -->
        <div class="callout">
          <h2>Prediction Endpoint</h2>
          <p>Send student data with <code>POST /predict</code> to receive predictions.</p>
        </div>
      </main>

      <footer>
        © 2025 Online Course Completion ML API. All rights reserved.
      </footer>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
# ✅ Match schema to training features
class CourseData(BaseModel):
    age: int
    continent: str
    education_level: str
    hours_per_week: float
    num_logins_last_month: int
    videos_watched_pct: float
    assignments_submitted: int
    discussion_posts: int
    is_working_professional: int
    preferred_device: str
 

@app.post("/predict")
def predict_course_completion(data: CourseData):
    # Convert Pydantic model → DataFrame
    df = pd.DataFrame([data.dict()])

    # Run prediction
    prediction = inference.predict(df)

    return {"completed_course": int(prediction[0])}


