# Online Course Completion Prediction

This project predicts whether a student will complete an online course using **Machine Learning**.  
It leverages student engagement data (logins, videos watched, assignments submitted, etc.) and trains multiple ML models to select the best-performing one.

The system also includes a **FastAPI backend** for real-time predictions and supports **model retraining**, making it suitable for production and MLOps workflows.

---

##  Features
- Data preprocessing with **feature engineering** (BMI, engagement score)
- Trains multiple models:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
- Automatically selects and saves the **best model**
- **Model retraining script** to handle new incoming data
- **FastAPI backend** with `/predict` endpoint
- **Comprehensive unit tests** using pytest (~92% coverage)
- **AWS S3 integration** for model storage (production)
- Dockerized and deployable on **AWS ECS (Fargate)**

---

##  Tech Stack
- Python 3.10+
- Pandas, NumPy
- Scikit-learn
- FastAPI
- Uvicorn
- Joblib
- Boto3 (AWS SDK)
- Pytest
- Docker
- AWS (S3, ECR, ECS – Fargate)

---

##  Project Structure

online-course-completion-prediction/
│── app/
│ ├── main.py # FastAPI application
│ ├── inference.py # Loads trained model & runs predictions
│ └── train_model.py # Training script (model selection)
│
│── models/ # Trained models (local dev only)
│── tests/ # Pytest unit tests
│── notebooks/ # Experiments & exploration
│── data/ # Dataset directory (not committed)
│
│── retraining.py # Model retraining script
│── Dockerfile
│── docker-compose.yml
│── requirements.txt
│── pyproject.toml
│── README.md


---

##  Dataset
- The dataset is **NOT included** in the repository.
- Place the dataset locally at:

data/online_course_completion.csv


This design avoids pushing large or sensitive data to GitHub and follows industry best practices.

---

##  Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/SahanaCodes7/online-course-completion-prediction.git
cd online-course-completion-prediction

2. Create & activate virtual environment
python -m venv venv

Windows
venv\Scripts\activate

Linux / Mac
source venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

Model Training

Train models and automatically select the best one:

python app/train_model.py --data data/online_course_completion.csv


This script:

Preprocesses the data

Trains Logistic Regression, Random Forest & Gradient Boosting

Selects the best model based on accuracy

Saves the model, scaler, and feature metadata in models/

Model Retraining (New Data Support)

A retraining pipeline is included to simulate retraining on new incoming data.

python retraining.py


What it does:

Loads the existing dataset (simulating new data)

Handles categorical features using one-hot encoding

Retrains the model

Evaluates accuracy

Saves the updated model in models/

🔹 This script is designed so real updated datasets can be plugged in later without changing the pipeline.

Testing

Run all tests:

pytest tests/ -v


Run with coverage:

pytest tests/ -v --cov=app --cov-report=html


~24 unit tests

~92% coverage

Covers:

API endpoints

Inference logic

Error handling

S3 integration (mocked)

Running the API Locally
uvicorn app.main:app --reload


Access:

Swagger UI → http://127.0.0.1:8000/docs

ReDoc → http://127.0.0.1:8000/redoc

Example Prediction Request
{
  "age": 25,
  "continent": "Asia",
  "education_level": "Bachelors",
  "hours_per_week": 10,
  "num_logins_last_month": 15,
  "videos_watched_pct": 80,
  "assignments_submitted": 5,
  "discussion_posts": 3,
  "is_working_professional": 1
}


Response:

{
  "completed_course": 1,
  "model": "gradient_boosting.pkl"
}

Model Storage – AWS S3

In production:

Models are stored in AWS S3

Containers download models on startup

Keeps Docker images small and flexible

Environment variables:

USE_S3=true
S3_BUCKET=online-course-models
S3_PREFIX=models

Deployment (AWS ECS – Fargate)

Dockerized FastAPI app

Image pushed to Amazon ECR

Deployed on AWS ECS (Fargate)

Models pulled from S3 at container startup

Public API served on port 8000

Architecture Overview
Developer → GitHub → ECR → ECS (Fargate)
                         ↓
                        S3 (Models)
                         ↓
                    Public FastAPI API

