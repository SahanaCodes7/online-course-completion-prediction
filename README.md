# Online Course Completion Prediction

This project predicts whether a student will complete an online course using **Machine Learning**.  
It leverages student engagement data (logins, videos watched, assignments submitted, etc.) and trains multiple ML models to select the best-performing one.

The system includes:
- A FastAPI backend for real-time predictions
- A retraining pipeline
- AWS integration (S3, ECR, ECS, Lambda container retraining)

This project demonstrates **end-to-end ML + MLOps deployment on AWS**.

---

## 🚀 Features

- Data preprocessing with feature engineering (BMI, engagement score)
- Trains multiple ML models:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
- Automatically selects and saves the best model
- Model retraining pipeline (supports new incoming data)
- FastAPI backend with `/predict` endpoint
- Comprehensive unit tests (~92% coverage)
- AWS S3 integration for model storage
- Dockerized application
- Deployable on AWS ECS (Fargate)
- Serverless retraining using AWS Lambda (container-based)

---

## 🛠 Tech Stack

- Python 3.10+
- Pandas, NumPy
- Scikit-learn
- FastAPI
- Uvicorn
- Joblib
- Boto3 (AWS SDK)
- Pytest
- Docker
- AWS (S3, ECR, ECS – Fargate, Lambda)

---

## 📂 Project Structure

```
online-course-completion-prediction/
│── app/
│   ├── main.py            # FastAPI application
│   ├── inference.py       # Model loading & prediction logic
│   └── train_model.py     # Model training & selection
│
│── models/                # Local trained models (not production)
│── tests/                 # Unit tests (pytest)
│── notebooks/             # Experimentation
│── data/                  # Dataset (not committed)
│
│── retraining.py          # Local retraining script
│── lambda_container/      # Container-based Lambda retraining setup
│── Dockerfile
│── docker-compose.yml
│── requirements.txt
│── pyproject.toml
│── README.md
```

---

## 📊 Dataset

The dataset is **NOT included** in this repository.

Place it locally at:

```
data/online_course_completion.csv
```

This follows industry best practice by avoiding large or sensitive data in GitHub.

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/SahanaCodes7/online-course-completion-prediction.git
cd online-course-completion-prediction
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

**Windows**
```bash
venv\Scripts\activate
```

**Linux / Mac**
```bash
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Model Training

Train and automatically select the best model:

```bash
python app/train_model.py --data data/online_course_completion.csv
```

This will:
- Preprocess data
- Train multiple models
- Select best based on accuracy
- Save model + scaler in `models/`

---

## 🔄 Model Retraining (Local)

```bash
python retraining.py
```

This simulates retraining using updated data.

It:
- Loads dataset
- Encodes categorical features
- Retrains model
- Evaluates accuracy
- Saves updated model

---

## ☁️ Serverless Retraining (AWS Lambda – Container Based)

A container-based AWS Lambda function is configured to:

1. Download dataset from S3
2. Retrain the model
3. Upload updated model back to S3

Execution time ≈ 50 seconds  
Memory used ≈ 700MB  

This demonstrates production-style MLOps retraining automation.

---

## 🚀 Recent Update – Production-Style Serverless Retraining

A fully containerized retraining pipeline was implemented using **AWS Lambda (Container-based deployment)**.

### What Was Achieved

- Dockerized retraining script with full ML dependencies
- Built Linux-compatible image (linux/amd64)
- Pushed image to Amazon ECR
- Deployed container-based Lambda function
- Configured IAM role for secure S3 access
- Lambda downloads dataset from S3
- Retrains model inside serverless environment
- Uploads updated model back to S3
- Optimized:
  - Timeout (3 minutes)
  - Memory (1024 MB)
  - Model complexity for faster execution

### Technical Challenges Solved

- Resolved Lambda zip size limitations (migrated to container image)
- Fixed multi-architecture Docker image compatibility issue
- Handled S3 key mismatch errors
- Debugged IAM permission issues
- Tuned Lambda timeout and memory usage
- Fixed Python indentation and container redeployment issues

This demonstrates hands-on experience in:

- Serverless ML retraining
- Container-based Lambda deployments
- AWS IAM & security configuration
- Real-world MLOps debugging

---

## 🧪 Testing

Run all tests:

```bash
pytest tests/ -v
```

Run with coverage:

```bash
pytest tests/ -v --cov=app --cov-report=html
```

Coverage ≈ 92%

---

## 🌐 Run API Locally

```bash
uvicorn app.main:app --reload
```

Access:
- Swagger → http://127.0.0.1:8000/docs
- ReDoc → http://127.0.0.1:8000/redoc

---

## 🚢 Deployment (AWS ECS – Fargate)

- Dockerized FastAPI app
- Image pushed to Amazon ECR
- Deployed on AWS ECS (Fargate)
- Models pulled from S3 on startup
- Public API served on port 8000

---

## 🏗 Architecture Overview

```
Developer → GitHub → ECR → ECS (Fargate)
                         ↓
                        S3 (Models)
                         ↓
                    Public FastAPI API
                         ↓
                Lambda (Serverless Retraining)
```


## 👩‍💻 Author

**Sahana L**  
GitHub: https://github.com/SahanaCodes7  

