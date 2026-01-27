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

## 2. Create & activate virtual environment
python -m venv venv




