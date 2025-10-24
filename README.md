# 🎓 Online Course Completion Prediction

This project predicts whether a student will *complete an online course* using Machine Learning.  
It uses real student engagement data (e.g., logins, videos watched, assignments submitted) and applies multiple ML models to select the *best performing one*.  

The API is built with *FastAPI*, making it easy to serve predictions in real-time.

---

## 🚀 Features
- Preprocessing with *feature engineering* (BMI, engagement score).  
- Trains *Logistic Regression, **Random Forest, and **Gradient Boosting*.  
- Automatically selects and saves the *best model*.  
- *FastAPI backend* with a /predict endpoint.  
- Easy integration with real applications.  

---

## 🛠 Tech Stack
- Python 3.10+  
- Pandas, NumPy  
- Scikit-learn  
- FastAPI  
- Uvicorn  
- Joblib  

---

## 📂 Project Structure
Online-Course-Completion-ML/ │── app/ │   ├── main.py          # FastAPI app (backend) │   ├── inference.py     # Loads trained model & makes predictions │   ├── train_model.py   # Training script │── models/              # Stores trained models & scalers │── d.csv                # Dataset (not uploaded to GitHub) │── requirements.txt     # Python dependencies │── README.md            # Documentation

---

## ⚡ Setup Instructions

1. *Clone the repo*
   ```bash
   git clone https://github.com/SahanaCodes27/online-course-completion-prediction.git
   cd Online-Course-Completion-ML

2. Create and activate virtual environment

python3 -m venv venv

source venv/bin/activate   # For Linux/Mac

venv\Scripts\activate      # For Windows


3. Install dependencies

pip install -r requirements.txt




---

 Training the Model

Run the training script with your dataset:

python app/train_model.py --data d.csv

This will:

Preprocess the dataset

Train Logistic Regression, Random Forest, and Gradient Boosting

Select the best model based on accuracy

Save the model, scaler, and feature names in the models/ folder



---

🌐 Running the API

Start the FastAPI server with Uvicorn:

uvicorn app.main:app --reload

Now open in browser:

Swagger UI → http://127.0.0.1:8000/docs

ReDoc → http://127.0.0.1:8000/redoc



---

🧪 Example Request

Go to Swagger UI and try /predict with the following JSON:

{
  "age": 25,
  "continent": "Asia",
  "education_level": "Bachelors",
  "hours_per_week": 10,
  "num_logins_last_month": 15,
  "videos_watched_pct": 80,
  "assignments_submitted": 5,
  "discussion_posts": 3,
  "is_working_professional": 1,
  "preferred_device": "Laptop",
  "weight_kg": 65,
  "height_cm": 170
}

Response:

{
  "completed_course": 1
}

(1 → Student is likely to complete the course, 0 → Not likely)
