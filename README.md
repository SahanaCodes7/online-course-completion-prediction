# 🎓 Online Course Completion Prediction

## 📌 Project Overview 
This project predicts whether a student will complete an online course based on their activity and performance data.  
We apply *Machine Learning algorithms* such as Logistic Regression, Random Forest, and Gradient Boosting to build predictive models.

The project is wrapped with a *FastAPI* service so predictions can be consumed via an API.

---

## 🚀 Features
- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Model training with multiple ML algorithms
- Performance evaluation with accuracy, precision, recall, F1-score
- Model saving/loading using joblib
- REST API with *FastAPI* for serving predictions
- Jupyter notebooks for experiments

---

## 🛠 Tech Stack
- *Python 3.10+*
- *Pandas / NumPy*
- *Scikit-learn*
- *FastAPI / Uvicorn*
- *Poetry* for dependency management
- *Jupyter Notebook* for analysis

---

## 📂 Project Structure

. ├── notebooks/                 # Jupyter notebooks │   └── online_course_completion.ipynb ├── app/ │   └── main.py                # FastAPI app ├── models/ │   └── model.pkl              # Saved trained model ├── requirements.txt           # Dependencies ├── README.md                  # Project documentation └── pyproject.toml             # Poetry config

---

## ⚙ Installation

### 1️⃣ Clone the repo
```bash
git clone https://github.com/<your-username>/Online-Course-Completion-ML.git
cd Online-Course-Completion-ML

2️⃣ Install dependencies
Using Poetry:
poetry install
Or using pip:
pip install -r requirements.txt
---
▶ Running the Project
Run Jupyter Notebook
jupyter notebook
Run FastAPI App
uvicorn app.main:app --reload
Then open: http://127.0.0.1:8000/docs
---
📊 Models Used
Logistic Regression
Random Forest Classifier
Gradient Boosting Classifier
---
📈 Results
Gradient Boosting Classifier achieved the highest accuracy 96% in experiments.
---
🧑‍💻 Author
Sahana L
