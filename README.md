# Online Course Completion Prediction

This project predicts whether a student will *complete an online course* using Machine Learning.  
It uses real student engagement data (e.g., logins, videos watched, assignments submitted) and applies multiple ML models to select the *best performing one*.  

The API is built with *FastAPI*, making it easy to serve predictions in real-time.

---

## Features
- Preprocessing with *feature engineering* (BMI, engagement score).  
- Trains *Logistic Regression, **Random Forest, and **Gradient Boosting*.  
- Automatically selects and saves the *best model*.  
- *FastAPI backend* with a /predict endpoint.  
- Easy integration with real applications.  

---

## Tech Stack
- Python 3.10+  
- Pandas, NumPy  
- Scikit-learn  
- FastAPI  
- Uvicorn  
- Joblib  

---

## Project Structure
Online-Course-Completion-ML/ │── app/ │   ├── main.py          # FastAPI app (backend) │   ├── inference.py     # Loads trained model & makes predictions │   ├── train_model.py   # Training script │── models/              # Stores trained models & scalers │── d.csv                # Dataset (not uploaded to GitHub) │── requirements.txt     # Python dependencies │── README.md            # Documentation

---

## Setup Instructions

1. *Clone the repo*
   ```bash
   git clone https://github.com/SahanaCodes27/online-course-completion-prediction.git
   cd Online-Course-Completion-ML

2. Create and activate virtual environment

python3 -m venv venv

source venv/bin/activate        # For Linux/Mac

venv\Scripts\activate           # For Windows

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

Running the API

Start the FastAPI server with Uvicorn:

uvicorn app.main:app --reload

Now open in browser:

Swagger UI → http://127.0.0.1:8000/docs

ReDoc → http://127.0.0.1:8000/redoc

---

Example Request

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


---

## Deployment on AWS ECS (Fargate)

This project has been containerized using Docker and deployed to **AWS Elastic Container Service (ECS)** using **Fargate** for serverless compute.

### Deployment Steps:

1. **Dockerize the Application**
   - Created a `Dockerfile` to containerize the FastAPI application
   - Built the Docker image locally
   - Tested the container to ensure it runs correctly

2. **Push to Amazon ECR (Elastic Container Registry)**
   ```bash
   # Authenticate Docker to AWS ECR
   aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin 052104148055.dkr.ecr.ap-south-1.amazonaws.com
   
   # Tag the image
   docker tag online-course-completion:latest 052104148055.dkr.ecr.ap-south-1.amazonaws.com/online-course-completion:latest
   
   # Push to ECR
   docker push 052104148055.dkr.ecr.ap-south-1.amazonaws.com/online-course-completion:latest
   ```

3. **Create ECS Cluster**
   - Created an ECS cluster named `cool-flamingo-08xrac`
   - Selected AWS Fargate as the launch type for serverless deployment

4. **Define Task Definition**
   - Created task definition: `course-completion-task`
   - Configured container with:
     - Image URI from ECR
     - Port mapping: 8000 (container) → 8000 (host)
     - Memory and CPU allocation

5. **Create ECS Service**
   - Service name: `course-completion-service`
   - Launch type: **FARGATE**
   - Desired tasks: 1
   - **Auto-assign Public IP: ENABLED** (critical for public access)
   - Configured VPC, subnets, and security groups

6. **Configure Security Group**
   - Opened inbound port **8000** for TCP traffic
   - Source: `0.0.0.0/0` (Anywhere-IPv4) for public access

### Accessing the Deployed API:

Once the ECS service is running:
1. Navigate to ECS → Cluster → Service → Tasks
2. Click on the running task
3. Go to the **Networking** tab
4. Copy the **Public IP** address
5. Access the API at: `http://<PUBLIC_IP>:8000/docs`

**Note:** The public IP changes each time you restart the service.

### Managing the Service:

**To Stop the Service (to avoid charges):**
- Update service → Set Desired tasks to `0`

**To Restart the Service:**
- Update service → Set Desired tasks to `1`
- Wait for task to be in "RUNNING" state
- Get the new public IP from the task's networking details

### Cost Optimization:
- Using **AWS Free Tier** eligible services
- Stop the service when not in use (set desired tasks to 0)
- Minimal ECR storage costs for the Docker image

---
