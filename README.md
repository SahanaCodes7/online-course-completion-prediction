
# Online Course Completion Prediction

This project predicts whether a student will complete an online course using Machine Learning.  
It uses real student engagement data (e.g., logins, videos watched, assignments submitted) and applies multiple ML models to select the best performing one.  

The API is built with **FastAPI**, making it easy to serve predictions in real-time.

---

## Features
- Preprocessing with **feature engineering** (BMI, engagement score).  
- Trains **Logistic Regression**, **Random Forest**, and **Gradient Boosting**.  
- Automatically selects and saves the **best model**.  
- **FastAPI backend** with `/predict` endpoint.  
- **Comprehensive unit tests** with pytest (92% code coverage).  
- **AWS S3 integration** for model storage in production.  
- Easy integration with real applications.  

---

## Tech Stack
- Python 3.10+  
- Pandas, NumPy  
- Scikit-learn  
- FastAPI  
- Uvicorn  
- Joblib  
- Boto3 (AWS SDK)  
- Pytest (Testing)  

---

## Project Structure
```
Online-Course-Completion-ML/
│── app/
│   ├── main.py          # FastAPI app (backend)
│   ├── inference.py     # Loads trained model & makes predictions
│   └── train_model.py   # Training script
│── models/              # Stores trained models & scalers (local development)
│── tests/               # Unit tests with pytest
│   ├── test_inference.py
│   ├── test_main.py
│   └── conftest.py
│── d.csv                # Dataset (not uploaded to GitHub)
│── requirements.txt     # Python dependencies
│── Dockerfile           # Docker container configuration
│── pyproject.toml       # Poetry configuration
└── README.md            # Documentation
```

---

## Setup Instructions

1. **Clone the repo**
   ```
   git clone https://github.com/SahanaCodes7/online-course-completion-prediction.git
   cd online-course-completion-prediction
   ```

2. **Create and activate virtual environment**
   ```
   python3 -m venv venv
   source venv/bin/activate        # For Linux/Mac
   venv\Scripts\activate           # For Windows
   ```

3. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

---

## Training the Model

Run the training script with your dataset:

```
python app/train_model.py --data d.csv
```

This will:
- Preprocess the dataset
- Train Logistic Regression, Random Forest, and Gradient Boosting
- Select the best model based on accuracy
- Save the model, scaler, and feature names in the `models/` folder

---

## Testing

### Run Unit Tests

The project includes comprehensive unit tests with **pytest**:

```
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=app --cov-report=html

# Run specific test file
pytest tests/test_inference.py -v

# Run specific test
pytest tests/test_main.py::TestFastAPIEndpoints::test_home_endpoint -v
```

### Test Coverage

- **24 comprehensive unit tests**
- **92% code coverage**
- Tests cover:
  - Model inference logic
  - Data preprocessing
  - API endpoints
  - Error handling
  - S3 integration

View detailed coverage report by opening `htmlcov/index.html` after running coverage tests.

---

## Running the API Locally

Start the FastAPI server with Uvicorn:

```
uvicorn app.main:app --reload
```

Now open in browser:
- Swagger UI → http://127.0.0.1:8000/docs
- ReDoc → http://127.0.0.1:8000/redoc

---

## Example Request

Go to Swagger UI and try `/predict` with the following JSON:

```
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

```

Response:
```
{
  "completed_course": 1,
  "model": "gradient_boosting.pkl"
}
```

(1 → Student is likely to complete the course, 0 → Not likely)

---

## Model Storage - AWS S3 Integration

In production, models are stored in **AWS S3** instead of being bundled in the Docker image. This provides:
- **Smaller Docker images** (faster deployments)
- **Easy model updates** without rebuilding containers
- **Version control** for ML models
- **Cost-effective storage**

### S3 Configuration

- **Bucket**: `online-course-models`
- **Region**: `ap-south-1` (Mumbai)
- **Storage**: ~68 MB (5 model files)
- **Files**:
  - `gradient_boosting.pkl` (140 KB)
  - `random_forest.pkl` (68 MB)
  - `scaler.pkl` (2 KB)
  - `features.pkl` (563 bytes)
  - `prep.pkl` (262 bytes)

### Environment Variables

The application uses environment variables to configure S3 integration:

```
USE_S3=true                          # Enable S3 model loading
S3_BUCKET=online-course-models       # S3 bucket name
S3_PREFIX=models                     # Folder path in S3
```

**Local Development** (USE_S3=false):
- Models are loaded from local `models/` folder

**Production Deployment** (USE_S3=true):
- Models are downloaded from S3 on container startup
- Downloaded once per container lifecycle
- Cached in memory for fast predictions

### IAM Permissions

The ECS task role (`ecsTaskExecutionRole`) has been configured with the following S3 permissions:

```
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::online-course-models",
                "arn:aws:s3:::online-course-models/*"
            ]
        }
    ]
}
```

---

## Deployment on AWS ECS (Fargate)

This project has been containerized using Docker and deployed to **AWS Elastic Container Service (ECS)** using **Fargate** for serverless compute.

### Deployment Steps:

1. **Dockerize the Application**
   - Created a `Dockerfile` to containerize the FastAPI application
   - Built the Docker image locally
   - Tested the container to ensure it runs correctly

2. **Push to Amazon ECR (Elastic Container Registry)**
   ```
   # Authenticate Docker to AWS ECR
   aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin 052104148055.dkr.ecr.ap-south-1.amazonaws.com
   
   # Tag the image
   docker tag online-course-completion:latest 052104148055.dkr.ecr.ap-south-1.amazonaws.com/online-course-completion:latest
   
   # Push to ECR
   docker push 052104148055.dkr.ecr.ap-south-1.amazonaws.com/online-course-completion:latest
   ```

3. **Create S3 Bucket for Models**
   - Created S3 bucket: `online-course-models`
   - Uploaded all model files to `s3://online-course-models/models/`
   - Configured IAM permissions for ECS access

4. **Create ECS Cluster**
   - Created an ECS cluster named `cool-flamingo-08xrac`
   - Selected AWS Fargate as the launch type for serverless deployment

5. **Define Task Definition**
   - Created task definition: `course-completion-task`
   - Configured container with:
     - Image URI from ECR
     - Port mapping: 8000 (container) → 8000 (host)
     - Memory and CPU allocation
     - **Environment variables** for S3 configuration
   - Attached IAM role for S3 access

6. **Create ECS Service**
   - Service name: `course-completion-service`
   - Launch type: **FARGATE**
   - Desired tasks: 1
   - **Auto-assign Public IP: ENABLED** (critical for public access)
   - Configured VPC, subnets, and security groups

7. **Configure Security Group**
   - Opened inbound port **8000** for TCP traffic
   - Source: `0.0.0.0/0` (Anywhere-IPv4) for public access

### Container Startup Process

When the ECS task starts:
1. Container pulls Docker image from ECR
2. Application detects `USE_S3=true`
3. Downloads all model files from S3 (takes ~5-10 seconds)
4. Loads models into memory
5. FastAPI server starts and serves predictions

**Logs confirm S3 integration:**
```
 USE_S3: True
 Downloading models from S3: s3://online-course-models/models/
  Downloading features.pkl... ✓
  Downloading gradient_boosting.pkl... ✓
  Downloading prep.pkl... ✓
  Downloading random_forest.pkl... ✓
  Downloading scaler.pkl... ✓
 Loading model: gradient_boosting.pkl
 Models loaded successfully!
Uvicorn running on http://0.0.0.0:8000
```

### Accessing the Deployed API

Once the ECS service is running:
1. Navigate to ECS → Cluster → Service → Tasks
2. Click on the running task
3. Go to the **Networking** tab
4. Copy the **Public IP** address
5. Access the API at: `http://<PUBLIC_IP>:8000/docs`

**Note:** The public IP changes each time you restart the service.

### Managing the Service

**To Stop the Service (to avoid charges):**
- Update service → Set Desired tasks to `0`

**To Restart the Service:**
- Update service → Set Desired tasks to `1`
- Wait for task to be in "RUNNING" state
- Models will be downloaded from S3 automatically
- Get the new public IP from the task's networking details

---

## Architecture Overview

```
┌─────────────────────────────────────────────┐
│  Developer (VS Code)                        │
│  - Write code                               │
│  - Run tests (pytest)                       │
│  - Build Docker image                       │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  GitHub Repository                          │
│  - Source code                              │
│  - Dockerfile                               │
│  - Unit tests                               │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  AWS ECR (Docker Registry)                  │
│  - Stores Docker images                     │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  AWS ECS (Fargate)                          │
│  - Pulls Docker image from ECR              │
│  - Downloads models from S3 on startup      │
│  - Runs FastAPI container                   │
└──────────────┬──────────────────────────────┘
               │
               ├──────────► AWS S3 Bucket
               │            (Models: 68 MB)
               │
               ▼
        Public API Endpoint
        http://<PUBLIC_IP>:8000
```

---

## Technologies & Skills Demonstrated

### Machine Learning & MLOps
- ✅ Model training with multiple algorithms
- ✅ Model evaluation and selection
- ✅ Feature engineering
- ✅ Model serialization with joblib
- ✅ Model versioning with S3

### Backend Development
- ✅ RESTful API with FastAPI
- ✅ API documentation (Swagger/OpenAPI)
- ✅ Request validation
- ✅ Error handling

### Testing & Quality Assurance
- ✅ Unit testing with pytest
- ✅ Test fixtures and mocking
- ✅ Code coverage reporting (92%)
- ✅ Continuous testing workflow

### Cloud & DevOps
- ✅ Docker containerization
- ✅ AWS ECS deployment with Fargate
- ✅ AWS S3 for model storage
- ✅ AWS ECR for container registry
- ✅ AWS IAM for security and permissions
- ✅ Environment-based configuration

### Best Practices
- ✅ Separation of concerns (models separate from code)
- ✅ Environment variables for configuration
- ✅ Proper error handling
- ✅ Security (private S3 bucket, IAM roles)
- ✅ Cost optimization strategies
- ✅ Comprehensive documentation

---

## Contact

**Sahana L**  
Email: sahanal2024@gmail.com  
GitHub: [@SahanaCodes7](https://github.com/SahanaCodes7)

---
```

Just copy everything above and replace your README.md file! 🎉
