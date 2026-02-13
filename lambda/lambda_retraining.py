import json
import pandas as pd
import joblib
import boto3
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def lambda_handler(event, context):

    # Load dataset from S3
    s3 = boto3.client("s3")
    bucket_name = "online-course-models"  # change if needed
    data_key = "data/online_course_completion.csv"

    obj = s3.get_object(Bucket=bucket_name, Key=data_key)
    data = pd.read_csv(obj["Body"])

    # Split
    X = data.drop("completed_course", axis=1)
    y = data["completed_course"]

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))

    # Save model temporarily in Lambda
    model_path = "/tmp/retrained_model.pkl"
    joblib.dump(model, model_path)

    # Upload model to S3
    s3.upload_file(model_path, bucket_name, "models/retrained_model.pkl")

    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": "Model retrained and uploaded to S3",
            "accuracy": accuracy
        })
    }
