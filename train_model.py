import argparse
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

class TrainModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.models = {
            "logistic_regression": LogisticRegression(max_iter=1000),
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "gradient_boosting": GradientBoostingClassifier(random_state=42)
        }
        self.scaler = StandardScaler()
        self.feature_names = []
        self.best_model_name = None
        self.best_model = None

    def load_and_preprocess(self):
        df = pd.read_csv(self.data_path)

        # Features & Target
        features = [
            'age', 'continent', 'education_level', 'hours_per_week',
            'num_logins_last_month', 'videos_watched_pct',
            'assignments_submitted', 'discussion_posts',
            'is_working_professional', 'preferred_device'
        ]
        X = df[features]
        y = df['completed_course']

        # Missing values
        X['education_level'] = X['education_level'].fillna(X['education_level'].mode()[0])
        X['videos_watched_pct'] = X['videos_watched_pct'].fillna(X['videos_watched_pct'].median())
        X['preferred_device'] = X['preferred_device'].fillna(X['preferred_device'].mode()[0])

        # One-hot encoding
        X = pd.get_dummies(X, columns=['continent', 'education_level', 'preferred_device'], drop_first=True)

        # Feature engineering
        df['BMI'] = df['weight_kg'] / (df['height_cm'] / 100) ** 2
        df['engagement_score'] = (
            df['videos_watched_pct'] +
            df['assignments_submitted'] +
            df['discussion_posts']
        )
        X['BMI'] = df['BMI']
        X['engagement_score'] = df['engagement_score'].fillna(df['engagement_score'].median())

        # Save feature names for inference
        self.feature_names = X.columns.tolist()

        # Scaling
        X_scaled = self.scaler.fit_transform(X)

        return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    def train_all(self):
        X_train, X_test, y_train, y_test = self.load_and_preprocess()
        best_accuracy = 0.0

        for name, model in self.models.items():
            print(f"\n===== Training {name} =====")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            print(f"Accuracy: {acc:.4f}")
            print(classification_report(y_test, y_pred))

            if acc > best_accuracy:
                best_accuracy = acc
                self.best_model_name = name
                self.best_model = model

        print(f"\n✅ Best Model: {self.best_model_name} with Accuracy: {best_accuracy:.4f}")

    def save_best_model(self):
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        os.makedirs(models_dir, exist_ok=True)

        # Save model
        model_path = os.path.join(models_dir, f"{self.best_model_name}.pkl")
        joblib.dump(self.best_model, model_path)

        # Save scaler & features
        joblib.dump(self.scaler, os.path.join(models_dir, "scaler.pkl"))
        joblib.dump(self.feature_names, os.path.join(models_dir, "features.pkl"))

        print(f"💾 Saved best model to: {model_path}")
        print(f"💾 Saved scaler and features for inference.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and save the best ML model.")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset CSV")
    args = parser.parse_args()

    trainer = TrainModel(args.data)
    trainer.train_all()
    trainer.save_best_model()
