# train_model.py
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
    def __init__(self, data_path, test_size=0.2, random_state=42):
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state

        # models to try
        self.models = {
            "logistic_regression": LogisticRegression(max_iter=1000),
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "gradient_boosting": GradientBoostingClassifier(random_state=42)
        }

        self.scaler = StandardScaler()
        self.feature_names = []
        self.best_model = None
        self.best_model_name = None
        self.prep = {}  # will store imputation defaults

    def load_and_preprocess(self):
        df = pd.read_csv(self.data_path)
        # copy to avoid SettingWithCopyWarning
        df = df.copy()

        # features used in training (per senior)
        features = [
            'age', 'continent', 'education_level', 'hours_per_week',
            'num_logins_last_month', 'videos_watched_pct',
            'assignments_submitted', 'discussion_posts',
            'is_working_professional', 'preferred_device'
        ]
        X = df[features].copy()
        y = df['completed_course'].copy()

        # --- compute imputation defaults (save for inference) ---
        self.prep['categorical_mode'] = {}
        for cat_col in ['education_level', 'preferred_device', 'continent']:
            if cat_col in X.columns and not X[cat_col].mode().empty:
                self.prep['categorical_mode'][cat_col] = X[cat_col].mode().iloc[0]
            else:
                self.prep['categorical_mode'][cat_col] = ""

        self.prep['numeric_median'] = {}
        if 'videos_watched_pct' in X.columns:
            self.prep['numeric_median']['videos_watched_pct'] = X['videos_watched_pct'].median()
        else:
            self.prep['numeric_median']['videos_watched_pct'] = 0.0

        # --- fill missing using prep defaults ---
        X['education_level'] = X['education_level'].fillna(self.prep['categorical_mode']['education_level'])
        X['videos_watched_pct'] = X['videos_watched_pct'].fillna(self.prep['numeric_median']['videos_watched_pct'])
        X['preferred_device'] = X['preferred_device'].fillna(self.prep['categorical_mode']['preferred_device'])
        X['continent'] = X['continent'].fillna(self.prep['categorical_mode']['continent'])

        # --- feature engineering (based on original notebook) ---
        if 'height_cm' in df.columns and 'weight_kg' in df.columns:
            df['BMI'] = df['weight_kg'] / (df['height_cm'] / 100) ** 2
        else:
            df['BMI'] = 0.0

        # engagement score: sum of certain engagement columns if present
        df['engagement_score'] = 0.0
        for col in ['videos_watched_pct', 'assignments_submitted', 'discussion_posts']:
            if col in df.columns:
                df['engagement_score'] += df[col].fillna(0.0)

        # add engineered features to X
        X['BMI'] = df['BMI']
        X['engagement_score'] = df['engagement_score'].fillna(df['engagement_score'].median())

        # --- encode categorical features ---
        X_encoded = pd.get_dummies(
            X,
            columns=['continent', 'education_level', 'preferred_device'],
            drop_first=True
        )

        # Save feature names (these are the columns we'll expect at inference)
        self.feature_names = X_encoded.columns.tolist()

        # ensure no NaNs (just in case)
        X_encoded = X_encoded.fillna(X_encoded.median())

        # scale
        X_scaled = self.scaler.fit_transform(X_encoded)

        return train_test_split(X_scaled, y, test_size=self.test_size, random_state=self.random_state)

    def train_all(self):
        X_train, X_test, y_train, y_test = self.load_and_preprocess()
        best_acc = -1.0

        for name, model in self.models.items():
            print(f"\n===== Training {name} =====")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            print(f"Accuracy: {acc:.4f}")
            print(classification_report(y_test, y_pred))

            if acc > best_acc:
                best_acc = acc
                self.best_model = model
                self.best_model_name = name

        print(f"\n✅ Best model: {self.best_model_name} (accuracy={best_acc:.4f})")

    def save_best_model(self):
        repo_root = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(repo_root, "models")
        os.makedirs(models_dir, exist_ok=True)

        # save best model, scaler, feature names, and preprocessing defaults
        model_path = os.path.join(models_dir, f"{self.best_model_name}.pkl")
        joblib.dump(self.best_model, model_path)
        joblib.dump(self.scaler, os.path.join(models_dir, "scaler.pkl"))
        joblib.dump(self.feature_names, os.path.join(models_dir, "features.pkl"))
        joblib.dump(self.prep, os.path.join(models_dir, "prep.pkl"))

        print(f"Saved model -> {model_path}")
        print(f"Saved scaler, features and prep -> {models_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and save the best ML model.")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset CSV")
    args = parser.parse_args()

    trainer = TrainModel(args.data)
    trainer.train_all()
    trainer.save_best_model()
