import pandas as pd
import joblib
import os

class InferenceModel:
    def __init__(self, model_path, scaler_path, features_path):
        # Load trained model, scaler, and feature names
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_names = joblib.load(features_path)

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the same preprocessing steps as in training.
        """
        # Fill missing values
        if 'education_level' in data.columns:
            data['education_level'] = data['education_level'].fillna(data['education_level'].mode()[0])
        if 'videos_watched_pct' in data.columns:
            data['videos_watched_pct'] = data['videos_watched_pct'].fillna(data['videos_watched_pct'].median())
        if 'preferred_device' in data.columns:
            data['preferred_device'] = data['preferred_device'].fillna(data['preferred_device'].mode()[0])

        # Feature engineering
        if 'weight_kg' in data.columns and 'height_cm' in data.columns:
            data['BMI'] = data['weight_kg'] / (data['height_cm'] / 100) ** 2
        if {'videos_watched_pct', 'assignments_submitted', 'discussion_posts'}.issubset(data.columns):
            data['engagement_score'] = (
                data['videos_watched_pct'] +
                data['assignments_submitted'] +
                data['discussion_posts']
            )

        # One-hot encode
        data = pd.get_dummies(
            data,
            columns=['continent', 'education_level', 'preferred_device'],
            drop_first=True
        )

        # Align columns with training features
        data = data.reindex(columns=self.feature_names, fill_value=0)

        # Scale
        data_scaled = self.scaler.transform(data)

        return data_scaled

    def predict(self, data: pd.DataFrame):
        """
        Predict course completion (1/0).
        """
        processed_data = self.preprocess(data)
        predictions = self.model.predict(processed_data)
        return predictions.tolist()

if __name__ == "__main__":
    # Example usage
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "models", "gradient_boosting.pkl")
    scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")
    features_path = os.path.join(BASE_DIR, "models", "features.pkl")

    inference = InferenceModel(model_path, scaler_path, features_path)

    # Example single prediction
    sample_data = pd.DataFrame([{
        'age': 25,
        'continent': 'Asia',
        'education_level': 'Bachelors',
        'hours_per_week': 10,
        'num_logins_last_month': 15,
        'videos_watched_pct': 80,
        'assignments_submitted': 5,
        'discussion_posts': 3,
        'is_working_professional': 1,
        'preferred_device': 'Laptop',
        'weight_kg': 65,
        'height_cm': 170
    }])

    print(inference.predict(sample_data))
