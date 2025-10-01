# app/inference.py
import os
import joblib
import pandas as pd
import numpy as np
from typing import Union, List, Dict

class InferenceModel:
    def __init__(self, models_dir: str = None, model_name: str = None):
        # default models_dir = repo_root/models
        if models_dir is None:
            app_dir = os.path.dirname(os.path.abspath(_file_))
            repo_root = os.path.dirname(app_dir)
            models_dir = os.path.join(repo_root, "models")
        self.models_dir = models_dir

        # choose model file automatically if not provided
        if model_name:
            model_file = model_name
        else:
            # pick first .pkl that isn't scaler/features/prep
            candidates = [f for f in os.listdir(self.models_dir) if f.endswith(".pkl")]
            candidates = [f for f in candidates if f not in ("scaler.pkl", "features.pkl", "prep.pkl")]
            if not candidates:
                raise FileNotFoundError("No model .pkl found in models directory.")
            model_file = candidates[0]

        self.model_path = os.path.join(self.models_dir, model_file)
        self.scaler_path = os.path.join(self.models_dir, "scaler.pkl")
        self.features_path = os.path.join(self.models_dir, "features.pkl")
        self.prep_path = os.path.join(self.models_dir, "prep.pkl")

        # load
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        self.feature_names = joblib.load(self.features_path)
        self.prep = joblib.load(self.prep_path)

    def _ensure_dataframe(self, data: Union[Dict, List[Dict], pd.DataFrame]) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            raise ValueError("Input must be a dict, list of dicts, or DataFrame.")
        return df

    def preprocess(self, raw: Union[Dict, List[Dict], pd.DataFrame]) -> np.ndarray:
        df = self._ensure_dataframe(raw).copy()

        # fill categorical defaults (if category column missing we create it with default)
        for col, default in self.prep.get('categorical_mode', {}).items():
            if col in df.columns:
                df[col] = df[col].fillna(default)
            else:
                # create column with default
                df[col] = default

        # fill numeric defaults
        for col, default in self.prep.get('numeric_median', {}).items():
            if col in df.columns:
                df[col] = df[col].fillna(default)
            else:
                df[col] = default

        # compute BMI if height and weight are present (if missing, create 0)
        if 'height_cm' in df.columns and 'weight_kg' in df.columns:
            df['BMI'] = df['weight_kg'] / (df['height_cm'] / 100).replace(0, np.nan) ** 2
            df['BMI'] = df['BMI'].fillna(0.0)
        else:
            df['BMI'] = 0.0

        # engagement_score if components present
        if {'videos_watched_pct', 'assignments_submitted', 'discussion_posts'}.issubset(df.columns):
            df['engagement_score'] = (
                df['videos_watched_pct'].fillna(0) +
                df['assignments_submitted'].fillna(0) +
                df['discussion_posts'].fillna(0)
            )
        else:
            df['engagement_score'] = 0.0

        # One-hot encode same categorical columns used in training
        df_encoded = pd.get_dummies(
            df,
            columns=['continent', 'education_level', 'preferred_device'],
            drop_first=True
        )

        # Align to training feature set
        df_aligned = df_encoded.reindex(columns=self.feature_names, fill_value=0)

        # scale
        X_scaled = self.scaler.transform(df_aligned)

        return X_scaled

    def predict(self, raw_input: Union[Dict, List[Dict], pd.DataFrame]) -> List[int]:
        X = self.preprocess(raw_input)
        preds = self.model.predict(X)
        # convert to python ints for jsonability
        return [int(x) for x in preds]

    def get_feature_names(self) -> List[str]:
        return list(self.feature_names)

    def get_model_name(self) -> str:
        return os.path.basename(self.model_path)
