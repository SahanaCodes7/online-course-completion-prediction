# retraining.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load data (simulated new data)
data = pd.read_csv("data/online_course_completion.csv")

# 2. Separate features and target
X = data.drop("completed_course", axis=1)
y = data["completed_course"]

# 3. Convert categorical columns to numeric
X = pd.get_dummies(X, drop_first=True)

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Retrain model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Retrained model accuracy: {accuracy}")

# 7. Save retrained model
joblib.dump(model, "models/retrained_model.pkl")

print("✅ Retraining completed using simulated new data")

# TODO: Integrate MLflow for experiment tracking
