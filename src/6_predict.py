
from pathlib import Path
import pandas as pd
import joblib

# Project root directory
ROOT = Path(__file__).resolve().parent.parent

# Load saved model and test data
model = joblib.load(ROOT / "models" / "xgboost_model.pkl")

X_test = pd.read_csv(ROOT / "data" / "processed" / "X_test.csv")
y_test = pd.read_csv(ROOT / "data" / "processed" / "y_test.csv")

# Predict using the first test sample
sample = X_test.iloc[[0]]
prediction = model.predict(sample)[0]
probability = model.predict_proba(sample)[0][1]

print("\n=== LAND ACQUISITION RISK PREDICTION ===")
print("Prediction:", "Delayed / High Risk" if prediction == 1 else "Not Delayed / Low Risk")
print(f"Delay probability: {probability:.2%}")

print("\nActual result:", int(y_test.iloc[0, 0]))
print("Feature count:", sample.shape[1])
print("Model feature count:", model.n_features_in_)
