
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

model = joblib.load(ROOT / "models" / "xgboost_model.pkl")
X_train = pd.read_csv(ROOT / "data" / "processed" / "X_train.csv")

importance = pd.Series(
    model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=True).tail(10)

plt.figure(figsize=(10, 6))
importance.plot(kind="barh")

plt.title("Top 10 Factors Influencing Delay Predictions")
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.tight_layout()

output_dir = ROOT / "outputs"
output_dir.mkdir(exist_ok=True)

plt.savefig(output_dir / "feature_importance.png", dpi=300)
plt.show()

print("Graph saved to outputs/feature_importance.png")
