
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Load data
# --------------------------------------------------
X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")

y_train = pd.read_csv(DATA_DIR / "y_train.csv").values.ravel()
y_test = pd.read_csv(DATA_DIR / "y_test.csv").values.ravel()

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# --------------------------------------------------
# 3. Validate data
# --------------------------------------------------
if list(X_train.columns) != list(X_test.columns):
    raise ValueError("Train and test columns do not match.")

if len(X_train) != len(y_train):
    raise ValueError("Training features and labels do not match.")

if len(X_test) != len(y_test):
    raise ValueError("Testing features and labels do not match.")

if not np.isfinite(X_train.select_dtypes(include="number")).all().all():
    raise ValueError("Training data contains NaN or infinity.")

if not np.isfinite(X_test.select_dtypes(include="number")).all().all():
    raise ValueError("Testing data contains NaN or infinity.")

if set(np.unique(y_train)) != {0, 1}:
    raise ValueError("Training target must contain 0 and 1.")

# --------------------------------------------------
# 4. Class imbalance
# --------------------------------------------------
negative_count = np.sum(y_train == 0)
positive_count = np.sum(y_train == 1)

scale_pos_weight = negative_count / positive_count

print("\nClass 0:", negative_count)
print("Class 1:", positive_count)
print("Positive class weight:", round(scale_pos_weight, 3))

# --------------------------------------------------
# 5. Cross-validation
# --------------------------------------------------
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# --------------------------------------------------
# 6. Hyperparameter tuning
# --------------------------------------------------
base_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=1
)

param_distributions = {
    "n_estimators": [150, 250, 350, 450],
    "max_depth": [2, 3, 4, 5],
    "learning_rate": [0.03, 0.05, 0.08, 0.1],
    "min_child_weight": [1, 3, 5],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "scale_pos_weight": [1, scale_pos_weight]
}

search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_distributions,
    n_iter=20,
    scoring="f1",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=1,
    refit=True
)

print("\nStarting hyperparameter tuning...")
search.fit(X_train, y_train)

# --------------------------------------------------
# 7. Best model
# --------------------------------------------------
best_model = search.best_estimator_

print("\n=== BEST CONFIGURATION ===")
print(search.best_params_)
print("Best CV F1:", round(search.best_score_, 3))

# --------------------------------------------------
# 8. Final evaluation on untouched test set
# --------------------------------------------------
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print("\n=== TUNED XGBOOST TEST RESULTS ===")
print("Accuracy :", round(accuracy_score(y_test, y_pred), 3))
print("Precision:", round(precision_score(y_test, y_pred, zero_division=0), 3))
print("Recall   :", round(recall_score(y_test, y_pred, zero_division=0), 3))
print("F1 Score :", round(f1_score(y_test, y_pred, zero_division=0), 3))
print("ROC-AUC  :", round(roc_auc_score(y_test, y_proba), 3))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# --------------------------------------------------
# 9. Save tuned model
# --------------------------------------------------
model_path = MODEL_DIR / "xgboost_model.pkl"
joblib.dump(best_model, model_path)

print("\nTuned model saved at:")
print(model_path)

# --------------------------------------------------
# 10. Feature importance
# --------------------------------------------------
importances = pd.Series(
    best_model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

print("\nTop 10 Feature Importances:")
print(importances.head(10).round(4))
