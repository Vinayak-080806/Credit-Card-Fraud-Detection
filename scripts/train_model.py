from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "creditcard.csv"
MODEL = ROOT / "app" / "models" / "fraud_detection_model.joblib"
METRICS = ROOT / "app" / "models" / "training_metrics.json"

df = pd.read_csv(DATA)

if "Class" not in df.columns:
    raise ValueError("Dataset must contain a 'Class' target column.")

df = df.dropna(subset=["Class"])
X = df.drop(columns=["Class"])
y = df["Class"].astype(int)

if y.nunique() != 2:
    raise ValueError("Expected a binary target with classes 0 and 1.")

X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([float("inf"), float("-inf")], float("nan"))

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))

metrics = {
    "rows": int(len(df)),
    "features": int(X.shape[1]),
    "fraud_cases": int(y.sum()),
    "roc_auc": float(roc_auc_score(y_test, probabilities)),
    "average_precision": float(
        average_precision_score(y_test, probabilities)
    )
}

MODEL.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL)

METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

print("\nTraining metrics:")
print(json.dumps(metrics, indent=2))
print("\nModel saved to:", MODEL)
