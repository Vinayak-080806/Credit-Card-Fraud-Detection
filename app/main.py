from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.db.database import initialize_database, save_prediction
from app.ml.llm import generate_explanation

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "app" / "models" / "fraud_detection_model.joblib"

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Machine learning API for transaction risk estimation.",
    version="1.0.0"
)

model = None

class TransactionInput(BaseModel):
    features: dict[str, float] = Field(
        ...,
        description="All model features, using the original training column names."
    )

@app.on_event("startup")
def startup_event():
    global model
    initialize_database()

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. Run scripts/train_model.py first."
        )

    model = joblib.load(MODEL_PATH)

@app.get("/")
def home():
    return {"message": "Credit Card Fraud Detection API is running"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict(transaction: TransactionInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    try:
        input_df = pd.DataFrame([transaction.features])
        expected = list(model.feature_names_in_)

        missing = sorted(set(expected) - set(input_df.columns))
        extra = sorted(set(input_df.columns) - set(expected))

        if missing or extra:
            raise HTTPException(
                status_code=400,
                detail={
                    "missing_features": missing,
                    "unexpected_features": extra
                }
            )

        input_df = input_df[expected]
        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])
        amount = float(transaction.features.get("Amount", 0.0))

        save_prediction(amount, prediction, probability)

        return {
            "prediction": prediction,
            "result": "Potential Fraud" if prediction == 1 else "Likely Legitimate",
            "fraud_probability": round(probability, 6),
            "explanation": generate_explanation(
                prediction, probability, amount
            )
        }

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error))
