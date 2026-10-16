from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "app" / "models" / "fraud_detection_model.joblib"

st.title("💳 Credit Card Fraud Detection")
st.write("Machine Learning Based Fraud Prediction System")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

if not MODEL_PATH.exists():
    st.error("Model file not found. Please train the model first.")
    st.stop()

model = load_model()
st.success("Machine learning model loaded successfully!")

st.subheader("Transaction Details")
st.warning(
    "The default feature values are for demonstration only. "
    "Use realistic transaction features for meaningful results."
)

features = list(model.feature_names_in_)
transaction = {}

for feature in features:
    if feature == "Amount":
        transaction[feature] = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=100.0,
            step=10.0
        )
    else:
        transaction[feature] = st.number_input(
            feature,
            value=0.0,
            step=0.1,
            format="%.6f"
        )

if st.button("Check Transaction", type="primary"):
    try:
        row = pd.DataFrame([transaction], columns=features)
        prediction = int(model.predict(row)[0])
        probability = float(model.predict_proba(row)[0][1])

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("⚠️ Potential Fraud")
        else:
            st.success("✅ Likely Legitimate")

        st.metric("Estimated Fraud Probability", f"{probability:.2%}")

        st.info(
            "This result is a model estimate, not proof that a transaction "
            "is fraudulent or legitimate."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
