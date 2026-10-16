def explain_prediction(transaction, prediction, probability):
    return {
        "prediction": int(prediction),
        "fraud_probability": float(probability),
        "features_received": list(transaction.keys())
    }
