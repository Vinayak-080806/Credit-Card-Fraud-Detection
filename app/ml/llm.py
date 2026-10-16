def generate_explanation(prediction, probability, amount):
    if int(prediction) == 1:
        result = "The model flagged this transaction as potentially fraudulent."
    else:
        result = "The model classified this transaction as likely legitimate."

    return (
        f"{result} Transaction amount: {float(amount):.2f}. "
        f"Estimated fraud probability: {float(probability):.2%}. "
        "This is a model estimate, not a definitive fraud determination."
    )
