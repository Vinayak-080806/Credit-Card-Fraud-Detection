# Credit Card Fraud Detection

A machine-learning project for detecting potentially fraudulent
credit-card transactions using Logistic Regression.

## Features

- Logistic Regression model
- Class-imbalance handling
- Model evaluation metrics
- FastAPI prediction API
- Streamlit web interface
- SQLite prediction history

## Project Structure

- app/: API, database, and machine-learning modules
- data/: Local dataset
- frontend/: Streamlit application
- notebooks/: Training notebook
- scripts/: Model training and dataset helper scripts
- tests/: Test files

## Requirements

- Python
- pip
- Credit-card fraud dataset

## Installation

Install the required dependencies:

    pip install -r requirements.txt

## Dataset

Place the dataset at:

    data/creditcard.csv

Obtain the dataset from an authorised source and check its licence.
The dataset is not included in this repository.

## Train the Model

Run this command from the project root:

    python scripts/train_model.py

## Start the API

Run:

    uvicorn app.main:app --reload

## Start the Frontend

Open another terminal and run:

    streamlit run frontend/streamlit_app.py

## Evaluation

Evaluate the model using precision, recall, F1-score, ROC-AUC,
average precision, and a confusion matrix.

Accuracy alone is insufficient for an imbalanced fraud dataset.

## Limitations

This is an educational project, not a production banking security
system. Predictions may be incorrect.

## Privacy

Do not publish confidential information, credentials, or private
transaction data on GitHub.

## Author

Student Data Science Project
