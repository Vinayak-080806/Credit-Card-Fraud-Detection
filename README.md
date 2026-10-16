# Credit Card Fraud Detection Using Machine Learning

## Project Overview
This project uses machine learning to identify potentially fraudulent credit card transactions. It analyzes transaction data and predicts whether a transaction is legitimate or potentially fraudulent.

## Objectives
- Detect fraudulent credit card transactions.
- Apply machine learning to an imbalanced dataset.
- Evaluate model performance using appropriate metrics.
- Explore the trade-off between precision and recall.
- Present predictions through an application interface.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- SQLite
- Jupyter Notebook

## Dataset
The project uses a credit card transaction dataset containing 284,807 transactions, including 492 fraudulent transactions.

The dataset is highly imbalanced, so accuracy alone is not sufficient for evaluating the model.

The dataset file is not included in this public repository. Obtain it from an authorized source and follow the instructions in the project files.

## Model Evaluation
The recorded model evaluation includes:

- ROC-AUC: approximately 0.9721
- Average Precision: approximately 0.7190
- Fraud detection recall at a 0.5 threshold: approximately 91.8%

These are previously recorded evaluation results. Performance can vary with the training process, dataset split, and decision threshold.

## Visualizations
The project analysis includes:
- Class distribution of legitimate and fraudulent transactions.
- Confusion matrix.
- Precision and recall across decision thresholds.
- Model evaluation metrics.

## Project Structure

```text
Credit-Card-Fraud-Detection/
├── app/
├── frontend/
├── notebooks/
├── scripts/
├── tests/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md


## Installation

Install the required dependencies:

    pip install -r requirements.txt


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

**Vinayak Shinde**  
BSc Data Science Student  
Indira College of Commerce and Science

GitHub: [Vinayak-080806](https://github.com/Vinayak-080806)
