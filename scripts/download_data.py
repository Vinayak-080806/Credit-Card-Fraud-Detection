"""
Dataset preparation helper.
Place the credit card fraud dataset at:
data/creditcard.csv
"""
from pathlib import Path

dataset = Path(__file__).resolve().parent.parent / "data" / "creditcard.csv"

if dataset.exists():
    print(f"Dataset found: {dataset}")
else:
    print("Dataset not found.")
    print("Place creditcard.csv inside the project's data folder.")
