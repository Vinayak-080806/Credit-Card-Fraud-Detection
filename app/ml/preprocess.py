import pandas as pd

def load_dataset(file_path):
    return pd.read_csv(file_path)

def separate_features_and_target(df):
    if "Class" not in df.columns:
        raise ValueError("Dataset must contain a Class column.")
    return df.drop(columns=["Class"]), df["Class"]

def prepare_features(df):
    return df.apply(pd.to_numeric, errors="coerce")
