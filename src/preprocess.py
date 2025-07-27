from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Config to show all columns
pd.set_option('display.max_columns', None)

# Load csv from the given filepath
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

# Removal of data values with values of zero
def remove_zeroes(df : pd.DataFrame) -> pd.DataFrame:
    # Features in diabetes.csv
    zero_features = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    # iterates through each feature replaces a 0 with the median of that feature column
    for col in zero_features:
        df[col] = df[col].replace(0, df[col].median())

    return df

def create_train_test_split(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    # Defines Features (X) and Labels (y)
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    # Normalize Features
    scalar = StandardScaler()
    X_scaled = scalar.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def preprocess(path : str) -> pd.DataFrame:
    df = load_data(path)
    df = remove_zeroes(df)
    return df



