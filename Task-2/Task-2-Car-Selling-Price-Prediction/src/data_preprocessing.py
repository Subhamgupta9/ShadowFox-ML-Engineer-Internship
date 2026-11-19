"""
Data Preprocessing and Feature Engineering Module.
ShadowFox ML Engineer Internship — Task 2: Car Selling Price Prediction
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

REF_YEAR = 2018
NUM_FEATURES = ['Present_Price', 'Kms_Driven', 'Vehicle_Age', 'Owner']
CAT_FEATURES = ['Fuel_Type', 'Seller_Type', 'Transmission']
FEATURE_COLUMNS = ['Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner', 'Vehicle_Age']
TARGET_COLUMN = 'Selling_Price'

def load_and_clean_data(data_path: str or Path) -> pd.DataFrame:
    """
    Loads raw car dataset, removes exact duplicate records, and engineers Vehicle_Age.
    
    Args:
        data_path: Path to car.csv dataset.
        
    Returns:
        Cleaned pandas DataFrame with engineered features.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
        
    df = pd.read_csv(path)
    
    # Prune duplicate records to prevent data leakage across splits
    df_clean = df.drop_duplicates().copy().reset_index(drop=True)
    
    # Feature Engineering: Vehicle Age relative to 2018 baseline
    df_clean['Vehicle_Age'] = REF_YEAR - df_clean['Year']
    
    return df_clean

def get_preprocessor() -> ColumnTransformer:
    """
    Constructs an unfitted ColumnTransformer pipeline for numeric scaling and categorical one-hot encoding.
    
    Returns:
        Configured ColumnTransformer instance.
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUM_FEATURES),
            ('cat', categorical_transformer, CAT_FEATURES)
        ],
        remainder='drop'
    )
    return preprocessor

def prepare_data(data_path: str or Path, test_size: float = 0.20, random_state: int = 42):
    """
    Loads data and splits into train and test sets strictly before fitting transformers to prevent leakage.
    
    Returns:
        X_train, X_test, y_train, y_test, preprocessor
    """
    df = load_and_clean_data(data_path)
    
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    preprocessor = get_preprocessor()
    return X_train, X_test, y_train, y_test, preprocessor

if __name__ == "__main__":
    test_path = Path(__file__).resolve().parent.parent / "dataset" / "car.csv"
    X_tr, X_te, y_tr, y_te, prep = prepare_data(test_path)
    print(f"Data Preprocessing Module Verified.")
    print(f"Train Shape: {X_tr.shape}, Test Shape: {X_te.shape}")
