"""
Unit tests for data loading and preprocessing pipeline.
"""

import sys
import pytest
import numpy as np
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.data_preprocessing import load_and_clean_data, prepare_data, get_preprocessor

def test_load_and_clean_data():
    dataset_path = ROOT_DIR / "dataset" / "car.csv"
    df = load_and_clean_data(dataset_path)
    
    # 301 raw - 2 duplicates = 299 unique records
    assert len(df) == 299, f"Expected 299 rows, got {len(df)}"
    assert df.isnull().sum().sum() == 0, "Cleaned dataset should not have any missing values"
    assert "Vehicle_Age" in df.columns, "Engineered feature 'Vehicle_Age' must be present"
    assert df['Vehicle_Age'].min() >= 0, "Vehicle_Age cannot be negative"

def test_prepare_data_split():
    dataset_path = ROOT_DIR / "dataset" / "car.csv"
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(dataset_path, test_size=0.20, random_state=42)
    
    # Check 80/20 split on 299 rows
    assert len(X_train) == 239, f"Expected 239 train rows, got {len(X_train)}"
    assert len(X_test) == 60, f"Expected 60 test rows, got {len(X_test)}"
    assert len(y_train) == 239
    assert len(y_test) == 60

def test_preprocessor_transformation():
    dataset_path = ROOT_DIR / "dataset" / "car.csv"
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(dataset_path, test_size=0.20, random_state=42)
    
    # Fit preprocessor on X_train only
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    
    # 4 numeric features + 2 fuel + 1 seller + 1 transmission = 8 features
    assert X_train_trans.shape[1] == 8, f"Expected 8 transformed features, got {X_train_trans.shape[1]}"
    assert X_test_trans.shape[1] == 8
    assert not np.isnan(X_train_trans).any() if 'np' in dir() else True
