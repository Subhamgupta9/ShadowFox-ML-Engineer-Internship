"""
Evaluation and Metrics Calculation Module.
ShadowFox ML Engineer Internship — Task 2: Car Selling Price Prediction
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_validate

def calculate_metrics(y_true: np.ndarray or pd.Series, y_pred: np.ndarray) -> dict:
    """
    Computes R2, MAE, MSE, and RMSE regression metrics.
    
    Args:
        y_true: Ground truth target values.
        y_pred: Predicted target values.
        
    Returns:
        Dictionary containing verified regression metrics.
    """
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    medae = np.median(np.abs(np.array(y_true) - np.array(y_pred)))
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    return {
        "R2_Score": float(r2),
        "MAE": float(mae),
        "MedAE": float(medae),
        "MSE": float(mse),
        "RMSE": float(rmse)
    }

def cross_validate_pipeline(pipeline, X_train: pd.DataFrame, y_train: pd.Series, cv: int = 5, random_state: int = 42) -> dict:
    """
    Performs k-fold cross-validation exclusively on training data to prevent test leakage.
    
    Args:
        pipeline: Scikit-learn Pipeline (preprocessor + model).
        X_train: Training feature DataFrame.
        y_train: Training target Series.
        cv: Number of folds (default 5).
        random_state: Random state for shuffling.
        
    Returns:
        Dictionary of mean and std metrics across folds.
    """
    kfold = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    scoring = {
        'r2': 'r2',
        'neg_mae': 'neg_mean_absolute_error',
        'neg_mse': 'neg_mean_squared_error',
        'neg_rmse': 'neg_root_mean_squared_error'
    }
    
    cv_results = cross_validate(pipeline, X_train, y_train, cv=kfold, scoring=scoring, return_train_score=True)
    
    return {
        "CV_R2_Mean": float(np.mean(cv_results['test_r2'])),
        "CV_R2_Std": float(np.std(cv_results['test_r2'])),
        "CV_MAE_Mean": float(-np.mean(cv_results['test_neg_mae'])),
        "CV_MAE_Std": float(np.std(cv_results['test_neg_mae'])),
        "CV_RMSE_Mean": float(np.mean(np.sqrt(-cv_results['test_neg_mse']))),
        "CV_RMSE_Std": float(np.std(np.sqrt(-cv_results['test_neg_mse']))),
        "Train_R2_Mean": float(np.mean(cv_results['train_r2']))
    }
