"""
Model Training, Benchmarking, and Hyperparameter Optimization Pipeline.
ShadowFox ML Engineer Internship — Task 2: Car Selling Price Prediction
"""

import sys
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Machine Learning Estimators
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV

# Internal modules
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.data_preprocessing import prepare_data, NUM_FEATURES, CAT_FEATURES
from src.evaluate import calculate_metrics, cross_validate_pipeline

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "dataset" / "car.csv"
MODEL_PATH = BASE_DIR / "models" / "car_price_model.joblib"
REPORTS_DIR = BASE_DIR / "reports"
FIG_DIR = REPORTS_DIR / "figures"
CSV_PATH = REPORTS_DIR / "experiment_results.csv"

FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# Plotting configuration
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Segoe UI', 'DejaVu Sans', 'Arial'

def select_best_model_cv(candidate_results: list[dict], metric: str = "CV_R2_Mean", mode: str = "max") -> str:
    """
    Selects the best performing candidate model strictly based on training cross-validation metrics.
    Enforces zero test-set leakage by forbidding the use of test-set metrics for model selection.

    Args:
        candidate_results: List of metric dictionaries for candidate models.
        metric: CV metric name to optimize (default: "CV_R2_Mean").
        mode: Optimization direction ('max' for R2, 'min' for MAE/RMSE).

    Returns:
        Name of the winning model.

    Raises:
        ValueError: If a test-set metric is supplied or candidate list is invalid.
    """
    if "test" in metric.lower():
        raise ValueError(
            f"Test-set leakage prevented: Cannot use test metric '{metric}' for model selection. "
            f"Model selection must strictly rely on training cross-validation metrics."
        )
    if not candidate_results:
        raise ValueError("Candidate results list cannot be empty.")

    for r in candidate_results:
        if metric not in r:
            raise KeyError(f"Metric '{metric}' not found in candidate result: {r.get('Model')}")

    if mode == "max":
        best = max(candidate_results, key=lambda x: x[metric])
    elif mode == "min":
        best = min(candidate_results, key=lambda x: x[metric])
    else:
        raise ValueError(f"Invalid mode '{mode}'. Choose 'max' or 'min'.")

    return best["Model"]

def run_benchmarks():
    print("=" * 70)
    print("CARSPHERE AI — MULTI-MODEL REGRESSION BENCHMARK (LEAKAGE-FREE)")
    print("=" * 70)

    # 1. Prepare Data with zero leakage (split before fitting)
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(DATA_PATH, test_size=0.20, random_state=42)
    print(f"Training split: {X_train.shape[0]} samples | Test split: {X_test.shape[0]} samples")
    print("Notice: Held-out test set will be evaluated strictly ONCE after model selection.\n")

    # 2. Define Candidate Algorithms
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "Extra Trees Regressor": ExtraTreesRegressor(n_estimators=100, random_state=42)
    }

    results = []
    fitted_pipelines = {}

    # -------------------------------------------------------------
    # Phase 1: 5-Fold Cross-Validation on Training Data ONLY
    # -------------------------------------------------------------
    print("--- Phase 1: 5-Fold Cross-Validation on Training Split ---")
    for name, estimator in models.items():
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', estimator)
        ])

        # Cross-validation strictly on training data
        cv_scores = cross_validate_pipeline(pipe, X_train, y_train, cv=5, random_state=42)

        # Fit on full training split
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe

        record = {
            "Model": name,
            "Type": "Baseline",
            "CV_R2_Mean": cv_scores["CV_R2_Mean"],
            "CV_R2_Std": cv_scores["CV_R2_Std"],
            "CV_MAE_Mean": cv_scores["CV_MAE_Mean"],
            "CV_RMSE_Mean": cv_scores["CV_RMSE_Mean"],
            "Train_R2": cv_scores["Train_R2_Mean"]
        }
        results.append(record)
        print(f"[{name}]")
        print(f"  CV R2:   {record['CV_R2_Mean']:.4f} ± {record['CV_R2_Std']:.4f} | CV MAE: {record['CV_MAE_Mean']:.4f} | CV RMSE: {record['CV_RMSE_Mean']:.4f}")

    # Baseline selection strictly via Training CV R2
    best_baseline_name = select_best_model_cv(results, metric="CV_R2_Mean")
    best_baseline_cv = next(r["CV_R2_Mean"] for r in results if r["Model"] == best_baseline_name)
    print(f"\nTop Performing Baseline (Selected strictly via CV R2): {best_baseline_name} (CV R2 = {best_baseline_cv:.4f})")

    # -------------------------------------------------------------
    # Phase 2: Hyperparameter Optimization on Training Split ONLY
    # -------------------------------------------------------------
    print("\n--- Phase 2: Hyperparameter Optimization via GridSearchCV on Training Split ---")

    if "Random Forest" in best_baseline_name:
        param_grid = {
            'regressor__n_estimators': [100, 150, 200, 250],
            'regressor__max_depth': [None, 5, 8, 12],
            'regressor__min_samples_split': [2, 3, 5],
            'regressor__min_samples_leaf': [1, 2, 4],
            'regressor__max_features': ['sqrt', 1.0]
        }
        base_estimator = RandomForestRegressor(random_state=42)
    elif "Gradient Boosting" in best_baseline_name:
        param_grid = {
            'regressor__n_estimators': [100, 150, 200],
            'regressor__learning_rate': [0.03, 0.05, 0.1, 0.15],
            'regressor__max_depth': [3, 4, 5],
            'regressor__min_samples_split': [2, 3, 5],
            'regressor__subsample': [0.8, 0.9, 1.0]
        }
        base_estimator = GradientBoostingRegressor(random_state=42)
    elif "Extra Trees" in best_baseline_name:
        param_grid = {
            'regressor__n_estimators': [100, 150, 200],
            'regressor__max_depth': [None, 6, 10, 15],
            'regressor__min_samples_split': [2, 3, 5],
            'regressor__min_samples_leaf': [1, 2]
        }
        base_estimator = ExtraTreesRegressor(random_state=42)
    else:
        param_grid = {
            'regressor__alpha': [0.01, 0.1, 1.0, 5.0, 10.0, 20.0, 50.0]
        }
        base_estimator = Ridge(random_state=42)

    tuning_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', base_estimator)
    ])

    grid_search = GridSearchCV(
        tuning_pipe,
        param_grid=param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        refit=True
    )

    # Tuning strictly on training data
    grid_search.fit(X_train, y_train)

    best_tuned_pipe = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print("Optimal Hyperparameters (Selected via CV):")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # Evaluate tuned model via cross-validation strictly on training data
    cv_tuned = cross_validate_pipeline(best_tuned_pipe, X_train, y_train, cv=5, random_state=42)

    tuned_name = f"{best_baseline_name} (Optimized)"
    tuned_record = {
        "Model": tuned_name,
        "Type": "Hyperparameter Tuned",
        "CV_R2_Mean": cv_tuned["CV_R2_Mean"],
        "CV_R2_Std": cv_tuned["CV_R2_Std"],
        "CV_MAE_Mean": cv_tuned["CV_MAE_Mean"],
        "CV_RMSE_Mean": cv_tuned["CV_RMSE_Mean"],
        "Train_R2": cv_tuned["Train_R2_Mean"]
    }
    results.append(tuned_record)
    fitted_pipelines[tuned_name] = best_tuned_pipe

    print(f"\n[{tuned_name}]")
    print(f"  CV R2:   {tuned_record['CV_R2_Mean']:.4f} ± {tuned_record['CV_R2_Std']:.4f} | CV MAE: {tuned_record['CV_MAE_Mean']:.4f} | CV RMSE: {tuned_record['CV_RMSE_Mean']:.4f}")

    # -------------------------------------------------------------
    # Phase 3: Final Model Selection (Strictly via CV Metrics)
    # -------------------------------------------------------------
    final_selected_name = select_best_model_cv(results, metric="CV_R2_Mean")
    final_pipeline = fitted_pipelines[final_selected_name]
    print(f"\n=======================================================")
    print(f"WINNING PRODUCTION MODEL: {final_selected_name}")
    print(f"Selected strictly by Training CV R2: {next(r['CV_R2_Mean'] for r in results if r['Model'] == final_selected_name):.4f}")
    print(f"=======================================================")

    # Save production model
    joblib.dump(final_pipeline, MODEL_PATH)
    print(f"[OK] Saved production model pipeline to: {MODEL_PATH}")

    # -------------------------------------------------------------
    # Phase 4: Single Unbiased Evaluation on Held-Out Test Set
    # -------------------------------------------------------------
    print(f"\n--- Phase 4: Held-Out Test Set Evaluation (Exclusively on Selected {final_selected_name}) ---")
    # Strict protocol: Evaluate X_test ONCE only on the selected winning model
    y_pred_test = final_pipeline.predict(X_test)
    test_scores = calculate_metrics(y_test, y_pred_test)

    for record in results:
        if record["Model"] == final_selected_name:
            record["Test_R2"] = test_scores["R2_Score"]
            record["Test_MAE"] = test_scores["MAE"]
            record["Test_MSE"] = test_scores["MSE"]
            record["Test_RMSE"] = test_scores["RMSE"]
        else:
            record["Test_R2"] = np.nan
            record["Test_MAE"] = np.nan
            record["Test_MSE"] = np.nan
            record["Test_RMSE"] = np.nan

    print(f"[{final_selected_name}] Test R2: {test_scores['R2_Score']:.4f} | Test MAE: {test_scores['MAE']:.4f} | Test RMSE: {test_scores['RMSE']:.4f}")

    final_df_results = pd.DataFrame(results)
    final_df_results.to_csv(CSV_PATH, index=False)
    print(f"\n[OK] Saved verified benchmark metrics to: {CSV_PATH}")

    # -------------------------------------------------------------
    # Phase 5: Generate Evaluation Figures
    # -------------------------------------------------------------
    generate_evaluation_figures(
        final_df_results,
        final_pipeline,
        X_train,
        X_test,
        y_test,
        y_pred_test,
        final_selected_name
    )

    return final_df_results, final_selected_name

def generate_evaluation_figures(df_results, best_pipe, X_train, X_test, y_test, y_pred, model_name):
    # 1. Model Comparison Chart
    plt.figure(figsize=(12, 6), dpi=300)
    plot_df = df_results.copy()

    x = np.arange(len(plot_df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 6), dpi=300)
    rects1 = ax.bar(x - width/2, plot_df['CV_R2_Mean'], width, label='5-Fold CV R² (Training Split)', color='#3B82F6', edgecolor='black', alpha=0.9)
    rects2 = ax.bar(x + width/2, plot_df['Test_R2'].fillna(0), width, label='Held-Out Test R² (Selected Model Only)', color='#10B981', edgecolor='black', alpha=0.9)

    ax.set_ylabel('R² Score (Coefficient of Determination)', fontsize=12, fontweight='bold')
    ax.set_title('Regression Benchmark: Training 5-Fold CV R² vs. Single Held-Out Test R²', fontsize=14, fontweight='bold', pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df['Model'], rotation=18, ha='right', fontsize=10, fontweight='bold')
    ax.legend(frameon=True, facecolor='white', loc='lower right', fontsize=11)
    ax.set_ylim(0, 1.05)

    # Attach metric labels
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
    for rect in rects2:
        h = rect.get_height()
        if h > 0.01:
            ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                        textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#047857')

    plt.tight_layout()
    p1 = FIG_DIR / "model_comparison_chart.png"
    plt.savefig(p1, bbox_inches='tight')
    plt.close()
    print(f"[OK] Generated {p1.name}")

    # 2. Residuals & Prediction Scatter Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # Scatter Actual vs Predicted
    axes[0].scatter(y_test, y_pred, color='#2563EB', alpha=0.8, edgecolor='black', s=60, label='Test Observations')
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal 1:1 Identity')
    axes[0].set_title(f'Actual vs. Predicted Selling Price ({model_name})', fontsize=13, fontweight='bold', pad=12)
    axes[0].set_xlabel('Actual Selling Price (Lakhs ₹)', fontsize=11)
    axes[0].set_ylabel('Predicted Selling Price (Lakhs ₹)', fontsize=11)
    axes[0].legend(frameon=True, facecolor='white')

    # Residual Distribution
    residuals = y_test - y_pred
    sns.histplot(residuals, kde=True, color='#8B5CF6', ax=axes[1], bins=20, edgecolor='black', alpha=0.6)
    axes[1].axvline(0, color='#DC2626', linestyle='--', linewidth=2, label='Zero Error Reference')
    axes[1].set_title(f'Residuals Distribution ({model_name})', fontsize=13, fontweight='bold', pad=12)
    axes[1].set_xlabel('Residual Error (Actual - Predicted in Lakhs ₹)', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].legend(frameon=True, facecolor='white')

    plt.tight_layout()
    p2 = FIG_DIR / "residuals_analysis.png"
    plt.savefig(p2, bbox_inches='tight')
    plt.close()
    print(f"[OK] Generated {p2.name}")

    # 3. Feature Importance
    regressor = best_pipe.named_steps['regressor']
    if hasattr(regressor, 'feature_importances_'):
        cat_encoder = best_pipe.named_steps['preprocessor'].named_transformers_['cat']
        encoded_cat = cat_encoder.get_feature_names_out(CAT_FEATURES).tolist()
        feature_names = NUM_FEATURES + encoded_cat
        importances = regressor.feature_importances_

        fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=True)

        plt.figure(figsize=(10, 6), dpi=300)
        colors = ['#3B82F6' if 'Present_Price' in f else '#60A5FA' for f in fi_df['Feature']]
        bars = plt.barh(fi_df['Feature'], fi_df['Importance'], color=colors, edgecolor='black', alpha=0.9)
        plt.title(f'Feature Importance Weights ({model_name})', fontsize=14, fontweight='bold', pad=14)
        plt.xlabel('Normalized Importance Score (MDI / Gini Importance)', fontsize=12)

        for bar in bars:
            w = bar.get_width()
            plt.annotate(f"{w:.3f}", xy=(w, bar.get_y() + bar.get_height()/2), xytext=(5, 0),
                         textcoords="offset points", ha='left', va='center', fontsize=10, fontweight='bold')

        plt.xlim(0, max(fi_df['Importance']) * 1.15)
        plt.tight_layout()
        p3 = FIG_DIR / "feature_importance.png"
        plt.savefig(p3, bbox_inches='tight')
        plt.close()
        print(f"[OK] Generated {p3.name}")

if __name__ == "__main__":
    run_benchmarks()
