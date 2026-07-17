"""
Classical ML baseline using sklearn.

Covers:
  - LinearRegression         (supervised — linear model)
  - RandomForestRegressor    (supervised — ensemble, tree-based)
  - XGBRegressor             (supervised — gradient boosting, often wins)
  - cross_val_score          (cross-validation — more reliable than one split)
  - GridSearchCV             (hyperparameter tuning)

Run:
    python data.py        # generate pirates.csv first
    python baseline.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from data import generate

FEATURES = ["devil_fruit", "haki_level", "crew_size", "notoriety_score", "region"]
TARGET = "bounty_berries"


def load_data():
    try:
        df = pd.read_csv("pirates.csv")
    except FileNotFoundError:
        df = generate()
    X = df[FEATURES].values
    y = df[TARGET].values
    return X, y


def run():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = {}

    # --- Linear Regression ---
    lr = LinearRegression()
    lr.fit(X_train_s, y_train)
    preds_lr = lr.predict(X_test_s)
    results["LinearRegression"] = {
        "MAE": mean_absolute_error(y_test, preds_lr),
        "R2": r2_score(y_test, preds_lr),
    }

    # --- Random Forest (tree-based — no scaling needed) ---
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    preds_rf = rf.predict(X_test)
    results["RandomForest"] = {
        "MAE": mean_absolute_error(y_test, preds_rf),
        "R2": r2_score(y_test, preds_rf),
    }

    # --- XGBoost ---
    # Gradient boosting: builds trees sequentially, each correcting the last.
    # Often the strongest classical ML model on tabular data.
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbosity=0)
    xgb.fit(X_train, y_train)
    preds_xgb = xgb.predict(X_test)
    results["XGBoost"] = {
        "MAE": mean_absolute_error(y_test, preds_xgb),
        "R2": r2_score(y_test, preds_xgb),
    }

    # --- Cross-Validation (5-fold on XGBoost) ---
    # Why: a single train/test split can be lucky or unlucky.
    # CV splits the data 5 different ways and averages the score — more reliable.
    # negative_mean_absolute_error → sklearn convention: higher = better, so MAE is negated
    cv_scores = cross_val_score(
        XGBRegressor(n_estimators=100, verbosity=0, random_state=42),
        X, y,
        cv=5,
        scoring="neg_mean_absolute_error",
    )
    cv_mae = -cv_scores.mean()
    print(f"\n5-Fold Cross-Validation (XGBoost)")
    print(f"  MAE per fold : {(-cv_scores).round(2)}")
    print(f"  Mean MAE     : {cv_mae:.2f} ± {cv_scores.std():.2f}")

    # --- Hyperparameter Tuning (GridSearchCV on RandomForest) ---
    # GridSearchCV tries every combination of params and picks the best using CV.
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [None, 10],
    }
    grid = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid,
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    best_preds = grid.best_estimator_.predict(X_test)
    results["RandomForest (tuned)"] = {
        "MAE": mean_absolute_error(y_test, best_preds),
        "R2": r2_score(y_test, best_preds),
    }
    print(f"\nBest RF params: {grid.best_params_}")

    # --- Summary ---
    print("\n=== Baseline Results (test set) ===")
    for name, metrics in results.items():
        print(f"\n{name}")
        print(f"  MAE : {metrics['MAE']:.2f}M berries")
        print(f"  R²  : {metrics['R2']:.4f}")

    print("\n[Beat these numbers with PyTorch in train.py]")
    return results


if __name__ == "__main__":
    run()
