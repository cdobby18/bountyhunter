"""
Model evaluation — two sections:

PART A — Regression: compare BountyNet vs sklearn baselines (MAE, R²)
PART B — Classification: demonstrate confusion matrix, precision, recall, F1, AUC-ROC
         using a "high bounty" binary classifier (bounty >= 500M berries → dangerous pirate)

Why two parts?
  The main project is regression (predict a number).
  But most AI engineering evaluation uses classification metrics — so this file
  shows both so you know when to use each.

Run:
    python train.py    # train first
    python evaluate.py
"""

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from baseline import run as baseline_results
from data import generate
from model import BountyNet

FEATURES = ["devil_fruit", "haki_level", "crew_size", "notoriety_score", "region"]
TARGET = "bounty_berries"
CHECKPOINT = "bounty_model.pt"


def load_test_set():
    try:
        df = pd.read_csv("pirates.csv")
    except FileNotFoundError:
        df = generate()

    X = df[FEATURES].values.astype(np.float32)
    y = df[TARGET].values.astype(np.float32)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_test, y_test


def pytorch_metrics(X_test: np.ndarray, y_test: np.ndarray) -> dict:
    checkpoint = torch.load(CHECKPOINT, weights_only=False)
    scaler = checkpoint["scaler"]
    state = checkpoint["model"]

    model = BountyNet(input_dim=len(FEATURES))
    model.load_state_dict(state)
    model.eval()

    X_scaled = scaler.transform(X_test).astype(np.float32)
    with torch.no_grad():
        preds = model(torch.tensor(X_scaled)).numpy()

    return {
        "MAE": mean_absolute_error(y_test, preds),
        "R2": r2_score(y_test, preds),
    }


def classification_demo(X_test: np.ndarray, y_test: np.ndarray):
    """
    PART B — Classification metrics.

    Task: predict whether a pirate is "dangerous" (bounty >= 500M berries).
    This is a binary classification problem: 1 = dangerous, 0 = not.

    Metrics explained:
      Accuracy    — correct / total. Misleading on imbalanced data.
      Precision   — of pirates we called dangerous, how many actually were?
                    High precision = fewer false alarms.
      Recall      — of all actually-dangerous pirates, how many did we catch?
                    High recall = fewer missed threats. (also called Sensitivity)
      F1          — harmonic mean of precision + recall. Use when both matter.
      AUC-ROC     — discrimination ability regardless of threshold.
                    0.5 = random guessing, 1.0 = perfect.
      Confusion Matrix:
                    Predicted 0  Predicted 1
        Actual 0  [  TN          FP  ]
        Actual 1  [  FN          TP  ]
    """
    try:
        df = pd.read_csv("pirates.csv")
    except FileNotFoundError:
        df = generate()

    X = df[FEATURES].values
    y_bounty = df[TARGET].values
    y_class = (y_bounty >= 500).astype(int)  # 1 = dangerous, 0 = not

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_class, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_c, y_train_c)
    y_pred = clf.predict(X_test_c)
    y_proba = clf.predict_proba(X_test_c)[:, 1]  # probability of class 1

    print("\n" + "=" * 52)
    print("PART B — Classification: 'Dangerous Pirate' Detector")
    print("  (bounty >= 500M berries → class 1)")
    print("=" * 52)

    print(f"\nClass distribution: {y_class.sum()} dangerous / {(y_class == 0).sum()} not")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test_c, y_pred)
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

    print("\nClassification Report:")
    print(classification_report(y_test_c, y_pred, target_names=["Not Dangerous", "Dangerous"]))

    auc = roc_auc_score(y_test_c, y_proba)
    print(f"AUC-ROC: {auc:.4f}  (1.0 = perfect, 0.5 = random guess)")


def run():
    X_test, y_test = load_test_set()

    # --- PART A: Regression ---
    print("=" * 52)
    print("PART A — Regression: Bounty Prediction (MAE, R²)")
    print("=" * 52)

    baselines = baseline_results()

    try:
        nn_metrics = pytorch_metrics(X_test, y_test)
        all_results = {**baselines, "BountyNet (PyTorch)": nn_metrics}
    except FileNotFoundError:
        print(f"[!] {CHECKPOINT} not found — run train.py first")
        all_results = baselines

    print("\n--- Regression Summary ---")
    print(f"{'Model':<25} {'MAE (M berries)':>16} {'R²':>8}")
    print("-" * 52)
    for name, m in all_results.items():
        marker = " ← best" if m["MAE"] == min(v["MAE"] for v in all_results.values()) else ""
        print(f"{name:<25} {m['MAE']:>14.2f}  {m['R2']:>8.4f}{marker}")

    # --- PART B: Classification metrics demo ---
    classification_demo(X_test, y_test)


if __name__ == "__main__":
    run()
