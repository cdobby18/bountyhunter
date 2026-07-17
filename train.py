"""
Train BountyNet on pirates.csv with W&B experiment tracking.

Steps (the loop to memorize):
    1. Forward pass   — model(X) → predictions
    2. Compute loss   — loss_fn(preds, y)
    3. Zero gradients — optimizer.zero_grad()
    4. Backprop       — loss.backward()
    5. Update weights — optimizer.step()

Run:
    python data.py     # generate pirates.csv first
    wandb login        # one-time setup
    python train.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data import generate
from model import BountyNet

FEATURES = ["devil_fruit", "haki_level", "crew_size", "notoriety_score", "region"]
TARGET = "bounty_berries"

CONFIG = {
    "epochs": 100,
    "lr": 1e-3,
    "batch_size": 64,
    "hidden": [64, 32],
    "seed": 42,
}


def load_tensors():
    try:
        df = pd.read_csv("pirates.csv")
    except FileNotFoundError:
        df = generate()

    X = df[FEATURES].values.astype(np.float32)
    y = df[TARGET].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=CONFIG["seed"]
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    to_t = lambda arr: torch.tensor(arr)
    return (
        to_t(X_train), to_t(y_train),
        to_t(X_test),  to_t(y_test),
        scaler,
    )


def train():
    torch.manual_seed(CONFIG["seed"])

    try:
        wandb.init(project="bountyhunter", config=CONFIG)
        use_wandb = True
    except Exception:
        use_wandb = False

    X_train, y_train, X_test, y_test, scaler = load_tensors()

    model = BountyNet(input_dim=len(FEATURES))
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    loss_fn = nn.MSELoss()

    n = len(X_train)
    bs = CONFIG["batch_size"]

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        indices = torch.randperm(n)

        epoch_loss = 0.0
        for start in range(0, n, bs):
            idx = indices[start : start + bs]
            Xb, yb = X_train[idx], y_train[idx]

            preds = model(Xb)           # 1. forward
            loss = loss_fn(preds, yb)   # 2. loss

            optimizer.zero_grad()       # 3. zero grads
            loss.backward()             # 4. backprop
            optimizer.step()            # 5. update

            epoch_loss += loss.item() * len(Xb)

        train_rmse = (epoch_loss / n) ** 0.5

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(X_test)
            val_loss = loss_fn(val_preds, y_test).item()
            val_rmse = val_loss ** 0.5
            val_mae = (val_preds - y_test).abs().mean().item()

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_rmse": train_rmse,
                "val_rmse": val_rmse,
                "val_mae": val_mae,
            })

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | train RMSE {train_rmse:.2f} | val RMSE {val_rmse:.2f} | val MAE {val_mae:.2f}")

    torch.save({"model": model.state_dict(), "scaler": scaler}, "bounty_model.pt")
    print("\nModel saved → bounty_model.pt")
    print(f"Final val MAE: {val_mae:.2f}M berries")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    train()
