# BountyHunter — Pirate Bounty Predictor

Train ML models to predict a pirate's Marine-issued bounty (in millions of berries) from crew stats. Covers the full ML lifecycle: synthetic data → classical baselines → PyTorch neural net → experiment tracking → evaluation → interactive demo.

---

## Features

- **Synthetic data generation** — NumPy-vectorized pirate dataset (1,000 samples, 5 features)
- **Classical ML baselines** — LinearRegression, RandomForest, XGBoost with cross-validation and GridSearchCV
- **PyTorch neural network** — 3-layer feedforward net with full training loop (forward → loss → zero_grad → backward → step)
- **Experiment tracking** — Weights & Biases logging (train/val RMSE, val MAE per epoch)
- **Model comparison** — Side-by-side regression metrics (MAE, R²) + classification demo (precision, recall, F1, AUC-ROC)
- **Streamlit UI** — Interactive web demo

---

## Tech Stack

Python · NumPy · Pandas · scikit-learn · XGBoost · PyTorch · Weights & Biases · Streamlit

---

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Generate data
python data.py

# 3. Train baselines
python baseline.py

# 4. Train PyTorch model (optional: wandb login first)
wandb login
python train.py

# 5. Compare all models
python evaluate.py

# 6. Launch demo
streamlit run streamlit_app.py
```

---

## Dataset

| Feature | Type | Range |
|---|---|---|
| `devil_fruit` | bool | 0 or 1 |
| `haki_level` | int | 0–3 |
| `crew_size` | int | 1–500 |
| `notoriety_score` | float | 0.0–10.0 |
| `region` | encoded int | 0=East Blue, 1=Grand Line, 2=New World |
| **`bounty_berries`** | float | target (millions) |

Generated synthetically with NumPy — no external data required.

---

## Expected Results

| Model | MAE (M berries) | R² |
|---|---|---|
| LinearRegression | ~30.0 | ~0.91 |
| RandomForest | ~28.0 | ~0.93 |
| RandomForest (tuned) | ~27.0 | ~0.94 |
| XGBoost | ~27.0 | ~0.94 |
| BountyNet (PyTorch) | ~29.0 | ~0.92 |

---

## Project Structure

```
phase-3-bountyhunter/
├── data.py              # Synthetic dataset generation
├── baseline.py          # sklearn baselines + CV + GridSearch
├── model.py             # BountyNet — PyTorch nn.Module
├── train.py             # Training loop + W&B logging
├── evaluate.py          # Regression + classification metrics
├── streamlit_app.py     # Streamlit interactive demo
├── tests/               # Pytest test suite
├── requirements.txt
└── README.md
```

---

## Deploy on Streamlit Community Cloud (Free)

1. Push this project to a **public GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"** → connect your GitHub repo
4. Set the main file to `streamlit_app.py`
5. Click **Deploy** — live at `https://your-app.streamlit.app`

---

## Key Concepts Demonstrated

- NumPy vectorized operations and broadcasting
- sklearn `fit`/`predict` pattern with StandardScaler
- Cross-validation for reliable performance estimates
- GridSearchCV for hyperparameter tuning
- PyTorch `nn.Module`, training loop, and checkpointing
- Regression metrics (MAE, RMSE, R²) vs classification metrics (precision, recall, F1, AUC-ROC)
- Experiment tracking with W&B
- Streamlit for model deployment
