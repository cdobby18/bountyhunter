"""
Gradio UI — BountyHunter

Enter a pirate's stats and get their predicted bounty in berries.
Auto-trains a model if no checkpoint exists.

Run:
    python app.py
"""

import numpy as np
import torch
import gradio as gr
from pathlib import Path

from model import BountyNet

FEATURES = ["devil_fruit", "haki_level", "crew_size", "notoriety_score", "region"]
CHECKPOINT = "bounty_model.pt"
REGION_MAP = {"East Blue": 0, "Grand Line": 1, "New World": 2}


def load_model():
    if not Path(CHECKPOINT).exists():
        print("No checkpoint found — generating data and training...")
        from data import generate
        from train import train as run_training
        df = generate()
        df.to_csv("pirates.csv", index=False)
        run_training()

    checkpoint = torch.load(CHECKPOINT, weights_only=False)
    model = BountyNet(input_dim=len(FEATURES))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint["scaler"]


MODEL, SCALER = load_model()


def predict(
    devil_fruit: bool,
    haki_level: int,
    crew_size: int,
    notoriety_score: float,
    region: str,
) -> str:
    x = np.array([[
        int(devil_fruit),
        int(haki_level),
        int(crew_size),
        float(notoriety_score),
        REGION_MAP[region],
    ]], dtype=np.float32)

    x_scaled = SCALER.transform(x).astype(np.float32)

    with torch.no_grad():
        bounty = MODEL(torch.tensor(x_scaled)).item()

    bounty = max(bounty, 10.0)

    if bounty >= 1000:
        display = f"{bounty / 1000:.1f} Billion Berries"
    else:
        display = f"{bounty:.0f} Million Berries"

    return f"Bounty: {display}"


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Checkbox(label="Devil Fruit User?"),
        gr.Slider(minimum=0, maximum=3, step=1, value=0, label="Haki Level (0–3)"),
        gr.Slider(minimum=1, maximum=500, step=1, value=10, label="Crew Size"),
        gr.Slider(minimum=0.0, maximum=10.0, step=0.1, value=5.0, label="Notoriety Score"),
        gr.Dropdown(choices=["East Blue", "Grand Line", "New World"], value="East Blue", label="Region"),
    ],
    outputs=gr.Textbox(label="Predicted Bounty"),
    title="BountyHunter — Pirate Bounty Predictor",
    description="Enter your pirate's stats to get their Marine-issued bounty prediction.",
    examples=[
        [True, 3, 10, 9.5, "New World"],
        [False, 0, 3, 2.0, "East Blue"],
        [True, 2, 50, 7.0, "Grand Line"],
    ],
)

if __name__ == "__main__":
    demo.launch()
