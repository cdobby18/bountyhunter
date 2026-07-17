"""
Streamlit UI — BountyHunter

Enter a pirate's stats and get their predicted bounty in berries.
Auto-trains a model if no checkpoint exists.

Run locally:
    streamlit run streamlit_app.py

Deploy on Streamlit Community Cloud:
    Push to GitHub → https://streamlit.io/cloud
"""

import numpy as np
import torch
import streamlit as st
from pathlib import Path

from model import BountyNet

FEATURES = ["devil_fruit", "haki_level", "crew_size", "notoriety_score", "region"]
CHECKPOINT = "bounty_model.pt"
REGION_MAP = {"East Blue": 0, "Grand Line": 1, "New World": 2}


@st.cache_resource
def load_model():
    if not Path(CHECKPOINT).exists():
        st.info("No checkpoint found — generating data and training...")
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


st.set_page_config(page_title="BountyHunter", page_icon="🏴‍☠️")
st.title("🏴‍☠️ BountyHunter")
st.markdown("Predict a pirate's Marine-issued bounty from their crew stats.")

MODEL, SCALER = load_model()

with st.form("pirate_form"):
    col1, col2 = st.columns(2)

    with col1:
        devil_fruit = st.checkbox("Devil Fruit User?")
        haki_level = st.slider("Haki Level", 0, 3, 0)
        crew_size = st.slider("Crew Size", 1, 500, 10)

    with col2:
        notoriety_score = st.slider("Notoriety Score", 0.0, 10.0, 5.0, step=0.1)
        region = st.selectbox("Region", ["East Blue", "Grand Line", "New World"])

    submitted = st.form_submit_button("Predict Bounty")

if submitted:
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

    st.success(f"**Predicted Bounty: {display}**")
