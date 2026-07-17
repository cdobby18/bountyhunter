"""
Generate a synthetic pirate dataset and save to pirates.csv.

Features:
  devil_fruit       — 0 or 1 (bool)
  haki_level        — 0 to 3
  crew_size         — 1 to 500
  notoriety_score   — 0.0 to 10.0
  region            — 0=East Blue, 1=Grand Line, 2=New World

Target:
  bounty_berries    — bounty in millions (continuous, regression)
"""

import numpy as np
import pandas as pd

SEED = 42
N = 1000


def generate(n: int = N, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    devil_fruit = rng.integers(0, 2, n)          # 0 or 1
    haki_level = rng.integers(0, 4, n)            # 0–3
    crew_size = rng.integers(1, 501, n)            # 1–500
    notoriety = rng.uniform(0, 10, n)             # 0.0–10.0
    region = rng.integers(0, 3, n)                # 0, 1, or 2

    # Bounty formula: captures domain logic + noise
    bounty = (
        devil_fruit * 200
        + haki_level * 150
        + crew_size * 0.8
        + notoriety * 50
        + region * 100
        + rng.normal(0, 30, n)   # noise
    )
    bounty = np.clip(bounty, 10, None)  # minimum bounty 10M berries

    df = pd.DataFrame({
        "devil_fruit": devil_fruit,
        "haki_level": haki_level,
        "crew_size": crew_size,
        "notoriety_score": notoriety.round(2),
        "region": region,
        "bounty_berries": bounty.round(2),
    })
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("pirates.csv", index=False)

    print(f"Dataset: {len(df)} pirates")
    print(f"\nFeature stats:")
    print(df.describe())
    print(f"\nBounty range: {df['bounty_berries'].min():.0f}M – {df['bounty_berries'].max():.0f}M berries")
    print("\nSaved → pirates.csv")
