"""
PyTorch model for bounty prediction.

BountyNet: 3-layer feedforward network (regression).
  Input  → 5 features
  Hidden → 64 → 32
  Output → 1 (bounty in millions)
"""

import torch
import torch.nn as nn


class BountyNet(nn.Module):
    def __init__(self, input_dim: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


if __name__ == "__main__":
    model = BountyNet()
    print(model)

    dummy = torch.randn(4, 5)  # batch of 4 pirates, 5 features
    out = model(dummy)
    print(f"\nInput shape : {dummy.shape}")
    print(f"Output shape: {out.shape}")   # (4,) — one bounty per pirate
    print(f"Sample output: {out.detach().numpy()}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params}")
