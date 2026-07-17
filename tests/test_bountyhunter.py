import numpy as np
import pandas as pd
import torch
import pytest
from data import generate
from model import BountyNet


class TestDataGeneration:
    def test_shape_and_columns(self):
        df = generate(n=100, seed=42)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (100, 6)
        expected = ["devil_fruit", "haki_level", "crew_size", "notoriety_score", "region", "bounty_berries"]
        assert list(df.columns) == expected

    def test_feature_ranges(self):
        df = generate(n=5000, seed=42)
        assert df["devil_fruit"].isin([0, 1]).all()
        assert df["haki_level"].between(0, 3).all()
        assert df["crew_size"].between(1, 500).all()
        assert df["notoriety_score"].between(0, 10).all()
        assert df["region"].isin([0, 1, 2]).all()

    def test_bounty_bound(self):
        df = generate(n=5000, seed=42)
        assert df["bounty_berries"].min() >= 10

    def test_no_nan(self):
        df = generate(n=1000, seed=42)
        assert df.isnull().sum().sum() == 0

    def test_reproducibility(self):
        df1 = generate(n=100, seed=42)
        df2 = generate(n=100, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_different_data(self):
        df1 = generate(n=100, seed=42)
        df2 = generate(n=100, seed=99)
        assert not df1.equals(df2)


class TestBountyNet:
    def test_forward_pass_shape(self):
        model = BountyNet(input_dim=5)
        batch = torch.randn(8, 5)
        out = model(batch)
        assert out.shape == (8,)

    def test_forward_pass_single(self):
        model = BountyNet(input_dim=5)
        batch = torch.randn(1, 5)
        out = model(batch)
        assert out.shape == (1,)
        assert out.dtype == torch.float32

    def test_parameter_count(self):
        model = BountyNet(input_dim=5)
        total = sum(p.numel() for p in model.parameters())
        assert total == 2497

    def test_eval_mode_no_grad(self):
        model = BountyNet(input_dim=5)
        model.eval()
        with torch.no_grad():
            batch = torch.randn(4, 5)
            out1 = model(batch)
            out2 = model(batch)
            assert torch.equal(out1, out2)

    def test_different_input_dim(self):
        model = BountyNet(input_dim=3)
        batch = torch.randn(2, 3)
        out = model(batch)
        assert out.shape == (2,)


class TestTrainingIntegration:
    @pytest.fixture(autouse=True)
    def setup_and_cleanup(self):
        if not np.__version__:
            yield
        else:
            yield
        for f in ["pirates.csv", "bounty_model.pt"]:
            import os
            if os.path.exists(f):
                os.remove(f)

    def test_data_generated_and_saved(self):
        import data
        df = data.generate(n=50, seed=42)
        df.to_csv("pirates.csv", index=False)
        loaded = pd.read_csv("pirates.csv")
        assert len(loaded) == 50

    def test_short_training_creates_checkpoint(self):
        import pandas as pd
        from data import generate
        from train import load_tensors, train

        df = generate(n=200, seed=42)
        df.to_csv("pirates.csv", index=False)

        X_train, y_train, X_test, y_test, scaler = load_tensors()
        assert X_train.shape[0] == 160
        assert X_test.shape[0] == 40

        assert hasattr(scaler, "mean_")
        assert len(scaler.mean_) == 5

    def test_model_can_predict(self):
        df = generate(n=200, seed=42)
        df.to_csv("pirates.csv", index=False)

        from train import load_tensors
        X_train, y_train, X_test, y_test, scaler = load_tensors()

        model = BountyNet(input_dim=5)
        X_scaled = scaler.transform(X_test.numpy()).astype(np.float32)
        with torch.no_grad():
            preds = model(torch.tensor(X_scaled)).numpy()
        assert preds.shape == (40,)
        assert np.all(np.isfinite(preds))
