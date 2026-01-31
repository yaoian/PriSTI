import unittest
import numpy as np
import torch

from dataset_trajectory import TrajectoryImputationDataset
from temporal_model import TemporalPriSTIDiffusion


class TestTemporalPriSTISampling(unittest.TestCase):
    def test_sampling_keeps_observed(self):
        torch.manual_seed(0)
        np.random.seed(0)

        length = 24
        traj = np.random.randn(length, 2).astype(np.float32)
        dataset = TrajectoryImputationDataset(
            traj,
            sparsity=0.5,
            keep_mode="random",
            seed=123,
        )
        sample = dataset[0]
        x_obs = sample["x_obs"].unsqueeze(0)
        x_interp = sample["x_interp"].unsqueeze(0)
        mask_obs = sample["mask_obs"].unsqueeze(0)

        config = {
            "model": {
                "temporal_d_model": 32,
                "temporal_layers": 2,
                "temporal_ffn_mult": 2,
            },
            "diffusion": {
                "num_steps": 10,
                "beta_start": 0.0001,
                "beta_end": 0.02,
                "schedule": "linear",
            },
        }
        model = TemporalPriSTIDiffusion(config, device="cpu")
        model.eval()

        x_pred, _ = model.impute(x_obs, x_interp, mask_obs, return_missing_only=True)

        mask = mask_obs.bool().unsqueeze(-1)
        diff = (x_pred[mask] - x_obs[mask]).abs().max().item() if mask.any() else 0.0
        self.assertLess(diff, 1e-5)


if __name__ == "__main__":
    unittest.main()
