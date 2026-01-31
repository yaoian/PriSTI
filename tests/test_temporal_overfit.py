import unittest
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_trajectory import TrajectoryImputationDataset
from temporal_model import TemporalPriSTIDiffusion


class TestTemporalPriSTIOverfit(unittest.TestCase):
    def test_overfit_loss_decrease(self):
        torch.manual_seed(0)
        np.random.seed(0)

        batch_size = 8
        length = 24
        total = 32

        t = np.linspace(0, 1, length, dtype=np.float32)
        base = np.stack([t, t * 0.5], axis=1)
        traj = np.repeat(base[None, :, :], total, axis=0)

        dataset = TrajectoryImputationDataset(
            traj,
            sparsity=0.5,
            keep_mode="random",
            seed=123,
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        batch = next(iter(loader))

        config = {
            "model": {
                "temporal_d_model": 32,
                "temporal_layers": 2,
                "temporal_ffn_mult": 2,
            },
            "diffusion": {
                "num_steps": 20,
                "beta_start": 0.0001,
                "beta_end": 0.02,
                "schedule": "linear",
            },
        }
        model = TemporalPriSTIDiffusion(config, device="cpu")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        model.train()
        losses = []
        for _ in range(30):
            torch.manual_seed(0)
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        self.assertLess(losses[-1], losses[0] * 0.7)


if __name__ == "__main__":
    unittest.main()
