import unittest
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_trajectory import TrajectoryImputationDataset


class TestTrajectoryImputationDataset(unittest.TestCase):
    def test_shapes_and_interp(self):
        batch_size = 4
        length = 24
        traj = np.random.randn(batch_size, length, 2).astype(np.float32)

        dataset = TrajectoryImputationDataset(
            traj,
            sparsity=0.5,
            keep_mode="random",
            seed=123,
        )

        sample = dataset[0]
        self.assertEqual(sample["x_gt"].shape, (length, 2))
        self.assertEqual(sample["x_obs"].shape, (length, 2))
        self.assertEqual(sample["mask_obs"].shape, (length,))
        self.assertEqual(sample["x_interp"].shape, (length, 2))
        self.assertFalse(torch.isnan(sample["x_interp"]).any())
        self.assertTrue(torch.all((sample["mask_obs"] == 0) | (sample["mask_obs"] == 1)))

        missing = sample["mask_obs"] == 0
        if missing.any():
            self.assertTrue(torch.all(sample["x_obs"][missing] == 0))

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        batch = next(iter(loader))
        self.assertEqual(batch["x_gt"].shape, (batch_size, length, 2))
        self.assertEqual(batch["x_obs"].shape, (batch_size, length, 2))
        self.assertEqual(batch["mask_obs"].shape, (batch_size, length))
        self.assertEqual(batch["x_interp"].shape, (batch_size, length, 2))
        self.assertFalse(torch.isnan(batch["x_interp"]).any())


if __name__ == "__main__":
    unittest.main()
