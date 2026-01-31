import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryImputationDataset(Dataset):
    """
    轨迹插补数据集包装器。

    每个样本输出字段与 shape：
    - x_gt: (L, 2)
    - x_obs: (L, 2)  # 缺失处置 0
    - mask_obs: (L,)  # 1=观测, 0=缺失
    - x_interp: (L, 2)  # 基于观测点按时间线性插值（边界外推）
    """

    def __init__(
        self,
        trajectories,
        window_length=None,
        stride=None,
        sparsity=0.0,
        keep_mode="random",
        interval=None,
        seed=None,
    ):
        if sparsity < 0 or sparsity > 1:
            raise ValueError("sparsity 需在 [0, 1] 范围内")
        if keep_mode not in {"random", "interval"}:
            raise ValueError("keep_mode 仅支持 'random' 或 'interval'")

        self.sparsity = float(sparsity)
        self.keep_mode = keep_mode
        self.interval = interval
        self.seed = seed

        sequences = self._normalize_trajectories(trajectories)
        self.samples = self._window_sequences(sequences, window_length, stride)

    @staticmethod
    def _normalize_trajectories(trajectories):
        if isinstance(trajectories, (list, tuple)) and not isinstance(trajectories, np.ndarray):
            sequences = [np.asarray(t, dtype=np.float32) for t in trajectories]
        else:
            arr = np.asarray(trajectories, dtype=np.float32)
            if arr.ndim == 2:
                sequences = [arr]
            elif arr.ndim == 3:
                sequences = [arr[i] for i in range(arr.shape[0])]
            else:
                raise ValueError("trajectories 需为 (L,2) 或 (N,L,2)")

        for seq in sequences:
            if seq.ndim != 2 or seq.shape[1] != 2:
                raise ValueError("每条轨迹需为形状 (L,2)")
        return sequences

    @staticmethod
    def _window_sequences(sequences, window_length, stride):
        if window_length is None:
            return [np.asarray(seq, dtype=np.float32) for seq in sequences]
        if window_length <= 0:
            raise ValueError("window_length 必须为正整数")

        stride = window_length if stride is None else stride
        if stride <= 0:
            raise ValueError("stride 必须为正整数")

        windows = []
        for seq in sequences:
            if seq.shape[0] < window_length:
                raise ValueError("序列长度不足以窗口化到指定长度")
            for start in range(0, seq.shape[0] - window_length + 1, stride):
                windows.append(seq[start : start + window_length])
        return windows

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        x_gt = np.asarray(self.samples[index], dtype=np.float32)
        length = x_gt.shape[0]
        mask_obs = self._make_mask(length, index)
        x_obs = x_gt * mask_obs[:, None]
        x_interp = self._linear_interpolate(x_gt, mask_obs)

        return {
            "x_gt": torch.from_numpy(x_gt),
            "x_obs": torch.from_numpy(x_obs),
            "mask_obs": torch.from_numpy(mask_obs),
            "x_interp": torch.from_numpy(x_interp),
        }

    def _make_mask(self, length, index):
        keep_prob = 1.0 - self.sparsity
        rng = np.random.default_rng(self.seed + index if self.seed is not None else None)

        if self.keep_mode == "random":
            mask = (rng.random(length) < keep_prob).astype(np.float32)
        else:
            interval = self.interval
            if interval is None:
                if keep_prob <= 0:
                    interval = length + 1
                else:
                    interval = max(1, int(round(1.0 / keep_prob)))
            mask = np.zeros(length, dtype=np.float32)
            mask[::interval] = 1.0

        if mask.sum() == 0:
            mask[rng.integers(0, length)] = 1.0
        return mask

    @staticmethod
    def _linear_interpolate(x_gt, mask_obs):
        length = x_gt.shape[0]
        idx = np.where(mask_obs > 0)[0]
        x_interp = np.zeros_like(x_gt, dtype=np.float32)
        if len(idx) == 0:
            return x_interp
        target_idx = np.arange(length)
        for d in range(x_gt.shape[1]):
            x_interp[:, d] = np.interp(target_idx, idx, x_gt[idx, d]).astype(np.float32)
        return x_interp
