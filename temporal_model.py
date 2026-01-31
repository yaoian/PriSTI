import math
import logging
import numpy as np
import torch
import torch.nn as nn


def build_sinusoidal_position_encoding(length, dim, device):
    position = torch.arange(length, device=device).float().unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim)
    )
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


def build_timestep_embedding(timesteps, dim, device):
    if not torch.is_tensor(timesteps):
        timesteps = torch.tensor(timesteps, device=device, dtype=torch.long)
    timesteps = timesteps.to(device).long()
    half = dim // 2
    if half == 0:
        return torch.zeros(timesteps.shape[0], dim, device=device)
    freq = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=device).float() / max(1, half - 1)
    )
    args = timesteps.float().unsqueeze(1) * freq.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros(timesteps.shape[0], 1, device=device)], dim=1)
    return emb


class TemporalBlock(nn.Module):
    def __init__(self, dim, ffn_mult=4):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Linear(dim * ffn_mult, dim),
        )

    def forward(self, noisy_emb, cond_emb):
        q = self.q_proj(cond_emb)
        k = self.k_proj(cond_emb)
        v = self.v_proj(noisy_emb)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        attn = torch.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn, v)
        out = self.out_proj(out)

        noisy_emb = self.norm1(noisy_emb + out)
        ffn_out = self.ffn(noisy_emb)
        noisy_emb = self.norm2(noisy_emb + ffn_out)
        return noisy_emb


class TemporalPriSTI(nn.Module):
    """
    输入:
      - x_t: (B, L, 2)
      - cond: (B, L, 5) = concat[x_obs, x_interp, mask]
    输出:
      - eps_pred: (B, L, 2)
    """

    def __init__(self, dim=128, num_layers=4, ffn_mult=4):
        super().__init__()
        self.dim = dim
        self.noisy_emb = nn.Linear(2, dim)
        self.cond_emb = nn.Linear(5, dim)
        self.layers = nn.ModuleList(
            [TemporalBlock(dim=dim, ffn_mult=ffn_mult) for _ in range(num_layers)]
        )
        self.out_proj = nn.Linear(dim, 2)

    def forward(self, x_t, cond=None, t=None):
        if isinstance(x_t, dict):
            batch = x_t
            if "x_t" not in batch or "cond" not in batch:
                raise ValueError("TemporalPriSTI 需 batch 包含 x_t 与 cond")
            x_t = batch["x_t"]
            cond = batch["cond"]
        if cond is None:
            raise ValueError("cond 不能为空，需为 (B, L, 5)")
        if x_t.dim() != 3 or x_t.size(-1) != 2:
            raise ValueError("x_t 需为 (B, L, 2)")
        if cond.dim() != 3 or cond.size(-1) != 5:
            raise ValueError("cond 需为 (B, L, 5)")

        noisy = self.noisy_emb(x_t)
        cond = self.cond_emb(cond)

        pos = build_sinusoidal_position_encoding(noisy.size(1), self.dim, noisy.device)
        noisy = noisy + pos
        cond = cond + pos

        if t is not None:
            t_emb = build_timestep_embedding(t, self.dim, noisy.device).unsqueeze(1)
            noisy = noisy + t_emb
            cond = cond + t_emb

        for layer in self.layers:
            noisy = layer(noisy, cond)

        eps_pred = self.out_proj(noisy)
        return eps_pred


class TemporalPriSTIDiffusion(nn.Module):
    """
    复用扩散训练流程的 temporal-only 模型包装器。
    输入 batch 需包含:
      - x_gt: (B, L, 2)
      - x_obs: (B, L, 2)
      - x_interp: (B, L, 2)
      - mask_obs: (B, L)
    输出: 标量 loss
    """

    def __init__(self, config, device):
        super().__init__()
        self.device = device

        model_cfg = config.get("model", {})
        diff_cfg = config.get("diffusion", {})

        dim = model_cfg.get("temporal_d_model", 128)
        num_layers = model_cfg.get("temporal_layers", 4)
        ffn_mult = model_cfg.get("temporal_ffn_mult", 4)

        self.eps_model = TemporalPriSTI(dim=dim, num_layers=num_layers, ffn_mult=ffn_mult)

        self.num_steps = diff_cfg.get("num_steps", 100)
        beta_start = diff_cfg.get("beta_start", 0.0001)
        beta_end = diff_cfg.get("beta_end", 0.2)
        schedule = diff_cfg.get("schedule", "linear")

        if schedule == "quad":
            beta = np.linspace(beta_start ** 0.5, beta_end ** 0.5, self.num_steps) ** 2
        else:
            beta = np.linspace(beta_start, beta_end, self.num_steps)

        self.beta = beta
        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = (
            torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)
        )
        self.beta_torch = torch.tensor(self.beta).float().to(self.device)
        self.alpha_hat_torch = torch.tensor(self.alpha_hat).float().to(self.device)
        self.alpha_torch_1d = torch.tensor(self.alpha).float().to(self.device)

        self._logged_shape = False

    def forward(self, batch, is_train=1, set_t=-1, return_stats=False):
        required_keys = ["x_gt", "x_obs", "x_interp", "mask_obs"]
        missing = [k for k in required_keys if k not in batch]
        if missing:
            raise ValueError(f"TemporalPriSTIDiffusion 缺少字段: {missing}")
        x_gt = batch["x_gt"].to(self.device).float()
        x_obs = batch["x_obs"].to(self.device).float()
        x_interp = batch["x_interp"].to(self.device).float()
        mask_obs = batch["mask_obs"].to(self.device).float()

        batch_size = x_gt.size(0)
        if is_train != 1:
            if set_t < 0:
                set_t = 0
            t = (torch.ones(batch_size) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [batch_size]).to(self.device)

        current_alpha = self.alpha_torch[t]
        eps_true = torch.randn_like(x_gt)
        x_t = (current_alpha ** 0.5) * x_gt + (1.0 - current_alpha) ** 0.5 * eps_true

        cond = torch.cat([x_obs, x_interp, mask_obs.unsqueeze(-1)], dim=-1)
        eps_pred = self.eps_model(x_t, cond, t)

        mask_missing = (1.0 - mask_obs).unsqueeze(-1)
        residual = (eps_true - eps_pred) * mask_missing
        num_eval = mask_missing.sum() * x_gt.size(-1)
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)

        if not self._logged_shape:
            msg = (
                f"[TemporalPriSTI] shapes: "
                f"x_gt={tuple(x_gt.shape)}, x_obs={tuple(x_obs.shape)}, "
                f"x_interp={tuple(x_interp.shape)}, mask_obs={tuple(mask_obs.shape)}, "
                f"x_t={tuple(x_t.shape)}, cond={tuple(cond.shape)}, eps_pred={tuple(eps_pred.shape)}"
            )
            print(msg)
            logging.info(msg)
            self._logged_shape = True

        if return_stats:
            stats = {
                "t": t.detach(),
                "eps_pred": eps_pred.detach(),
                "eps_true": eps_true.detach(),
                "mask_obs": mask_obs.detach(),
            }
            return loss, stats
        return loss

    @torch.no_grad()
    def impute(self, x_obs, x_interp, mask_obs, return_missing_only=True):
        """
        反向扩散采样：
          - x_obs: (B, L, 2)
          - x_interp: (B, L, 2)
          - mask_obs: (B, L) 1=观测, 0=缺失
        返回：
          - x_pred: (B, L, 2)
          - x_missing: (B, L, 2) 仅缺失点（可选）
        """
        x_obs = x_obs.to(self.device).float()
        x_interp = x_interp.to(self.device).float()
        mask_obs = mask_obs.to(self.device).float()

        if x_obs.dim() != 3 or x_obs.size(-1) != 2:
            raise ValueError("x_obs 需为 (B, L, 2)")
        if x_interp.dim() != 3 or x_interp.size(-1) != 2:
            raise ValueError("x_interp 需为 (B, L, 2)")
        if mask_obs.dim() != 2:
            raise ValueError("mask_obs 需为 (B, L)")

        batch_size, length, _ = x_obs.shape
        cond = torch.cat([x_obs, x_interp, mask_obs.unsqueeze(-1)], dim=-1)

        x_t = torch.randn_like(x_obs)
        x_t = mask_obs.unsqueeze(-1) * x_obs + (1.0 - mask_obs.unsqueeze(-1)) * x_t

        for t in range(self.num_steps - 1, -1, -1):
            t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            eps_pred = self.eps_model(x_t, cond, t_tensor)

            alpha_hat_t = self.alpha_hat_torch[t]
            alpha_t = self.alpha_torch_1d[t]
            coeff1 = 1.0 / torch.sqrt(alpha_hat_t)
            coeff2 = (1.0 - alpha_hat_t) / torch.sqrt(1.0 - alpha_t)
            x_t = coeff1 * (x_t - coeff2 * eps_pred)

            if t > 0:
                beta_t = self.beta_torch[t]
                alpha_t_prev = self.alpha_torch_1d[t - 1]
                sigma = torch.sqrt((1.0 - alpha_t_prev) / (1.0 - alpha_t) * beta_t)
                x_t = x_t + sigma * torch.randn_like(x_t)

            x_t = mask_obs.unsqueeze(-1) * x_obs + (1.0 - mask_obs.unsqueeze(-1)) * x_t

        x_pred = x_t
        x_missing = (1.0 - mask_obs.unsqueeze(-1)) * x_pred
        if return_missing_only:
            return x_pred, x_missing
        return x_pred
