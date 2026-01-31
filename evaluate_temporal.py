import argparse
import datetime as _dt
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_trajectory import TrajectoryImputationDataset
from temporal_model import TemporalPriSTIDiffusion


def parse_list(val, cast_fn=float):
    if val is None:
        return None
    if isinstance(val, (list, tuple)):
        return [cast_fn(v) for v in val]
    if isinstance(val, str) and "," in val:
        return [cast_fn(v.strip()) for v in val.split(",") if v.strip() != ""]
    return [cast_fn(val)]


def _ensure_b_l_2(x):
    if torch.is_tensor(x):
        t = x
    else:
        t = torch.tensor(x)
    if t.ndim == 2 and t.shape[1] == 2:
        t = t.unsqueeze(0)
    elif t.ndim == 3 and t.shape[1] == 2:
        t = t.permute(0, 2, 1)
    if t.ndim != 3 or t.shape[-1] != 2:
        raise ValueError(f"trajectory shape unsupported: {tuple(t.shape)}")
    return t


def load_raw_trajs(path):
    obj = torch.load(path, map_location="cpu")
    if torch.is_tensor(obj):
        return _ensure_b_l_2(obj)
    if isinstance(obj, dict):
        for key in ["trajs", "traj", "data", "loc", "loc_0", "xy", "coords"]:
            if key in obj:
                return _ensure_b_l_2(obj[key])
    if isinstance(obj, (list, tuple)):
        return _ensure_b_l_2(torch.stack([torch.tensor(x) for x in obj], dim=0))
    raise ValueError(f"unsupported data format in {path}")


def load_test_batch(path):
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError("test file must be a dict with loc_0/mask fields")
    if "loc_0" not in obj or "mask" not in obj:
        raise ValueError("test file missing loc_0 or mask")
    loc_0 = _ensure_b_l_2(obj["loc_0"])
    mask = obj["mask"]
    if torch.is_tensor(mask):
        mask_t = mask.clone()
    else:
        mask_t = torch.tensor(mask)
    if mask_t.ndim == 3 and mask_t.shape[1] == 1:
        mask_t = mask_t[:, 0, :]
    if mask_t.ndim != 2:
        raise ValueError(f"mask shape unsupported: {tuple(mask_t.shape)}")

    valid = mask_t >= 0
    mask_obs = ((mask_t <= 0.1) & valid).float()
    x_gt = loc_0
    x_obs = x_gt * mask_obs.unsqueeze(-1)
    x_interp = None
    if "loc_guess" in obj:
        x_interp = _ensure_b_l_2(obj["loc_guess"])
    return x_gt, x_obs, mask_obs, valid.float(), x_interp


def linear_interpolate_np(x_gt, mask_obs):
    length = x_gt.shape[0]
    idx = np.where(mask_obs > 0)[0]
    if len(idx) == 0:
        return np.zeros_like(x_gt, dtype=np.float32)
    target_idx = np.arange(length)
    out = np.zeros_like(x_gt, dtype=np.float32)
    for d in range(x_gt.shape[1]):
        out[:, d] = np.interp(target_idx, idx, x_gt[idx, d]).astype(np.float32)
    return out


def jsd_2d(p, q, n_grids=64, normalize=True, bounds=None):
    if p.size == 0 or q.size == 0:
        return None
    if bounds is None:
        xmin = float(min(p[:, 0].min(), q[:, 0].min()))
        xmax = float(max(p[:, 0].max(), q[:, 0].max()))
        ymin = float(min(p[:, 1].min(), q[:, 1].min()))
        ymax = float(max(p[:, 1].max(), q[:, 1].max()))
        bounds = (xmin, xmax, ymin, ymax)
    xmin, xmax, ymin, ymax = bounds
    if xmax <= xmin or ymax <= ymin:
        return None
    hist_p, _, _ = np.histogram2d(p[:, 0], p[:, 1], bins=n_grids, range=[[xmin, xmax], [ymin, ymax]])
    hist_q, _, _ = np.histogram2d(q[:, 0], q[:, 1], bins=n_grids, range=[[xmin, xmax], [ymin, ymax]])
    if normalize:
        hist_p = hist_p / max(hist_p.sum(), 1e-12)
        hist_q = hist_q / max(hist_q.sum(), 1e-12)
    m = 0.5 * (hist_p + hist_q)
    eps = 1e-12
    kl_pm = np.sum(hist_p * np.log((hist_p + eps) / (m + eps)))
    kl_qm = np.sum(hist_q * np.log((hist_q + eps) / (m + eps)))
    return 0.5 * (kl_pm + kl_qm)


def ndtw_distance(gt, pr):
    if gt.shape[0] < 2 or pr.shape[0] < 2:
        return None
    n = gt.shape[0]
    m = pr.shape[0]
    cost = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = np.linalg.norm(gt[i - 1] - pr[j - 1])
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return cost[n, m] / max(1, n)


def build_time_index(length):
    return np.arange(length, dtype=np.float32)


def compute_metrics(x_gt, x_pred, mask_obs, valid_mask, unified_scope="erased", compute_jsd=True, compute_ndtw=True, max_ndtw=None):
    if valid_mask is None:
        valid_mask = torch.ones_like(mask_obs)
    valid = valid_mask > 0.5
    erased = (mask_obs <= 0.5) & valid
    observed = (mask_obs > 0.5) & valid

    if unified_scope == "valid":
        unified = valid
    elif unified_scope == "evalpy":
        unified = erased
    else:
        unified = erased

    def _mse(mask_1d):
        if mask_1d.sum() == 0:
            return None
        mask_2d = mask_1d.unsqueeze(-1).repeat(1, 1, 2)
        diff = (x_pred - x_gt) ** 2
        return diff[mask_2d].mean().item()

    mse_erased = _mse(erased)
    mse_valid = _mse(valid)
    mse_unified = _mse(unified)
    mse_evalpy = mse_erased

    record = {
        "points_valid": int(valid.sum().item()),
        "points_observed": int(observed.sum().item()),
        "points_erased": int(erased.sum().item()),
        "unified_scope": unified_scope,
        "points_unified": int(unified.sum().item()),
        "mse_erased_x1000": float(mse_erased * 1000.0) if mse_erased is not None else None,
        "mse_valid_x1000": float(mse_valid * 1000.0) if mse_valid is not None else None,
        "mse_evalpy_x1000": float(mse_evalpy * 1000.0) if mse_evalpy is not None else None,
        "mse_unified_x1000": float(mse_unified * 1000.0) if mse_unified is not None else None,
    }

    if compute_jsd:
        valid_2d = valid.unsqueeze(-1).repeat(1, 1, 2)
        orig_xy = x_gt[valid_2d].view(-1, 2).cpu().numpy()
        rec_xy = x_pred[valid_2d].view(-1, 2).cpu().numpy()
        jsd = jsd_2d(orig_xy, rec_xy, n_grids=64, normalize=True)
        record["jsd_grids64"] = float(jsd) if jsd is not None else None
        record["jsd_grids64_x1000"] = float(jsd * 1000.0) if jsd is not None else None

        unified_2d = unified.unsqueeze(-1).repeat(1, 1, 2)
        orig_xy_u = x_gt[unified_2d].view(-1, 2).cpu().numpy()
        rec_xy_u = x_pred[unified_2d].view(-1, 2).cpu().numpy()
        if orig_xy_u.size > 0:
            bounds = (
                float(orig_xy_u[:, 0].min()),
                float(orig_xy_u[:, 0].max()),
                float(orig_xy_u[:, 1].min()),
                float(orig_xy_u[:, 1].max()),
            )
            jsd_u = jsd_2d(orig_xy_u, rec_xy_u, n_grids=64, normalize=True, bounds=bounds)
            record["jsd_unified_grids64"] = float(jsd_u) if jsd_u is not None else None
            record["jsd_unified_grids64_x1000"] = float(jsd_u * 1000.0) if jsd_u is not None else None
        else:
            record["jsd_unified_grids64"] = None
            record["jsd_unified_grids64_x1000"] = None

    if compute_ndtw:
        ndtw_vals = []
        ndtw_u_vals = []
        B = x_gt.shape[0]
        max_n = B if max_ndtw is None else max(1, min(B, int(max_ndtw)))
        for i in range(max_n):
            v = valid[i]
            u = unified[i]
            if v.sum() >= 2:
                t_idx = build_time_index(int(v.shape[0]))
                gt_seq = torch.cat([x_gt[i], torch.tensor(t_idx).unsqueeze(1)], dim=1)[v]
                pr_seq = torch.cat([x_pred[i], torch.tensor(t_idx).unsqueeze(1)], dim=1)[v]
                nd = ndtw_distance(gt_seq.cpu().numpy(), pr_seq.cpu().numpy())
                if nd is not None:
                    ndtw_vals.append(nd)
            if u.sum() >= 2:
                t_idx = build_time_index(int(u.shape[0]))
                gt_seq = torch.cat([x_gt[i], torch.tensor(t_idx).unsqueeze(1)], dim=1)[u]
                pr_seq = torch.cat([x_pred[i], torch.tensor(t_idx).unsqueeze(1)], dim=1)[u]
                nd = ndtw_distance(gt_seq.cpu().numpy(), pr_seq.cpu().numpy())
                if nd is not None:
                    ndtw_u_vals.append(nd)
        if ndtw_vals:
            arr = np.asarray(ndtw_vals, dtype=np.float64)
            record["ndtw_mean"] = float(arr.mean())
            record["ndtw_median"] = float(np.median(arr))
            record["ndtw_n"] = int(len(arr))
            record["ndtw_mean_x1000"] = float(arr.mean() * 1000.0)
        else:
            record["ndtw_mean"] = None
            record["ndtw_median"] = None
            record["ndtw_n"] = None
            record["ndtw_mean_x1000"] = None

        if ndtw_u_vals:
            arr = np.asarray(ndtw_u_vals, dtype=np.float64)
            record["ndtw_unified_mean"] = float(arr.mean())
            record["ndtw_unified_median"] = float(np.median(arr))
            record["ndtw_unified_n"] = int(len(arr))
            record["ndtw_unified_mean_x1000"] = float(arr.mean() * 1000.0)
        else:
            record["ndtw_unified_mean"] = None
            record["ndtw_unified_median"] = None
            record["ndtw_unified_n"] = None
            record["ndtw_unified_mean_x1000"] = None

    return record


def evaluate_dataset(model, loader, device, unified_scope, compute_jsd, compute_ndtw, max_ndtw):
    x_gt_all = []
    x_pred_all = []
    mask_all = []
    valid_all = []
    for batch in loader:
        x_gt = batch["x_gt"].to(device).float()
        x_obs = batch["x_obs"].to(device).float()
        x_interp = batch["x_interp"].to(device).float()
        mask_obs = batch["mask_obs"].to(device).float()
        x_pred, _ = model.impute(x_obs, x_interp, mask_obs, return_missing_only=True)
        x_gt_all.append(x_gt.cpu())
        x_pred_all.append(x_pred.cpu())
        mask_all.append(mask_obs.cpu())
        valid_all.append(torch.ones_like(mask_obs.cpu()))
    x_gt_all = torch.cat(x_gt_all, dim=0)
    x_pred_all = torch.cat(x_pred_all, dim=0)
    mask_all = torch.cat(mask_all, dim=0)
    valid_all = torch.cat(valid_all, dim=0)
    return compute_metrics(
        x_gt_all,
        x_pred_all,
        mask_all,
        valid_all,
        unified_scope=unified_scope,
        compute_jsd=compute_jsd,
        compute_ndtw=compute_ndtw,
        max_ndtw=max_ndtw,
    )


def evaluate_test_batch(model, x_gt, x_obs, mask_obs, valid_mask, x_interp, device, unified_scope, compute_jsd, compute_ndtw, max_ndtw):
    if x_interp is None:
        x_interp = []
        for i in range(x_gt.shape[0]):
            x_interp.append(
                linear_interpolate_np(x_gt[i].numpy(), mask_obs[i].numpy())
            )
        x_interp = torch.tensor(np.stack(x_interp, axis=0))
    x_pred, _ = model.impute(
        x_obs.to(device),
        x_interp.to(device),
        mask_obs.to(device),
        return_missing_only=True,
    )
    return compute_metrics(
        x_gt.cpu(),
        x_pred.cpu(),
        mask_obs.cpu(),
        valid_mask.cpu() if valid_mask is not None else None,
        unified_scope=unified_scope,
        compute_jsd=compute_jsd,
        compute_ndtw=compute_ndtw,
        max_ndtw=max_ndtw,
    )


def build_table(records):
    headers = ["dataset", "sparsity", "L", "mse_erased_x1000", "mse_valid_x1000"]
    if any(r.get("ndtw_mean") is not None for r in records):
        headers.append("ndtw_mean_x1000")
    if any(r.get("jsd_grids64") is not None for r in records):
        headers.append("jsd_grids64_x1000")
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in records:
        row = [r.get("dataset", ""), r.get("sparsity", ""), r.get("L", "")]
        row.append(_fmt(r.get("mse_erased_x1000")))
        row.append(_fmt(r.get("mse_valid_x1000")))
        if "ndtw_mean_x1000" in headers:
            row.append(_fmt(r.get("ndtw_mean_x1000")))
        if "jsd_grids64_x1000" in headers:
            row.append(_fmt(r.get("jsd_grids64_x1000")))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _fmt(val):
    if val is None:
        return "NA"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def main():
    parser = argparse.ArgumentParser(description="Temporal-only PriSTI evaluation.")
    parser.add_argument("--dataset", type=str, default="xian", choices=["xian", "chengdu"])
    parser.add_argument("--data-file", type=str, default=None, help="Raw trajectory cache (.pth)")
    parser.add_argument("--test-file", type=str, default=None, help="Saved test batch (.pth)")
    parser.add_argument("--config", type=str, default=None, help="YAML config to init model")
    parser.add_argument("--ckpt", type=str, default=None, help="Model checkpoint state_dict")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--L", type=str, default="512", help="Trajectory length (single or comma list)")
    parser.add_argument("--sparsity", type=str, default="0.5", help="Missing ratio (single or comma list)")
    parser.add_argument("--keep-mode", type=str, default="random", choices=["random", "interval"])
    parser.add_argument("--interval", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--unified-scope", type=str, default="erased", choices=["erased", "valid", "evalpy"])
    parser.add_argument("--no-ndtw", action="store_true")
    parser.add_argument("--no-jsd", action="store_true")
    parser.add_argument("--max-ndtw", type=int, default=None)
    parser.add_argument("--out-json", type=str, default=None, help="Write JSON records to path")
    args = parser.parse_args()

    lengths = parse_list(args.L, int)
    sparsities = parse_list(args.sparsity, float)

    device = torch.device(args.device)

    if args.config is not None:
        import yaml
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "model": {
                "temporal_d_model": 128,
                "temporal_layers": 4,
                "temporal_ffn_mult": 4,
            },
            "diffusion": {
                "num_steps": 50,
                "beta_start": 0.0001,
                "beta_end": 0.2,
                "schedule": "linear",
            },
        }

    model = TemporalPriSTIDiffusion(config, device=str(device))
    if args.ckpt:
        state = torch.load(args.ckpt, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    records = []
    for L in lengths:
        for sp in sparsities:
            if args.test_file:
                x_gt, x_obs, mask_obs, valid_mask, x_interp = load_test_batch(args.test_file)
                if L is not None and x_gt.shape[1] != L:
                    x_gt = x_gt[:, :L, :]
                    x_obs = x_obs[:, :L, :]
                    mask_obs = mask_obs[:, :L]
                    valid_mask = valid_mask[:, :L]
                    if x_interp is not None:
                        x_interp = x_interp[:, :L, :]
                record = evaluate_test_batch(
                    model,
                    x_gt,
                    x_obs,
                    mask_obs,
                    valid_mask,
                    x_interp,
                    device,
                    args.unified_scope,
                    compute_jsd=not args.no_jsd,
                    compute_ndtw=not args.no_ndtw,
                    max_ndtw=args.max_ndtw,
                )
            else:
                data_file = args.data_file
                if data_file is None:
                    if args.dataset == "xian":
                        data_file = "data/trajs/Xian_nov_cache.pth"
                    else:
                        data_file = "data/trajs/Chengdu_nov_cache.pth"
                trajs = load_raw_trajs(data_file)
                if args.max_samples is not None:
                    trajs = trajs[: args.max_samples]
                dataset = TrajectoryImputationDataset(
                    trajs,
                    window_length=L,
                    stride=L,
                    sparsity=sp,
                    keep_mode=args.keep_mode,
                    interval=args.interval,
                    seed=42,
                )
                loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
                record = evaluate_dataset(
                    model,
                    loader,
                    device,
                    args.unified_scope,
                    compute_jsd=not args.no_jsd,
                    compute_ndtw=not args.no_ndtw,
                    max_ndtw=args.max_ndtw,
                )
            record["dataset"] = args.dataset
            record["sparsity"] = sp
            record["L"] = L
            record["timestamp"] = _dt.datetime.now().isoformat(timespec="seconds")
            records.append(record)

    print(build_table(records))

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
