"""
Evaluate a trained checkpoint on the saved test batch and report metrics:
- MSE on erased points (and optionally overall valid points)
- MSE on query points (eval.py style, using bool_mask)
- NDTW (average over trajectories)
- JSD between point distributions
- Write a JSON record next to the checkpoint (default behavior)

Typical usage (recommended to use venv python to avoid uv cache permission issues):
  .venv/bin/python tools/eval_checkpoint.py --ckpt Runs/<run>/best.pth --test-file Dataset/test_Xian_B100_l512_E05.pth
"""

import argparse
import datetime as _dt
import json
import os
import sys
from typing import Optional, Tuple

import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Configs import (
    Trace as DefaultTrace,
    Trace_args as default_trace_args,
    Linkage as DefaultLinkage,
    link_args as default_link_args,
    dataset_name as default_cfg_dataset_name,
    diffusion_args as cfg_diffusion_args,
)
from DDM import DDIM
from EvalUtils import JSD, NDTW, NDTW_BACKEND, NDTW_NATIVE_ERROR
from Models import (
    Embedder,
    Trace_MultiSeq_Add,
    Trace_MultiSeq_Add_Linkage,
    Trace_MultiSeq_Cat,
    Trace_MultiSeq_Cat_Linkage,
    Trace_MultiSeq_CA,
    Trace_MultiSeq_CA_Linkage,
    Trace_MultiVec_Add,
    Trace_MultiVec_Add_Linkage,
    Trace_Seq_Cat,
    Trace_Seq_Cat_Linkage,
)
from device_utils import get_default_device

MODEL_REGISTRY = {
    "Trace_MultiSeq_Add": (Trace_MultiSeq_Add, Trace_MultiSeq_Add_Linkage),
    "Trace_MultiSeq_Cat": (Trace_MultiSeq_Cat, Trace_MultiSeq_Cat_Linkage),
    "Trace_MultiSeq_CA": (Trace_MultiSeq_CA, Trace_MultiSeq_CA_Linkage),
    "Trace_SingleSeq": (Trace_Seq_Cat, Trace_Seq_Cat_Linkage),
    "Trace_MultiVec_Add": (Trace_MultiVec_Add, Trace_MultiVec_Add_Linkage),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint metrics on a saved test batch.")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint .pth (default: latest best/last)")
    parser.add_argument("--test-file", type=str, default=None, help="Path to test batch .pth (default by dataset_name)")
    parser.add_argument("--device", type=str, default=None, help="Device override, e.g. cpu/cuda")
    parser.add_argument("--max-ndtw", type=int, default=None, help="Compute NDTW on at most N trajectories")
    parser.add_argument("--no-ndtw", action="store_true", help="Skip NDTW (can be slow)")
    parser.add_argument("--no-jsd", action="store_true", help="Skip JSD")
    parser.add_argument(
        "--unified-scope",
        type=str,
        default="erased",
        choices=("erased", "valid", "evalpy"),
        help="Align MSE/JSD/NDTW on the same point subset: erased(query) / valid / evalpy(bool_mask).",
    )
    parser.add_argument(
        "--indoor-denorm",
        action="store_true",
        help=(
            "If dataset is Indoor, de-normalize x/y (z-score) before computing metrics. "
            "Requires --indoor-norm-stats (or default Dataset/Indoor_norm_stats.json)."
        ),
    )
    parser.add_argument(
        "--indoor-norm-stats",
        type=str,
        default="Dataset/Indoor_norm_stats.json",
        help="Path to Indoor z-score stats JSON: {'mean':[mx,my,mt],'std':[sx,sy,st]}.",
    )
    parser.add_argument(
        "--report-guess",
        action="store_true",
        help="Also report metrics for loc_guess baseline (time interpolation) for quick comparison.",
    )
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Do not write a JSON record file next to the checkpoint.",
    )
    parser.add_argument(
        "--record-path",
        type=str,
        default=None,
        help="Override record file path (default: create under the checkpoint directory).",
    )
    return parser.parse_args()


def resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def find_latest_ckpt() -> str:
    candidates = []
    runs_dir = os.path.join(PROJECT_ROOT, "Runs")
    for root, _, files in os.walk(runs_dir):
        if "best.pth" in files:
            candidates.append(os.path.join(root, "best.pth"))
    if candidates:
        return max(candidates, key=os.path.getmtime)
    for root, _, files in os.walk(runs_dir):
        if "last.pth" in files:
            candidates.append(os.path.join(root, "last.pth"))
    if candidates:
        return max(candidates, key=os.path.getmtime)
    raise FileNotFoundError("No checkpoint found under Runs/. Please pass --ckpt.")


def default_test_file(ds_name: str) -> str:
    if ds_name == "apartments":
        return os.path.join(PROJECT_ROOT, "Dataset/test_20240711_B100_l512_E05.pth")
    candidates = [
        os.path.join(PROJECT_ROOT, f"Dataset/test_{ds_name}_B100_l512_E05.pth"),
        os.path.join(PROJECT_ROOT, f"Dataset/test_{ds_name}_B100_l512_E0.5.pth"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    dataset_dir = os.path.join(PROJECT_ROOT, "Dataset")
    for p in sorted(Path(dataset_dir).glob(f"test_{ds_name}_B*_l512_E*.pth")):
        if p.is_file():
            return str(p)
    return candidates[0]


def _safe_stem(name: str) -> str:
    stem = os.path.splitext(os.path.basename(name))[0]
    return "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in stem)


def default_record_path(ckpt_path: str, test_file: str) -> str:
    ckpt_dir = os.path.dirname(ckpt_path)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_stem = _safe_stem(ckpt_path)
    test_stem = _safe_stem(test_file)
    return os.path.join(ckpt_dir, f"eval_{ckpt_stem}_{test_stem}_{ts}.json")


def load_train_config_for_ckpt(ckpt_path: str) -> Tuple[Optional[dict], Optional[str]]:
    ckpt_dir = os.path.dirname(ckpt_path)
    cfg_path = os.path.join(ckpt_dir, "train_config.json")
    if not os.path.isfile(cfg_path):
        return None, None
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        print(f"warn: failed to read train_config.json: {cfg_path} ({e})")
        return None, cfg_path
    return cfg, cfg_path


def _load_indoor_norm_stats(path: str) -> Tuple[float, float, float, float]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mean = data.get("mean")
    std = data.get("std")
    if not (isinstance(mean, list) and isinstance(std, list) and len(mean) >= 2 and len(std) >= 2):
        raise ValueError(f"Invalid Indoor norm stats JSON: {path}")
    mx, my = float(mean[0]), float(mean[1])
    sx, sy = float(std[0]), float(std[1])
    if sx <= 0 or sy <= 0:
        raise ValueError(f"Invalid std in Indoor norm stats JSON: {path} std={std}")
    return mx, my, sx, sy


def _maybe_denorm_xy_for_metrics(
    dataset_name: str,
    do_denorm: bool,
    stats_path: str,
    loc_0: torch.Tensor,
    loc_rec: torch.Tensor,
    loc_guess: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[dict]]:
    if dataset_name != "Indoor" or not do_denorm:
        return loc_0, loc_rec, loc_guess, None

    if not os.path.isfile(stats_path):
        raise FileNotFoundError(
            f"Indoor de-normalization requested but stats file not found: {stats_path}. "
            "Create it with keys {'mean':[mx,my,mt],'std':[sx,sy,st]}."
        )

    mx, my, sx, sy = _load_indoor_norm_stats(stats_path)
    mean = torch.tensor([mx, my], dtype=loc_0.dtype, device=loc_0.device).view(1, 2, 1)
    std = torch.tensor([sx, sy], dtype=loc_0.dtype, device=loc_0.device).view(1, 2, 1)

    loc_0_dn = loc_0 * std + mean
    loc_rec_dn = loc_rec * std + mean
    loc_guess_dn = loc_guess * std + mean
    meta = {
        "stats_path": stats_path,
        "mean_xy": [mx, my],
        "std_xy": [sx, sy],
    }
    return loc_0_dn, loc_rec_dn, loc_guess_dn, meta


def infer_ckpt_in_c(checkpoint: dict) -> int:
    unet_state = checkpoint.get("unet")
    if not isinstance(unet_state, dict):
        raise ValueError("Checkpoint missing 'unet' state_dict")
    if "pre_embed.weight" not in unet_state:
        raise ValueError("Checkpoint missing 'unet.pre_embed.weight' (cannot infer input channels)")
    w = unet_state["pre_embed.weight"]
    if not isinstance(w, torch.Tensor) or w.ndim != 3:
        raise ValueError(f"Unexpected pre_embed.weight: type={type(w)} shape={getattr(w, 'shape', None)}")
    return int(w.shape[1])


def load_checkpoint_from_state(
    checkpoint: dict,
    device: torch.device,
    traj_len: int,
    in_c: int,
    embed_dim: int,
    trace_cls: type[torch.nn.Module],
    linkage_cls: type[torch.nn.Module],
    trace_args: dict,
    linkage_args: dict,
) -> Tuple[torch.nn.Module, torch.nn.Module, Optional[torch.nn.Module]]:
    trace_args = dict(trace_args)
    trace_args["in_c"] = in_c

    unet = trace_cls(**trace_args).to(device).eval()
    linkage_shapes = unet.getFeatureShapes(traj_len) if hasattr(unet, "getFeatureShapes") else unet.getStateShapes(traj_len)
    linkage = linkage_cls(linkage_shapes, **linkage_args).to(device).eval()
    embedder = Embedder(embed_dim).to(device).eval() if embed_dim > 0 else None

    unet.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in checkpoint["unet"].items()})
    linkage.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in checkpoint["linkage"].items()})
    if embedder is not None:
        if "embedder" not in checkpoint:
            raise ValueError("Checkpoint expects embedder but 'embedder' key not found")
        embedder.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in checkpoint["embedder"].items()})
    return unet, linkage, embedder


def resolve_model_spec(train_cfg: Optional[dict]) -> tuple[type[torch.nn.Module], type[torch.nn.Module], dict, dict, str]:
    if isinstance(train_cfg, dict):
        model_name = train_cfg.get("model_name")
        if isinstance(model_name, str) and model_name in MODEL_REGISTRY:
            trace_cls, linkage_cls = MODEL_REGISTRY[model_name]
            trace_args = train_cfg.get("Trace_args")
            linkage_args = train_cfg.get("link_args")
            if isinstance(trace_args, dict) and isinstance(linkage_args, dict):
                return trace_cls, linkage_cls, trace_args, linkage_args, model_name
    return DefaultTrace, DefaultLinkage, dict(default_trace_args), dict(default_link_args), "Configs.py"


def load_test_batch(path: str):
    batch = torch.load(path, map_location="cpu")
    if not isinstance(batch, tuple):
        raise TypeError(f"Unexpected test batch type: {type(batch)}")

    if len(batch) == 10:
        # apartments: loc_0, loc_T, loc_guess, loc_mean, meta, time, mask, bool_mask, query_len, observe_len
        loc_0, loc_T, loc_guess, loc_mean, meta, time, mask, bool_mask, *_ = batch
        return True, loc_0, loc_T, loc_guess, time, mask, bool_mask, loc_mean, meta

    if len(batch) == 8:
        # taxi: loc_0, loc_T, loc_guess, time, mask, bool_mask, query_len, observe_len
        loc_0, loc_T, loc_guess, time, mask, bool_mask, *_ = batch
        loc_mean = torch.zeros(loc_0.shape[0], 2, 1, dtype=loc_0.dtype)
        meta = None
        return False, loc_0, loc_T, loc_guess, time, mask, bool_mask, loc_mean, meta

    raise ValueError(f"Unexpected test batch tuple length: {len(batch)}")


@torch.no_grad()
def recover_batch(
    unet: torch.nn.Module,
    linkage: torch.nn.Module,
    embedder: Optional[torch.nn.Module],
    ddm: DDIM,
    loc_T: torch.Tensor,
    time: torch.Tensor,
    loc_guess: torch.Tensor,
    mask: torch.Tensor,
    loc_mean: torch.Tensor,
    meta: Optional[torch.Tensor],
    uses_embedder: bool,
) -> torch.Tensor:
    B, _, L = loc_T.shape
    s_T = [torch.zeros(B, *shape, dtype=torch.float32, device=loc_T.device) for shape in unet.getStateShapes(L)]

    if uses_embedder:
        assert embedder is not None and meta is not None
        embed = embedder(meta, loc_mean)
        return ddm.diffusionBackwardWithE(unet, linkage, embed, loc_T, s_T, time, loc_guess, mask)
    return ddm.diffusionBackward(unet, linkage, loc_T, s_T, time, loc_guess, mask)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else get_default_device()

    ckpt_path = resolve_path(args.ckpt) if args.ckpt else find_latest_ckpt()

    train_cfg, train_cfg_path = load_train_config_for_ckpt(ckpt_path)
    cfg_dataset_name = default_cfg_dataset_name
    if isinstance(train_cfg, dict) and isinstance(train_cfg.get("dataset_name"), str):
        cfg_dataset_name = train_cfg["dataset_name"]

    test_file = resolve_path(args.test_file) if args.test_file else default_test_file(cfg_dataset_name)

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not os.path.isfile(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")

    batch_uses_embedder, loc_0, loc_T, loc_guess, time, mask, bool_mask, loc_mean, meta = load_test_batch(test_file)

    B = loc_0.shape[0]
    traj_len = loc_0.shape[-1]

    checkpoint = torch.load(ckpt_path, map_location=device)
    ckpt_in_c = infer_ckpt_in_c(checkpoint)
    ckpt_embed_dim = max(0, ckpt_in_c - 6)
    ckpt_uses_embedder = ckpt_embed_dim > 0
    if ckpt_uses_embedder != batch_uses_embedder:
        raise ValueError(
            "Checkpoint/test mismatch: "
            f"ckpt_in_c={ckpt_in_c} (embed_dim={ckpt_embed_dim}) "
            f"but test_batch_uses_embedder={batch_uses_embedder}. "
            "Use a matching test batch (apartments<->apartments, taxi<->taxi)."
        )

    loc_0 = loc_0.to(device)
    loc_T = loc_T.to(device)
    loc_guess = loc_guess.to(device)
    time = time.to(device)
    mask = mask.to(device)
    bool_mask = bool_mask.to(device)
    loc_mean = loc_mean.to(device)
    meta = meta.to(device) if meta is not None else None

    trace_cls, linkage_cls, trace_args, linkage_args, model_spec = resolve_model_spec(train_cfg)

    unet, linkage, embedder = load_checkpoint_from_state(
        checkpoint,
        device,
        traj_len,
        ckpt_in_c,
        ckpt_embed_dim,
        trace_cls=trace_cls,
        linkage_cls=linkage_cls,
        trace_args=trace_args,
        linkage_args=linkage_args,
    )

    diffusion_args = cfg_diffusion_args
    diffusion_args_source = "Configs.py"
    if isinstance(train_cfg, dict):
        maybe = train_cfg.get("diffusion_args")
        if isinstance(maybe, dict):
            diffusion_args = maybe
            diffusion_args_source = train_cfg_path or "train_config.json"

    ddm = DDIM(**diffusion_args, device=device)

    loc_rec = recover_batch(
        unet, linkage, embedder, ddm, loc_T, time, loc_guess, mask, loc_mean, meta, ckpt_uses_embedder
    )

    # Metric space selection (for Indoor, optionally de-normalize x/y back to original scale).
    loc_0_m, loc_rec_m, loc_guess_m, indoor_denorm_meta = _maybe_denorm_xy_for_metrics(
        dataset_name=cfg_dataset_name,
        do_denorm=bool(args.indoor_denorm),
        stats_path=resolve_path(args.indoor_norm_stats),
        loc_0=loc_0,
        loc_rec=loc_rec,
        loc_guess=loc_guess,
    )

    valid = mask[:, 0, :] >= 0
    erased = (mask[:, 0, :] > 0.1) & valid
    observed = (mask[:, 0, :] <= 0.1) & valid

    erased_2d = erased.unsqueeze(1).repeat(1, 2, 1)
    valid_2d = valid.unsqueeze(1).repeat(1, 2, 1)

    mse_erased = torch.nn.functional.mse_loss(loc_rec_m[erased_2d], loc_0_m[erased_2d]).item() * 1000.0
    mse_valid = torch.nn.functional.mse_loss(loc_rec_m[valid_2d], loc_0_m[valid_2d]).item() * 1000.0
    bool_mask_eval = bool_mask
    if bool_mask_eval.dtype != torch.bool:
        bool_mask_eval = bool_mask_eval > 0.1
    if bool_mask_eval.ndim == 2:
        bool_mask_eval = bool_mask_eval.unsqueeze(1)
    if bool_mask_eval.shape[1] == 1:
        bool_mask_eval = bool_mask_eval.repeat(1, 2, 1)
    mse_evalpy = torch.nn.functional.mse_loss(loc_rec_m[bool_mask_eval], loc_0_m[bool_mask_eval]).item() * 1000.0

    if args.unified_scope == "valid":
        unified_1d = valid
    elif args.unified_scope == "evalpy":
        unified_1d = bool_mask_eval.any(dim=1)
    else:
        unified_1d = erased
    unified_2d = unified_1d.unsqueeze(1).repeat(1, 2, 1)
    unified_points = int(unified_1d.sum().item())

    mse_unified = torch.nn.functional.mse_loss(loc_rec_m[unified_2d], loc_0_m[unified_2d]).item() * 1000.0

    mse_guess_erased = None
    mse_guess_valid = None
    mse_guess_unified = None
    if args.report_guess:
        mse_guess_erased = torch.nn.functional.mse_loss(loc_guess_m[erased_2d], loc_0_m[erased_2d]).item() * 1000.0
        mse_guess_valid = torch.nn.functional.mse_loss(loc_guess_m[valid_2d], loc_0_m[valid_2d]).item() * 1000.0
        mse_guess_unified = torch.nn.functional.mse_loss(loc_guess_m[unified_2d], loc_0_m[unified_2d]).item() * 1000.0

    record = {
        "ckpt": ckpt_path,
        "test_file": test_file,
        "dataset_hint": "apartments" if ckpt_uses_embedder else "taxi",
        "config_dataset_name": cfg_dataset_name,
        "model_name": model_spec,
        "device": str(device),
        "diffusion_args": diffusion_args,
        "diffusion_args_source": diffusion_args_source,
        "B": int(B),
        "traj_len": int(traj_len),
        "ckpt_in_c": int(ckpt_in_c),
        "ndtw_backend": NDTW_BACKEND,
        "ndtw_native_error": NDTW_NATIVE_ERROR,
        "metrics_space": "denorm_xy" if indoor_denorm_meta is not None else "normalized_xy",
        "indoor_denorm": indoor_denorm_meta,
        "points_valid": int(valid.sum().item()),
        "points_observed": int(observed.sum().item()),
        "points_erased": int(erased.sum().item()),
        "unified_scope": args.unified_scope,
        "points_unified": int(unified_points),
        "mse_erased_x1000": float(mse_erased),
        "mse_valid_x1000": float(mse_valid),
        "mse_evalpy_x1000": float(mse_evalpy),
        "mse_unified_x1000": float(mse_unified),
        "mse_guess_erased_x1000": float(mse_guess_erased) if mse_guess_erased is not None else None,
        "mse_guess_valid_x1000": float(mse_guess_valid) if mse_guess_valid is not None else None,
        "mse_guess_unified_x1000": float(mse_guess_unified) if mse_guess_unified is not None else None,
        "jsd_grids64": None,
        "jsd_grids64_x1000": None,
        "jsd_guess_grids64": None,
        "jsd_guess_grids64_x1000": None,
        "ndtw_mean": None,
        "ndtw_median": None,
        "ndtw_n": None,
        "ndtw_mean_x1000": None,
        "ndtw_guess_mean": None,
        "ndtw_guess_median": None,
        "ndtw_guess_n": None,
        "ndtw_guess_mean_x1000": None,
        "jsd_unified_grids64": None,
        "jsd_unified_grids64_x1000": None,
        "jsd_guess_unified_grids64": None,
        "jsd_guess_unified_grids64_x1000": None,
        "ndtw_unified_mean": None,
        "ndtw_unified_median": None,
        "ndtw_unified_n": None,
        "ndtw_unified_mean_x1000": None,
        "ndtw_guess_unified_mean": None,
        "ndtw_guess_unified_median": None,
        "ndtw_guess_unified_n": None,
        "ndtw_guess_unified_mean_x1000": None,
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
    }

    print(f"ckpt={ckpt_path}")
    print(f"test_file={test_file}")
    ds_name = "apartments" if ckpt_uses_embedder else "taxi"
    print(
        f"dataset_hint={ds_name} config_dataset_name={cfg_dataset_name} "
        f"device={device} B={B} traj_len={traj_len} ckpt_in_c={ckpt_in_c}"
    )
    print(f"model_name={model_spec}")
    print(f"diffusion_args_source={diffusion_args_source}")
    print(f"diffusion_args={diffusion_args}")
    print(f"points_valid={int(valid.sum().item())} points_observed={int(observed.sum().item())} points_erased={int(erased.sum().item())}")
    print(f"unified_scope={args.unified_scope} points_unified={unified_points}")
    print(f"ndtw_backend={NDTW_BACKEND}")
    if NDTW_BACKEND != "native" and NDTW_NATIVE_ERROR:
        print(f"ndtw_native_error={NDTW_NATIVE_ERROR}")
    if indoor_denorm_meta is not None:
        print(f"metrics_space=denorm_xy indoor_norm_stats={indoor_denorm_meta['stats_path']}")
    print(f"mse_erased_x1000={mse_erased:.6f}")
    print(f"mse_valid_x1000={mse_valid:.6f}")
    print(f"mse_evalpy_x1000={mse_evalpy:.6f}")
    print(f"mse_unified_x1000={mse_unified:.6f}")
    if args.report_guess:
        assert mse_guess_erased is not None and mse_guess_valid is not None
        print(f"mse_guess_erased_x1000={mse_guess_erased:.6f}")
        print(f"mse_guess_valid_x1000={mse_guess_valid:.6f}")
        assert mse_guess_unified is not None
        print(f"mse_guess_unified_x1000={mse_guess_unified:.6f}")

    if not args.no_jsd:
        orig_xy = loc_0_m[valid_2d].view(-1, 2).detach().cpu()
        rec_xy = loc_rec_m[valid_2d].view(-1, 2).detach().cpu()
        jsd = JSD(orig_xy, rec_xy, n_grids=64, normalize=True)
        record["jsd_grids64"] = float(jsd)
        record["jsd_grids64_x1000"] = float(jsd * 1000.0)
        print(f"jsd_grids64={jsd:.8f}")
        print(f"jsd_grids64_x1000={jsd * 1000.0:.6f}")
        if args.report_guess:
            guess_xy = loc_guess_m[valid_2d].view(-1, 2).detach().cpu()
            jsd_guess = JSD(orig_xy, guess_xy, n_grids=64, normalize=True)
            record["jsd_guess_grids64"] = float(jsd_guess)
            record["jsd_guess_grids64_x1000"] = float(jsd_guess * 1000.0)
            print(f"jsd_guess_grids64={jsd_guess:.8f}")
            print(f"jsd_guess_grids64_x1000={jsd_guess * 1000.0:.6f}")

        # Unified-scope JSD (aligned grid bounds based on GT only)
        orig_xy_u = loc_0_m[unified_2d].view(-1, 2).detach().cpu()
        rec_xy_u = loc_rec_m[unified_2d].view(-1, 2).detach().cpu()
        if orig_xy_u.numel() > 0:
            bounds = (
                float(orig_xy_u[:, 0].min().item()),
                float(orig_xy_u[:, 0].max().item()),
                float(orig_xy_u[:, 1].min().item()),
                float(orig_xy_u[:, 1].max().item()),
            )
            jsd_u = JSD(orig_xy_u, rec_xy_u, n_grids=64, normalize=True, bounds=bounds)
            record["jsd_unified_grids64"] = float(jsd_u)
            record["jsd_unified_grids64_x1000"] = float(jsd_u * 1000.0)
            print(f"jsd_unified_grids64={jsd_u:.8f}")
            print(f"jsd_unified_grids64_x1000={jsd_u * 1000.0:.6f}")
            if args.report_guess:
                guess_xy_u = loc_guess_m[unified_2d].view(-1, 2).detach().cpu()
                jsd_guess_u = JSD(orig_xy_u, guess_xy_u, n_grids=64, normalize=True, bounds=bounds)
                record["jsd_guess_unified_grids64"] = float(jsd_guess_u)
                record["jsd_guess_unified_grids64_x1000"] = float(jsd_guess_u * 1000.0)
                print(f"jsd_guess_unified_grids64={jsd_guess_u:.8f}")
                print(f"jsd_guess_unified_grids64_x1000={jsd_guess_u * 1000.0:.6f}")
        else:
            print("jsd_unified: no points to evaluate")

    if not args.no_ndtw:
        max_n = args.max_ndtw if args.max_ndtw is not None else B
        max_n = max(1, min(B, int(max_n)))
        ndtw_values = []
        ndtw_guess_values = [] if args.report_guess else None
        loc_0_cpu = loc_0_m.detach().cpu()
        loc_rec_cpu = loc_rec_m.detach().cpu()
        loc_guess_cpu = loc_guess_m.detach().cpu() if args.report_guess else None
        time_cpu = time.detach().cpu()
        valid_cpu = valid.detach().cpu()
        unified_cpu = unified_1d.detach().cpu()
        for i in range(max_n):
            vi = valid_cpu[i]
            n = int(vi.sum().item())
            if n < 2:
                continue
            gt = torch.cat([loc_0_cpu[i, :, vi], time_cpu[i, :, vi]], dim=0)  # (3, n)
            pr = torch.cat([loc_rec_cpu[i, :, vi], time_cpu[i, :, vi]], dim=0)  # (3, n)
            ndtw_values.append(NDTW(gt, pr))
            if args.report_guess:
                assert ndtw_guess_values is not None and loc_guess_cpu is not None
                pr_guess = torch.cat([loc_guess_cpu[i, :, vi], time_cpu[i, :, vi]], dim=0)  # (3, n)
                ndtw_guess_values.append(NDTW(gt, pr_guess))
        if ndtw_values:
            arr = np.asarray(ndtw_values, dtype=np.float64)
            record["ndtw_mean"] = float(arr.mean())
            record["ndtw_median"] = float(np.median(arr))
            record["ndtw_n"] = int(len(arr))
            record["ndtw_mean_x1000"] = float(arr.mean() * 1000.0)
            print(f"ndtw_mean={arr.mean():.6f} ndtw_median={np.median(arr):.6f} ndtw_n={len(arr)}")
            print(f"ndtw_mean_x1000={arr.mean() * 1000.0:.6f}")
            if args.report_guess and ndtw_guess_values:
                arr_g = np.asarray(ndtw_guess_values, dtype=np.float64)
                record["ndtw_guess_mean"] = float(arr_g.mean())
                record["ndtw_guess_median"] = float(np.median(arr_g))
                record["ndtw_guess_n"] = int(len(arr_g))
                record["ndtw_guess_mean_x1000"] = float(arr_g.mean() * 1000.0)
                print(f"ndtw_guess_mean={arr_g.mean():.6f} ndtw_guess_median={np.median(arr_g):.6f} ndtw_guess_n={len(arr_g)}")
                print(f"ndtw_guess_mean_x1000={arr_g.mean() * 1000.0:.6f}")
        else:
            print("ndtw: no valid trajectories to evaluate")

        # Unified-scope NDTW
        ndtw_u_values = []
        ndtw_guess_u_values = [] if args.report_guess else None
        for i in range(max_n):
            mi = unified_cpu[i]
            n = int(mi.sum().item())
            if n < 2:
                continue
            gt = torch.cat([loc_0_cpu[i, :, mi], time_cpu[i, :, mi]], dim=0)  # (3, n)
            pr = torch.cat([loc_rec_cpu[i, :, mi], time_cpu[i, :, mi]], dim=0)  # (3, n)
            ndtw_u_values.append(NDTW(gt, pr))
            if args.report_guess:
                assert ndtw_guess_u_values is not None and loc_guess_cpu is not None
                pr_guess = torch.cat([loc_guess_cpu[i, :, mi], time_cpu[i, :, mi]], dim=0)  # (3, n)
                ndtw_guess_u_values.append(NDTW(gt, pr_guess))

        if ndtw_u_values:
            arr_u = np.asarray(ndtw_u_values, dtype=np.float64)
            record["ndtw_unified_mean"] = float(arr_u.mean())
            record["ndtw_unified_median"] = float(np.median(arr_u))
            record["ndtw_unified_n"] = int(len(arr_u))
            record["ndtw_unified_mean_x1000"] = float(arr_u.mean() * 1000.0)
            print(f"ndtw_unified_mean={arr_u.mean():.6f} ndtw_unified_median={np.median(arr_u):.6f} ndtw_unified_n={len(arr_u)}")
            print(f"ndtw_unified_mean_x1000={arr_u.mean() * 1000.0:.6f}")
            if args.report_guess and ndtw_guess_u_values:
                arr_gu = np.asarray(ndtw_guess_u_values, dtype=np.float64)
                record["ndtw_guess_unified_mean"] = float(arr_gu.mean())
                record["ndtw_guess_unified_median"] = float(np.median(arr_gu))
                record["ndtw_guess_unified_n"] = int(len(arr_gu))
                record["ndtw_guess_unified_mean_x1000"] = float(arr_gu.mean() * 1000.0)
                print(
                    f"ndtw_guess_unified_mean={arr_gu.mean():.6f} "
                    f"ndtw_guess_unified_median={np.median(arr_gu):.6f} "
                    f"ndtw_guess_unified_n={len(arr_gu)}"
                )
                print(f"ndtw_guess_unified_mean_x1000={arr_gu.mean() * 1000.0:.6f}")
        else:
            print("ndtw_unified: no trajectories to evaluate")

    if not args.no_record:
        out_path = resolve_path(args.record_path) if args.record_path else default_record_path(ckpt_path, test_file)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        print(f"record_saved={out_path}")


if __name__ == "__main__":
    main()
