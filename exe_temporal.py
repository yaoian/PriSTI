import argparse
import datetime
import json
import logging
import os
import time
from collections import deque

import numpy as np
import torch
import yaml
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from dataset_trajectory import TrajectoryImputationDataset
from temporal_model import TemporalPriSTIDiffusion
from utils import resolve_device, seed_everything

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


class MovingAverage:
    def __init__(self, window_size):
        self.window_size = max(1, int(window_size))
        self._buf = deque(maxlen=self.window_size)

    def update(self, value):
        self._buf.append(float(value))
        return self.value

    @property
    def value(self):
        if not self._buf:
            return 0.0
        return sum(self._buf) / len(self._buf)


def _move_optimizer_state(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def _save_full_checkpoint(path, model, optimizer, scheduler, global_step, epoch, best_valid):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "global_step": int(global_step),
        "epoch": int(epoch),
        "best_valid": float(best_valid),
    }
    torch.save(ckpt, path)


def _load_full_checkpoint(path, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location=model.device)
    start_epoch = 0
    global_step = 0
    best_valid = float("inf")
    is_full = False

    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=True)
        if optimizer is not None and ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
            _move_optimizer_state(optimizer, model.device)
        if scheduler is not None and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", 0))
        global_step = int(ckpt.get("global_step", 0))
        best_valid = float(ckpt.get("best_valid", float("inf")))
        is_full = True
    else:
        model.load_state_dict(ckpt, strict=True)
    return start_epoch, global_step, best_valid, is_full


def _ensure_b_l_2(arr):
    t = arr if torch.is_tensor(arr) else torch.as_tensor(arr)
    if t.ndim == 2:
        if t.shape[1] == 2:
            t = t.unsqueeze(0)
        elif t.shape[0] == 2:
            t = t.transpose(0, 1).unsqueeze(0)
        elif t.shape[0] >= 3:
            # assume (C, L) where first two channels are x/y
            t = t[:2, :].transpose(0, 1).unsqueeze(0)
    elif t.ndim == 3 and t.shape[1] == 2:
        t = t.permute(0, 2, 1)
    if t.ndim != 3 or t.shape[-1] != 2:
        raise ValueError(f"trajectory shape unsupported: {tuple(t.shape)}")
    return t


def _find_first_array(obj):
    if torch.is_tensor(obj) or isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            found = _find_first_array(v)
            if found is not None:
                return found
    if isinstance(obj, (list, tuple)):
        for v in obj:
            found = _find_first_array(v)
            if found is not None:
                return found
    return None


def _extract_traj_from_item(item):
    if isinstance(item, dict):
        for key in ["loc", "loc_0", "traj", "trajs", "data", "xy", "coords"]:
            if key in item:
                return item[key]
        fallback = _find_first_array(item)
        if fallback is not None:
            return fallback
        raise ValueError("trajectory dict missing array-like fields")
    return item


def _stack_list_to_b_l_2(items):
    if len(items) == 0:
        raise ValueError("empty trajectory list")
    processed = []
    for x in items:
        x = _extract_traj_from_item(x)
        t = x if torch.is_tensor(x) else torch.as_tensor(x)
        t = _ensure_b_l_2(t)
        processed.append(t)
    return torch.cat(processed, dim=0)


def _linear_interpolate_np(x_gt, mask_obs):
    length = x_gt.shape[0]
    idx = np.where(mask_obs > 0)[0]
    if len(idx) == 0:
        return np.zeros_like(x_gt, dtype=np.float32)
    target_idx = np.arange(length)
    out = np.zeros_like(x_gt, dtype=np.float32)
    for d in range(x_gt.shape[1]):
        out[:, d] = np.interp(target_idx, idx, x_gt[idx, d]).astype(np.float32)
    return out


def _select_data_by_key(obj, data_key):
    if data_key is None:
        return obj
    if isinstance(obj, (list, tuple)):
        if data_key in ("a", "A"):
            return obj[0]
        if data_key in ("b", "B"):
            return obj[1] if len(obj) > 1 else obj[0]
        try:
            idx = int(data_key)
            return obj[idx]
        except Exception:
            raise ValueError(f"data_key={data_key} not valid for list/tuple")
    if isinstance(obj, dict):
        if data_key in obj:
            return obj[data_key]
        raise ValueError(f"data_key={data_key} not found in dict keys")
    return obj


def load_raw_trajs(path, data_key=None):
    obj = torch.load(path, map_location="cpu")
    # 先处理 dict 顶层
    if isinstance(obj, dict):
        if data_key is not None and data_key in obj:
            obj = obj[data_key]
            data_key = None
        else:
            for key in ["trajs", "traj", "data", "loc", "loc_0", "xy", "coords"]:
                if key in obj:
                    obj = obj[key]
                    break
    # 若指定 data_key，且 obj 为列表/对象数组，则按元素选择
    if data_key is not None:
        if isinstance(obj, np.ndarray) and getattr(obj, "dtype", None) == object:
            obj = [_select_data_by_key(item, data_key) for item in list(obj)]
        elif isinstance(obj, (list, tuple)):
            obj = [_select_data_by_key(item, data_key) for item in obj]
        else:
            obj = _select_data_by_key(obj, data_key)
    # 若未指定 data_key，但数据是 (a,b) 结构列表，默认取 a
    if data_key is None and isinstance(obj, (list, tuple)) and len(obj) > 0:
        first = obj[0]
        if isinstance(first, (list, tuple)) and len(first) >= 2:
            try:
                _ = _ensure_b_l_2(first[0])
                obj = [item[0] for item in obj]
            except Exception:
                pass
    if torch.is_tensor(obj):
        return _ensure_b_l_2(obj)
    if isinstance(obj, np.ndarray) and getattr(obj, "dtype", None) == object:
        return _stack_list_to_b_l_2(list(obj))
    if isinstance(obj, (list, tuple)):
        return _stack_list_to_b_l_2(obj)
    raise ValueError(f"unsupported data format in {path}")


def load_test_batch(path, traj_len=None):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, tuple):
        if len(obj) == 8:
            loc_0, _, loc_guess, _, mask, *_ = obj
        elif len(obj) == 10:
            loc_0, _, loc_guess, _, _, _, mask, *_ = obj
        else:
            raise ValueError(f"valid_file tuple length unsupported: {len(obj)}")
        x_gt = _ensure_b_l_2(loc_0)
        mask_t = mask.clone() if torch.is_tensor(mask) else torch.tensor(mask)
        if mask_t.ndim == 3 and mask_t.shape[1] == 1:
            mask_t = mask_t[:, 0, :]
        if mask_t.ndim != 2:
            raise ValueError(f"mask shape unsupported: {tuple(mask_t.shape)}")
        valid = mask_t >= 0
        mask_obs = ((mask_t <= 0.1) & valid).float()
        x_obs = x_gt * mask_obs.unsqueeze(-1)
        x_interp = _ensure_b_l_2(loc_guess) if loc_guess is not None else None
    elif isinstance(obj, dict):
        if "loc_0" not in obj or "mask" not in obj:
            raise ValueError("valid_file missing loc_0 or mask")
        x_gt = _ensure_b_l_2(obj["loc_0"])
        mask = obj["mask"]
        mask_t = mask.clone() if torch.is_tensor(mask) else torch.tensor(mask)
        if mask_t.ndim == 3 and mask_t.shape[1] == 1:
            mask_t = mask_t[:, 0, :]
        if mask_t.ndim != 2:
            raise ValueError(f"mask shape unsupported: {tuple(mask_t.shape)}")
        valid = mask_t >= 0
        mask_obs = ((mask_t <= 0.1) & valid).float()
        x_obs = x_gt * mask_obs.unsqueeze(-1)
        x_interp = _ensure_b_l_2(obj["loc_guess"]) if "loc_guess" in obj else None
    else:
        raise ValueError("valid_file must be dict or tuple batch")

    if x_interp is None:
        x_interp = []
        for i in range(x_gt.shape[0]):
            x_interp.append(_linear_interpolate_np(x_gt[i].numpy(), mask_obs[i].numpy()))
        x_interp = torch.tensor(np.stack(x_interp, axis=0))
    if traj_len is not None and x_gt.shape[1] != traj_len:
        x_gt = x_gt[:, :traj_len, :]
        x_obs = x_obs[:, :traj_len, :]
        mask_obs = mask_obs[:, :traj_len]
        x_interp = x_interp[:, :traj_len, :]
    return x_gt, x_obs, mask_obs, x_interp


class BatchTrajectoryDataset(Dataset):
    def __init__(self, x_gt, x_obs, mask_obs, x_interp):
        self.x_gt = x_gt
        self.x_obs = x_obs
        self.mask_obs = mask_obs
        self.x_interp = x_interp

    def __len__(self):
        return self.x_gt.shape[0]

    def __getitem__(self, index):
        return {
            "x_gt": self.x_gt[index],
            "x_obs": self.x_obs[index],
            "mask_obs": self.mask_obs[index],
            "x_interp": self.x_interp[index],
        }


def _eval_loss(model, valid_loader):
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in valid_loader:
            loss = model(batch, is_train=1)
            total += loss.item()
            count += 1
    model.train()
    return total / max(1, count)


def _make_recovery_figure(x_gt, x_pred):
    if plt is None:
        return None
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)
    gt = x_gt.reshape(-1, 2).detach().cpu().numpy()
    pr = x_pred.reshape(-1, 2).detach().cpu().numpy()
    ax.scatter(gt[:, 0], gt[:, 1], s=1, c="blue", alpha=0.6, label="gt")
    ax.scatter(pr[:, 0], pr[:, 1], s=1, c="red", alpha=0.6, label="rec")
    ax.set_aspect("equal", "box")
    ax.legend(markerscale=3, fontsize=8)
    return fig


@torch.no_grad()
def _eval_recovery(model, batch):
    model.eval()
    x_gt = batch["x_gt"].to(model.device).float()
    x_obs = batch["x_obs"].to(model.device).float()
    x_interp = batch["x_interp"].to(model.device).float()
    mask_obs = batch["mask_obs"].to(model.device).float()

    out = model.impute(x_obs, x_interp, mask_obs, return_missing_only=False)
    if isinstance(out, tuple):
        x_pred = out[0]
    else:
        x_pred = out
    mask_missing = (1.0 - mask_obs) > 0.5
    if mask_missing.any():
        mse = ((x_pred - x_gt) ** 2)[mask_missing.unsqueeze(-1).expand_as(x_gt)].mean().item()
    else:
        mse = 0.0
    recovery_loss = mse * 1000.0
    fig = _make_recovery_figure(x_gt, x_pred)
    model.train()
    return recovery_loss, fig


def train_temporal(
    model,
    config_train,
    train_loader,
    valid_loader=None,
    validate_every_steps=1000,
    foldername="",
    tb_dir=None,
    recovery_batch=None,
    resume_path=None,
):
    optimizer = Adam(model.parameters(), lr=config_train["lr"], weight_decay=1e-6)
    is_lr_decay = config_train.get("is_lr_decay", False)
    lr_scheduler = None
    if is_lr_decay:
        p1 = int(0.75 * config_train["epochs"])
        p2 = int(0.9 * config_train["epochs"])
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[p1, p2], gamma=0.1
        )
    writer = SummaryWriter(tb_dir) if (tb_dir and SummaryWriter is not None) else None
    mov_avg_interval = int(os.environ.get("TRACE_MOV_AVG_INTERVAL", config_train.get("mov_avg_interval", 100)))
    log_interval = int(os.environ.get("TRACE_TB_LOG_INTERVAL", config_train.get("log_interval", 50)))
    tb_hist_interval = int(os.environ.get("TRACE_TB_HIST_INTERVAL", "500"))
    mov_avg_loss = MovingAverage(mov_avg_interval)

    if writer and tb_dir:
        os.makedirs(tb_dir, exist_ok=True)
        writer.add_text("run/log_dir", tb_dir, 0)
        writer.add_text("run/save_dir", foldername, 0)
        info_path = os.path.join(tb_dir, "info.txt")
        config_path = os.path.join(tb_dir, "train_config.json")
        with open(info_path, "w") as f:
            f.write(f"log_dir: {tb_dir}\n")
            f.write(f"save_dir: {foldername}\n")
        with open(config_path, "w") as f:
            json.dump(config_train, f, indent=2)

    output_path = os.path.join(foldername, "model.pth") if foldername else None
    if foldername:
        logging.basicConfig(filename=os.path.join(foldername, "train_model.log"), level=logging.DEBUG)

    best_valid = float("inf")
    global_step = 0
    start_epoch = 0
    if resume_path:
        start_epoch, global_step, best_valid, is_full = _load_full_checkpoint(
            resume_path, model, optimizer=optimizer, scheduler=lr_scheduler
        )
        if is_full:
            logging.info(
                "resume_full:%s, epoch:%s, global_step:%s, best_valid:%s",
                resume_path,
                start_epoch,
                global_step,
                best_valid,
            )
        else:
            logging.info("resume_weights_only:%s", resume_path)
    if start_epoch >= config_train["epochs"]:
        logging.info("resume epoch >= total epochs, skip training")
        return

    for epoch_no in range(start_epoch, config_train["epochs"]):
        avg_loss = 0.0
        model.train()
        last_step_time = time.perf_counter()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, batch in enumerate(it, start=1):
                optimizer.zero_grad()
                need_log = writer is not None and log_interval > 0 and global_step % log_interval == 0
                need_hist = writer is not None and tb_hist_interval > 0 and global_step % tb_hist_interval == 0
                need_stats = need_log or need_hist
                if need_stats:
                    loss, stats = model(batch, return_stats=True)
                else:
                    loss = model(batch)
                    stats = None
                loss.backward()
                optimizer.step()

                loss_float = float(loss.item())
                avg_loss += loss_float
                mov_avg_loss.update(loss_float)
                if writer:
                    writer.add_scalar("train/loss", loss_float, global_step)
                global_step += 1

                it.set_postfix(
                    ordered_dict={
                        "loss": loss_float,
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )

                if writer:
                    step = global_step - 1
                if writer and log_interval > 0 and step % log_interval == 0:
                    writer.add_scalar("Loss", float(mov_avg_loss.value), step)
                    writer.add_scalar("LR", optimizer.param_groups[0]["lr"], step)
                    writer.add_scalar("Loss/mov_avg", float(mov_avg_loss.value), step)
                    writer.add_scalar("Loss/raw", float(loss_float), step)

                    for gi, group in enumerate(optimizer.param_groups):
                        if "lr" in group:
                            writer.add_scalar(f"LR/group{gi}", float(group["lr"]), step)

                    if stats is not None:
                        t = stats["t"]
                        writer.add_scalar("Diffusion/t_mean", float(t.float().mean().item()), step)
                        writer.add_scalar("Diffusion/t_min", float(t.min().item()), step)
                        writer.add_scalar("Diffusion/t_max", float(t.max().item()), step)

                        mask_obs = stats["mask_obs"]
                        valid_1d = mask_obs >= 0
                        erased_1d = mask_obs <= 0.5
                        valid_counts = valid_1d.sum(dim=1).float()
                        erased_counts = erased_1d.sum(dim=1).float()
                        erase_rate_per = torch.where(
                            valid_counts > 0,
                            erased_counts / valid_counts,
                            torch.zeros_like(valid_counts),
                        )

                        writer.add_scalar(
                            "Data/points_valid_mean", float(valid_counts.mean().item()), step
                        )
                        writer.add_scalar(
                            "Data/points_valid_median", float(valid_counts.median().item()), step
                        )
                        writer.add_scalar(
                            "Data/points_erased_mean", float(erased_counts.mean().item()), step
                        )
                        writer.add_scalar(
                            "Data/erase_rate_mean", float(erase_rate_per.mean().item()), step
                        )
                        writer.add_scalar(
                            "Data/erase_rate_median", float(erase_rate_per.median().item()), step
                        )

                        eps_pred = stats["eps_pred"]
                        eps_true = stats["eps_true"]
                        writer.add_scalar(
                            "Eps/output_abs_mean", float(eps_pred.abs().mean().item()), step
                        )
                        writer.add_scalar(
                            "Eps/target_abs_mean", float(eps_true.abs().mean().item()), step
                        )

                    step_time_ms = (time.perf_counter() - last_step_time) * 1000.0
                    last_step_time = time.perf_counter()
                    writer.add_scalar("Time/step_ms", float(step_time_ms), step)
                    if torch.cuda.is_available():
                        device = next(model.parameters()).device
                        if device.type == "cuda":
                            writer.add_scalar(
                                "CUDA/max_memory_mb",
                                float(torch.cuda.max_memory_allocated(device) / 1024 / 1024),
                                step,
                            )
                            writer.add_scalar(
                                "CUDA/reserved_memory_mb",
                                float(torch.cuda.memory_reserved(device) / 1024 / 1024),
                                step,
                            )

                if writer and stats is not None and tb_hist_interval > 0 and step % tb_hist_interval == 0:
                    mask_obs = stats["mask_obs"]
                    valid_1d = mask_obs >= 0
                    erased_1d = mask_obs <= 0.5
                    valid_counts = valid_1d.sum(dim=1).float()
                    erased_counts = erased_1d.sum(dim=1).float()
                    erase_rate_per = torch.where(
                        valid_counts > 0,
                        erased_counts / valid_counts,
                        torch.zeros_like(valid_counts),
                    )
                    writer.add_histogram("Data/sample_length", valid_counts, step)
                    writer.add_histogram("Data/erase_rate", erase_rate_per, step)

                if valid_loader is not None and validate_every_steps > 0 and global_step % validate_every_steps == 0:
                    valid_loss = _eval_loss(model, valid_loader)
                    if writer:
                        writer.add_scalar("valid/loss", valid_loss, global_step)
                    logging.info("valid_loss:%s, step:%s", valid_loss, global_step)
                    if valid_loss < best_valid and foldername:
                        best_valid = valid_loss
                        torch.save(model.state_dict(), os.path.join(foldername, "best.pth"))
                        _save_full_checkpoint(
                            os.path.join(foldername, "best_full.pth"),
                            model,
                            optimizer,
                            lr_scheduler,
                            global_step,
                            epoch_no + 1,
                            best_valid,
                        )
                    if writer and recovery_batch is not None:
                        recovery_loss, fig = _eval_recovery(model, recovery_batch)
                        writer.add_scalar("Recovery Loss", recovery_loss, global_step)
                        if fig is not None:
                            writer.add_figure("Recovery Figure", fig, global_step)
                            plt.close(fig)

        if writer:
            writer.add_scalar("train/avg_epoch_loss", avg_loss / batch_no, epoch_no)
        logging.info("avg_epoch_loss:%s, epoch:%s", avg_loss / batch_no, epoch_no)
        if is_lr_decay:
            lr_scheduler.step()
        if foldername:
            _save_full_checkpoint(
                os.path.join(foldername, "last.pth"),
                model,
                optimizer,
                lr_scheduler,
                global_step,
                epoch_no + 1,
                best_valid,
            )

    if valid_loader is not None:
        valid_loss = _eval_loss(model, valid_loader)
        if writer:
            writer.add_scalar("valid/final_loss", valid_loss, global_step)
        logging.info("valid_final_loss:%s", valid_loss)
        if valid_loss < best_valid and foldername:
            best_valid = valid_loss
            torch.save(model.state_dict(), os.path.join(foldername, "best.pth"))
            _save_full_checkpoint(
                os.path.join(foldername, "best_full.pth"),
                model,
                optimizer,
                lr_scheduler,
                global_step,
                config_train["epochs"],
                best_valid,
            )
        if writer and recovery_batch is not None:
            recovery_loss, fig = _eval_recovery(model, recovery_batch)
            writer.add_scalar("Recovery Loss", recovery_loss, global_step)
            if fig is not None:
                writer.add_figure("Recovery Figure", fig, global_step)
                plt.close(fig)
        if foldername:
            _save_full_checkpoint(
                os.path.join(foldername, "last.pth"),
                model,
                optimizer,
                lr_scheduler,
                global_step,
                config_train["epochs"],
                best_valid,
            )

    if output_path is not None:
        torch.save(model.state_dict(), output_path)
    if writer:
        writer.close()


def main(args):
    device = resolve_device(args.device)
    seed_everything(args.seed, device)

    path = "config/" + args.config
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    config["seed"] = args.seed
    config["device"] = device

    train_cfg = config.get("train", {})
    data_cfg = config.get("data", {})
    dataset_name = args.dataset or data_cfg.get("dataset", "xian")
    traj_len = args.traj_len or data_cfg.get("traj_len", 512)
    sparsity = args.sparsity if args.sparsity is not None else data_cfg.get("sparsity", 0.5)
    keep_mode = args.keep_mode or data_cfg.get("keep_mode", "random")
    interval = args.interval if args.interval is not None else data_cfg.get("interval", None)
    max_samples = args.max_samples if args.max_samples is not None else data_cfg.get("max_samples", None)
    valid_ratio = args.valid_ratio if args.valid_ratio is not None else data_cfg.get("valid_ratio", 0.1)
    validate_every_steps = args.validate_every_steps if args.validate_every_steps is not None else data_cfg.get("validate_every_steps", 1000)
    valid_file = args.valid_file or data_cfg.get("valid_file", None)
    data_file = args.data_file or data_cfg.get("data_file", None)
    data_key = args.data_key or data_cfg.get("data_key", None)
    if data_file is None:
        if dataset_name.lower() == "xian":
            data_file = "data/trajs/Xian_nov_cache.pth"
        elif dataset_name.lower() == "chengdu":
            data_file = "data/trajs/Chengdu_nov_cache.pth"
        else:
            raise ValueError(f"unknown dataset: {dataset_name}")

    trajs = load_raw_trajs(data_file, data_key=data_key)
    if max_samples is not None:
        trajs = trajs[: int(max_samples)]

    full_dataset = TrajectoryImputationDataset(
        trajs,
        window_length=int(traj_len),
        stride=int(traj_len),
        sparsity=float(sparsity),
        keep_mode=keep_mode,
        interval=interval,
        seed=args.seed,
    )

    if valid_file:
        try:
            x_gt_v, x_obs_v, mask_obs_v, x_interp_v = load_test_batch(valid_file, traj_len=int(traj_len))
            valid_dataset = BatchTrajectoryDataset(x_gt_v, x_obs_v, mask_obs_v, x_interp_v)
        except Exception as e:
            print(f"[TemporalPriSTI] valid_file fallback to raw trajs ({e})")
            valid_trajs = load_raw_trajs(valid_file, data_key=data_key)
            valid_dataset = TrajectoryImputationDataset(
                valid_trajs,
                window_length=int(traj_len),
                stride=int(traj_len),
                sparsity=float(sparsity),
                keep_mode=keep_mode,
                interval=interval,
                seed=args.seed,
            )
        train_dataset = full_dataset
    else:
        if valid_ratio and valid_ratio > 0 and len(full_dataset) > 1:
            n_valid = max(1, int(len(full_dataset) * float(valid_ratio)))
            n_train = len(full_dataset) - n_valid
            train_dataset, valid_dataset = random_split(
                full_dataset,
                [n_train, n_valid],
                generator=torch.Generator().manual_seed(args.seed),
            )
        else:
            train_dataset, valid_dataset = full_dataset, None

    train_loader = DataLoader(
        train_dataset, batch_size=config["train"]["batch_size"], shuffle=True, num_workers=args.num_workers
    )
    valid_loader = None
    if valid_dataset is not None:
        valid_loader = DataLoader(
            valid_dataset, batch_size=config["train"]["batch_size"], shuffle=False, num_workers=args.num_workers
        )

    model = TemporalPriSTIDiffusion(config, device).to(device)

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    foldername = os.path.join("./save", f"temporal_{dataset_name}_{current_time}")
    os.makedirs(foldername, exist_ok=True)
    with open(os.path.join(foldername, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    tb_dir = os.path.join("runs", f"temporal_{dataset_name}_{current_time}")
    if SummaryWriter is None:
        print("[TemporalPriSTI] tensorboard not available (missing torch.utils.tensorboard).")
        tb_dir = None
    if plt is None:
        print("[TemporalPriSTI] matplotlib not available (skip Recovery Figure).")

    recovery_batch = None
    if valid_loader is not None:
        try:
            recovery_batch = next(iter(valid_loader))
        except Exception:
            recovery_batch = None

    resume_path = args.resume or train_cfg.get("resume_from", None)

    train_temporal(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        validate_every_steps=validate_every_steps,
        foldername=foldername,
        tb_dir=tb_dir,
        recovery_batch=recovery_batch,
        resume_path=resume_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Temporal-only PriSTI training")
    parser.add_argument("--config", type=str, default="trajectory.yaml")
    parser.add_argument("--dataset", type=str, default="xian", choices=["xian", "chengdu"])
    parser.add_argument("--data_file", type=str, default=None)
    parser.add_argument("--data_key", type=str, default=None, help="For tuple/dict datasets: select a/b or index")
    parser.add_argument("--traj_len", type=int, default=None)
    parser.add_argument("--sparsity", type=float, default=None)
    parser.add_argument("--keep_mode", type=str, default=None, choices=["random", "interval"])
    parser.add_argument("--interval", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--valid_ratio", type=float, default=None)
    parser.add_argument("--validate_every_steps", type=int, default=None)
    parser.add_argument("--valid_file", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Resume full checkpoint path")
    parser.add_argument("--device", default="cuda:0", help="运行设备：cpu | cuda[:id] | auto")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    print(args)
    main(args)
