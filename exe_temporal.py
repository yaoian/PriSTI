import argparse
import datetime
import json
import logging
import os

import numpy as np
import torch
import yaml
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, random_split

from dataset_trajectory import TrajectoryImputationDataset
from temporal_model import TemporalPriSTIDiffusion
from utils import resolve_device, seed_everything

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


def _ensure_b_l_2(arr):
    t = arr if torch.is_tensor(arr) else torch.as_tensor(arr)
    if t.ndim == 2:
        if t.shape[1] == 2:
            t = t.unsqueeze(0)
        elif t.shape[0] == 2:
            t = t.transpose(0, 1).unsqueeze(0)
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


def load_raw_trajs(path):
    obj = torch.load(path, map_location="cpu")
    if torch.is_tensor(obj):
        return _ensure_b_l_2(obj)
    if isinstance(obj, np.ndarray) and getattr(obj, "dtype", None) == object:
        return _stack_list_to_b_l_2(list(obj))
    if isinstance(obj, dict):
        for key in ["trajs", "traj", "data", "loc", "loc_0", "xy", "coords"]:
            if key in obj:
                data = obj[key]
                if isinstance(data, (list, tuple)):
                    return _stack_list_to_b_l_2(data)
                return _ensure_b_l_2(data)
        fallback = _find_first_array(obj)
        if fallback is not None:
            return _ensure_b_l_2(fallback)
    if isinstance(obj, (list, tuple)):
        return _stack_list_to_b_l_2(obj)
    raise ValueError(f"unsupported data format in {path}")


def load_test_batch(path, traj_len=None):
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError("valid_file must be a dict with loc_0/mask fields")
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
    if "loc_guess" in obj:
        x_interp = _ensure_b_l_2(obj["loc_guess"])
    else:
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


def train_temporal(
    model,
    config_train,
    train_loader,
    valid_loader=None,
    validate_every_steps=1000,
    foldername="",
    tb_dir=None,
):
    optimizer = Adam(model.parameters(), lr=config_train["lr"], weight_decay=1e-6)
    is_lr_decay = config_train.get("is_lr_decay", False)
    if is_lr_decay:
        p1 = int(0.75 * config_train["epochs"])
        p2 = int(0.9 * config_train["epochs"])
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[p1, p2], gamma=0.1
        )
    writer = SummaryWriter(tb_dir) if (tb_dir and SummaryWriter is not None) else None

    output_path = os.path.join(foldername, "model.pth") if foldername else None
    if foldername:
        logging.basicConfig(filename=os.path.join(foldername, "train_model.log"), level=logging.DEBUG)

    best_valid = float("inf")
    global_step = 0
    for epoch_no in range(config_train["epochs"]):
        avg_loss = 0.0
        model.train()
        for batch_no, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            if writer:
                writer.add_scalar("train/loss", loss.item(), global_step)
            global_step += 1

            if valid_loader is not None and validate_every_steps > 0 and global_step % validate_every_steps == 0:
                valid_loss = _eval_loss(model, valid_loader)
                if writer:
                    writer.add_scalar("valid/loss", valid_loss, global_step)
                logging.info("valid_loss:%s, step:%s", valid_loss, global_step)
                if valid_loss < best_valid and foldername:
                    best_valid = valid_loss
                    torch.save(model.state_dict(), os.path.join(foldername, "best.pth"))

        if writer:
            writer.add_scalar("train/avg_epoch_loss", avg_loss / batch_no, epoch_no)
        logging.info("avg_epoch_loss:%s, epoch:%s", avg_loss / batch_no, epoch_no)
        if is_lr_decay:
            lr_scheduler.step()

    if valid_loader is not None:
        valid_loss = _eval_loss(model, valid_loader)
        if writer:
            writer.add_scalar("valid/final_loss", valid_loss, global_step)
        logging.info("valid_final_loss:%s", valid_loss)
        if valid_loss < best_valid and foldername:
            best_valid = valid_loss
            torch.save(model.state_dict(), os.path.join(foldername, "best.pth"))

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
    if data_file is None:
        if dataset_name.lower() == "xian":
            data_file = "data/trajs/Xian_nov_cache.pth"
        elif dataset_name.lower() == "chengdu":
            data_file = "data/trajs/Chengdu_nov_cache.pth"
        else:
            raise ValueError(f"unknown dataset: {dataset_name}")

    trajs = load_raw_trajs(data_file)
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
        x_gt_v, x_obs_v, mask_obs_v, x_interp_v = load_test_batch(valid_file, traj_len=int(traj_len))
        valid_dataset = BatchTrajectoryDataset(x_gt_v, x_obs_v, mask_obs_v, x_interp_v)
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

    train_temporal(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        validate_every_steps=validate_every_steps,
        foldername=foldername,
        tb_dir=tb_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Temporal-only PriSTI training")
    parser.add_argument("--config", type=str, default="trajectory.yaml")
    parser.add_argument("--dataset", type=str, default="xian", choices=["xian", "chengdu"])
    parser.add_argument("--data_file", type=str, default=None)
    parser.add_argument("--traj_len", type=int, default=None)
    parser.add_argument("--sparsity", type=float, default=None)
    parser.add_argument("--keep_mode", type=str, default=None, choices=["random", "interval"])
    parser.add_argument("--interval", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--valid_ratio", type=float, default=None)
    parser.add_argument("--validate_every_steps", type=int, default=None)
    parser.add_argument("--valid_file", type=str, default=None)
    parser.add_argument("--device", default="cuda:0", help="运行设备：cpu | cuda[:id] | auto")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    print(args)
    main(args)
