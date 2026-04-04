from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from bc_baseline.Algorithm.bc_net import BCActor
from bc_baseline.datasets.bc_dataset import BCDataset


@dataclass
class TrainConfig:
    data_path: str
    batch_size: int
    lr: float
    epochs: int
    val_ratio: float
    num_workers: int
    seed: int
    device: str
    log_dir: str
    ckpt_dir: str


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Behavior Cloning (BC) supervised training.")
    parser.add_argument("--data_path", type=str, required=True, help="Phase 2 生成的 .npz 文件路径。")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--val_ratio", type=float, default=0.2, help="验证集比例，默认 0.2（80/20 划分）。")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="训练设备：'cuda' 或 'cpu'。",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=os.path.join("bc_baseline", "outputs", "logs", "tensorboard"),
        help="TensorBoard 日志根目录。",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=os.path.join("bc_baseline", "outputs", "checkpoints"),
        help="Checkpoint 保存目录。",
    )

    args = parser.parse_args()
    if not (0.0 < args.val_ratio < 1.0):
        raise ValueError("--val_ratio 必须在 (0, 1) 之间。")

    return TrainConfig(
        data_path=args.data_path,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        log_dir=args.log_dir,
        ckpt_dir=args.ckpt_dir,
    )


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_n = 0
    for obs, actions in loader:
        obs = obs.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        pred = model(obs)
        loss = criterion(pred, actions)
        bs = int(obs.shape[0])
        total_loss += float(loss.item()) * bs
        total_n += bs
    return total_loss / max(total_n, 1)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_n = 0
    for obs, actions in loader:
        obs = obs.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)

        pred = model(obs)
        loss = criterion(pred, actions)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = int(obs.shape[0])
        total_loss += float(loss.item()) * bs
        total_n += bs

    return total_loss / max(total_n, 1)


def save_checkpoint(
    path: str,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    config: TrainConfig,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "config": config.__dict__,
        },
        path,
    )


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)

    device = torch.device(cfg.device)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    # 1) 数据集与划分（80/20）
    dataset = BCDataset(cfg.data_path)
    n_total = len(dataset)
    n_val = int(round(n_total * cfg.val_ratio))
    n_train = n_total - n_val
    if n_train <= 0 or n_val <= 0:
        raise RuntimeError(f"数据量不足以划分 train/val：total={n_total}, train={n_train}, val={n_val}")

    gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=gen)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # 2) 模型、损失、优化器
    model = BCActor(obs_dim=51, hidden_dim=256, action_dim=2).to(device)
    criterion = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # 3) TensorBoard
    run_name = time.strftime("bc_%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=os.path.join(cfg.log_dir, run_name))

    best_val_loss = float("inf")

    # 4) 训练循环
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)

        print(
            f"[BC Train] epoch {epoch:04d}/{cfg.epochs} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
        )

        # 保存 last checkpoint
        last_path = os.path.join(cfg.ckpt_dir, "last_bc.pth")
        save_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_loss=best_val_loss,
            config=cfg,
        )

        # 保存 best checkpoint（以 val loss 为准）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(cfg.ckpt_dir, "best_bc.pth")
            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_loss=best_val_loss,
                config=cfg,
            )

    writer.close()
    print(f"[BC Train] Done. best_val_loss={best_val_loss:.6f}")


if __name__ == "__main__":
    main()

