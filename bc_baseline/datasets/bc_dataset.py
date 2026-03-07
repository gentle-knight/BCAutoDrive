from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class BCDataset(Dataset):
    """
    从 Phase 2 生成的 .npz 文件读取 Behavior Cloning 数据集。

    .npz 文件格式（由 generate_bc_data.py 生成）：
        - 'obs'     : np.ndarray, shape = (N, 45)
        - 'actions' : np.ndarray, shape = (N, 2)

    本类在 __init__ 中一次性加载到内存，并转换为 torch.FloatTensor。
    """

    def __init__(self, npz_path: str):
        super().__init__()
        data = np.load(npz_path)
        if "obs" not in data or "actions" not in data:
            raise KeyError("BCDataset: .npz 必须包含键 'obs' 和 'actions'。")

        obs = data["obs"]
        actions = data["actions"]

        if obs.ndim != 2 or obs.shape[1] != 45:
            raise ValueError(f"BCDataset: obs 期望形状 (N, 45)，实际为 {obs.shape}。")
        if actions.ndim != 2 or actions.shape[1] != 2:
            raise ValueError(f"BCDataset: actions 期望形状 (N, 2)，实际为 {actions.shape}。")
        if obs.shape[0] != actions.shape[0]:
            raise ValueError(
                f"BCDataset: obs 与 actions 的样本数不一致：{obs.shape[0]} vs {actions.shape[0]}。"
            )

        self.obs = torch.as_tensor(obs, dtype=torch.float32)
        self.actions = torch.as_tensor(actions, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.obs.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        return self.obs[idx], self.actions[idx]


__all__ = ["BCDataset"]

