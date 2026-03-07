import torch
from torch import nn


class BCActor(nn.Module):
    """
    Behavior Cloning (BC) Actor 网络。

    输入：
        - obs: shape = (B, 45) 或 (45,)

    输出：
        - action: shape = (B, 2) 或 (2,)
          对应 [steering, acceleration]，并通过 Tanh 限制在 [-1, 1] 区间内。
    """

    def __init__(self, obs_dim: int = 45, hidden_dim: int = 256, action_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


__all__ = ["BCActor"]

