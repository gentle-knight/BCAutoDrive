from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np


def wrap_angle_rad(angle: np.ndarray) -> np.ndarray:
    """Wrap angle(s) to [-pi, pi], same form as VBD wrap_angle."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


@dataclass(frozen=True)
class VBDIKConfig:
    """
    Config for converting (accel, yaw_rate) to MetaDrive-compatible normalized actions.

    Notes:
    - VBD inverse_kinematics outputs [accel, yaw_rate] in physical units.
    - MetaDrive ScenarioEnv in this repo expects action [steering, acceleration] normalized to [-1, 1].
    """

    dt: float = 0.1
    max_acc: float = 8.0
    max_steering: float = 0.7
    wheelbase: float = 3.15  # ~0.7 * 4.5m, consistent with BCExpertEnv.InverseDynamics
    min_speed_for_steer: float = 0.1


def vbd_style_accel_yawrate_from_track_state(
    state: Dict[str, Any],
    *,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute (accel, yaw_rate, valid) from a ScenarioNet track 'state' dict.

    This matches VBD's `inverse_kinematics()` core formulas:
      speed = ||velocity_xy||
      yaw_rate = wrap(diff(heading)) / dt
      accel = diff(speed) / dt
      valid = valid[t] & valid[t+1]

    Args:
        state: track["state"] with keys: "velocity" (T,2), "heading" (T,), "valid" (T,)
        dt: seconds

    Returns:
        accel: shape (T-1,), m/s^2
        yaw_rate: shape (T-1,), rad/s
        valid: shape (T-1,), bool (both frames valid)
    """
    vel = np.asarray(state["velocity"], dtype=np.float64)
    heading = np.asarray(state["heading"], dtype=np.float64)
    valid_t = np.asarray(state["valid"], dtype=bool)

    if vel.ndim != 2 or vel.shape[1] < 2:
        raise ValueError(f"Invalid velocity shape: {vel.shape}")
    if heading.ndim != 1:
        raise ValueError(f"Invalid heading shape: {heading.shape}")
    if valid_t.ndim != 1:
        raise ValueError(f"Invalid valid shape: {valid_t.shape}")
    if not (vel.shape[0] == heading.shape[0] == valid_t.shape[0]):
        raise ValueError(
            f"Length mismatch: vel={vel.shape[0]}, heading={heading.shape[0]}, valid={valid_t.shape[0]}"
        )

    speed = np.linalg.norm(vel[:, :2], axis=1)
    accel = np.diff(speed) / float(dt)

    d_heading = np.diff(heading)
    d_heading = wrap_angle_rad(d_heading)
    yaw_rate = d_heading / float(dt)

    valid = valid_t[:-1] & valid_t[1:]
    accel = np.where(valid, accel, 0.0)
    yaw_rate = np.where(valid, yaw_rate, 0.0)
    return accel.astype(np.float32), yaw_rate.astype(np.float32), valid


def metadrive_action_from_accel_yawrate(
    accel: np.ndarray,
    yaw_rate: np.ndarray,
    speed: np.ndarray,
    *,
    cfg: VBDIKConfig,
) -> np.ndarray:
    """
    Convert physical (accel, yaw_rate) sequence to MetaDrive action sequence:
      action[t] = [steering_norm, acc_norm]

    Steering conversion uses the same bicycle relation as BCExpertEnv (rearranged):
      yaw_rate = v / L * tan(delta)  =>  delta = atan(L * yaw_rate / v)

    Args:
        accel: (T-1,) m/s^2
        yaw_rate: (T-1,) rad/s
        speed: (T-1,) m/s, recommended to use speed[t] aligned with accel/yaw_rate timestep.
        cfg: VBDIKConfig

    Returns:
        actions: (T-1, 2) float32 in [-1,1]
    """
    accel = np.asarray(accel, dtype=np.float64)
    yaw_rate = np.asarray(yaw_rate, dtype=np.float64)
    speed = np.asarray(speed, dtype=np.float64)
    if accel.shape != yaw_rate.shape or accel.shape != speed.shape:
        raise ValueError(f"Shape mismatch: accel={accel.shape}, yaw_rate={yaw_rate.shape}, speed={speed.shape}")

    v = np.maximum(speed, 0.0)
    steering = np.zeros_like(yaw_rate, dtype=np.float64)
    mask = v >= float(cfg.min_speed_for_steer)
    steering[mask] = np.arctan(float(cfg.wheelbase) * yaw_rate[mask] / np.maximum(v[mask], 1e-3))

    steering_norm = np.clip(steering / float(cfg.max_steering), -1.0, 1.0)
    acc_norm = np.clip(accel / float(cfg.max_acc), -1.0, 1.0)
    return np.stack([steering_norm, acc_norm], axis=-1).astype(np.float32)


