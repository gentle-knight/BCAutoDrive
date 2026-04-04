from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def wrap_angle_rad(angle: float) -> float:
    return float(math.atan2(math.sin(angle), math.cos(angle)))


def circular_std(angles_rad: np.ndarray) -> float:
    """
    Circular standard deviation (rad).
    """
    angles_rad = np.asarray(angles_rad, dtype=np.float64)
    if angles_rad.size == 0:
        return 0.0
    sin_mean = float(np.mean(np.sin(angles_rad)))
    cos_mean = float(np.mean(np.cos(angles_rad)))
    R = math.sqrt(sin_mean * sin_mean + cos_mean * cos_mean)
    if R < 1e-8:
        # Near uniform, std approaches pi/sqrt(3) (~1.81), but keep safe numeric bound.
        return float(math.pi)
    std = math.sqrt(max(0.0, -2.0 * math.log(R)))
    return float(std)


def compute_ttc_between(
    ego_pos: np.ndarray,
    ego_vel: np.ndarray,
    other_pos: np.ndarray,
    other_vel: np.ndarray,
    *,
    default_ttc: float = 60.0,
    max_ttc_clip: float = 60.0,
    eps: float = 1e-6,
) -> float:
    """
    TTC: approx dist / closing_rate along the line-of-sight direction.
    """
    ego_pos = np.asarray(ego_pos, dtype=np.float64)
    other_pos = np.asarray(other_pos, dtype=np.float64)
    ego_vel = np.asarray(ego_vel, dtype=np.float64)
    other_vel = np.asarray(other_vel, dtype=np.float64)

    r = other_pos - ego_pos
    dist = float(np.linalg.norm(r))
    if dist < eps:
        return 0.0

    u = r / dist
    v_rel = ego_vel - other_vel
    approach_rate = float(np.dot(v_rel, u))
    if approach_rate <= 1e-4:
        return float(default_ttc)

    ttc = dist / approach_rate
    ttc = float(max(0.0, min(float(max_ttc_clip), ttc)))
    return ttc


def compute_thw_between(
    ego_pos: np.ndarray,
    ego_vel: np.ndarray,
    other_pos: np.ndarray,
    other_vel: np.ndarray,
    *,
    default_thw: float = 10.0,
    max_thw_clip: float = 10.0,
    thw_min_speed: float = 1.0,
    eps: float = 1e-6,
) -> float:
    """
    THW (time headway) in a simple form:
      - If other is roughly in front of ego (positive projection), use distance / ego_speed.
      - Otherwise return default_thw.
    """
    ego_pos = np.asarray(ego_pos, dtype=np.float64)
    other_pos = np.asarray(other_pos, dtype=np.float64)
    ego_vel = np.asarray(ego_vel, dtype=np.float64)

    r = other_pos - ego_pos
    dist = float(np.linalg.norm(r))
    if dist < eps:
        return 0.0

    ego_speed = float(np.linalg.norm(ego_vel))
    if ego_speed < thw_min_speed:
        return float(default_thw)

    ego_dir = ego_vel / max(ego_speed, eps)
    unit_r = r / dist
    in_front = float(np.dot(unit_r, ego_dir)) > 0.0
    if not in_front:
        return float(default_thw)

    thw = dist / ego_speed
    thw = float(max(0.0, min(float(max_thw_clip), thw)))
    return thw


def min_over_time_masked(
    values: List[float],
    *,
    default_value: float,
) -> float:
    if not values:
        return float(default_value)
    return float(min(values))


def find_valid_overlap_indices(valid_a: np.ndarray, valid_b: np.ndarray) -> np.ndarray:
    valid_a = np.asarray(valid_a, dtype=bool)
    valid_b = np.asarray(valid_b, dtype=bool)
    T = min(valid_a.shape[0], valid_b.shape[0])
    if T <= 0:
        return np.zeros((0,), dtype=int)
    mask = valid_a[:T] & valid_b[:T]
    idx = np.where(mask)[0]
    return idx


@dataclass
class VehicleTrack:
    track_id: str
    state: Dict[str, Any]
    valid_mask: np.ndarray

    @property
    def position_seq(self) -> np.ndarray:
        return np.asarray(self.state["position"], dtype=np.float64)

    @property
    def velocity_seq(self) -> np.ndarray:
        return np.asarray(self.state["velocity"], dtype=np.float64)

    @property
    def heading_seq(self) -> np.ndarray:
        return np.asarray(self.state["heading"], dtype=np.float64)


def lane_change_score_from_positions(
    position_seq: np.ndarray,
    heading_seq: np.ndarray,
    valid_mask: np.ndarray,
    *,
    lat_delta_threshold: float = 1.0,
    min_transition_frames: int = 5,
) -> Tuple[bool, float, float]:
    """
    Approx lane-change by lateral drift in ego-initial heading frame.
    Returns (is_lane_change, total_lat_change, peak_abs_lat).
    """
    valid_mask = np.asarray(valid_mask, dtype=bool)
    idx = np.where(valid_mask)[0]
    if idx.size < 2:
        return False, 0.0, 0.0

    t0 = int(idx[0])
    h0 = float(heading_seq[t0])
    c = math.cos(-h0)
    s = math.sin(-h0)

    pos = np.asarray(position_seq[:, :2], dtype=np.float64)
    dx = pos - pos[t0]
    # rotate world delta into initial-heading frame
    x_ego = c * dx[:, 0] - s * dx[:, 1]
    y_ego = s * dx[:, 0] + c * dx[:, 1]

    y = y_ego[idx]
    lat_change = float(y[-1] - y[0])
    total_lat = float(np.max(y) - np.min(y))
    peak_abs_lat = float(np.max(np.abs(y)))

    # Estimate transition duration: count frames where abs(lat - median) exceeds threshold
    med = float(np.median(y))
    trans_frames = int(np.sum(np.abs(y - med) >= float(lat_delta_threshold)))
    is_change = (abs(lat_change) >= float(lat_delta_threshold)) and (trans_frames >= int(min_transition_frames))
    return bool(is_change), total_lat, peak_abs_lat


def lane_merge_score_from_lateral(
    y_lat_seq: np.ndarray,
    heading_seq: np.ndarray,
    valid_idx: np.ndarray,
    *,
    merge_lat_min: float = 1.0,
    merge_lat_max: float = 4.0,
    monotonic_ratio_threshold: float = 0.7,
    heading_change_max_rad: float = math.radians(15.0),
) -> bool:
    """
    Very rough merge score from monotonic lateral movement with small heading change.
    """
    if valid_idx.size < 3:
        return False

    y = np.asarray(y_lat_seq, dtype=np.float64)[valid_idx]
    if y.size < 3:
        return False

    lat_total = float(y[-1] - y[0])
    if abs(lat_total) < float(merge_lat_min) or abs(lat_total) > float(merge_lat_max):
        return False

    dy = np.diff(y)
    if dy.size == 0:
        return False

    trend = float(np.sign(lat_total))
    if trend == 0.0:
        return False
    good = np.sum(np.sign(dy) == trend)
    monotonic_ratio = float(good) / float(dy.size)
    if monotonic_ratio < float(monotonic_ratio_threshold):
        return False

    # heading change
    h_start = float(heading_seq[int(valid_idx[0])])
    h_end = float(heading_seq[int(valid_idx[-1])])
    d_heading = abs(wrap_angle_rad(h_end - h_start))
    if d_heading > float(heading_change_max_rad):
        return False

    return True

