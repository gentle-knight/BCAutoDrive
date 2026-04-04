from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from bc_baseline.scene_labeling.interaction_utils import (
    VehicleTrack,
    circular_std,
    lane_change_score_from_positions,
    lane_merge_score_from_lateral,
    wrap_angle_rad,
    find_valid_overlap_indices,
)
from bc_baseline.scene_labeling.scene_rules import LabelingConfig


@dataclass
class TrackFeatures:
    # traffic
    num_objects: int
    num_moving_vehicles: int
    mean_vehicle_speed: float
    p90_vehicle_speed: float
    heading_dispersion: float

    # topology & interaction
    num_candidate_interactions: int
    num_lane_changes: int
    num_merge_candidates: int
    num_crossing_candidates: int
    num_turning_tracks: int
    mean_min_ttc: float
    mean_min_thw: float

    # derived
    parallel_lane_ratio: float

    def to_scene_features_updates(self) -> Dict[str, Any]:
        return {
            "num_objects": int(self.num_objects),
            "num_moving_vehicles": int(self.num_moving_vehicles),
            "mean_vehicle_speed": float(self.mean_vehicle_speed),
            "p90_vehicle_speed": float(self.p90_vehicle_speed),
            "heading_dispersion": float(self.heading_dispersion),
            "num_candidate_interactions": int(self.num_candidate_interactions),
            "num_lane_changes": int(self.num_lane_changes),
            "num_merge_candidates": int(self.num_merge_candidates),
            "num_crossing_candidates": int(self.num_crossing_candidates),
            "num_turning_tracks": int(self.num_turning_tracks),
            "mean_min_ttc": float(self.mean_min_ttc),
            "mean_min_thw": float(self.mean_min_thw),
            "parallel_lane_ratio": float(self.parallel_lane_ratio),
        }


def _wrap_heading_diff_abs(a: float, b: float) -> float:
    diff = float(a - b)
    diff = wrap_angle_rad(diff)
    return abs(diff)


def _iter_tracks(tracks_obj: Any) -> Iterable[Tuple[str, Dict[str, Any]]]:
    if isinstance(tracks_obj, dict):
        for k, v in tracks_obj.items():
            if isinstance(v, dict):
                yield str(k), v
    elif isinstance(tracks_obj, list):
        # best-effort: list items might have object_id inside metadata
        for i, item in enumerate(tracks_obj):
            if isinstance(item, dict):
                oid = item.get("object_id", i)
                yield str(oid), item


def _vehicle_track_from_state(track_id: str, track: Dict[str, Any]) -> Optional[VehicleTrack]:
    track_type = track.get("type", None)
    if isinstance(track_type, str):
        if "VEHICLE" not in track_type:
            return None
    # If type missing, still try parse state

    state = track.get("state", None)
    if not isinstance(state, dict):
        return None
    valid = state.get("valid", None)
    if valid is None:
        return None
    valid_mask = np.asarray(valid, dtype=bool)
    if valid_mask.size == 0:
        return None
    # Need minimal keys for features
    if "position" not in state or "velocity" not in state or "heading" not in state:
        return None
    return VehicleTrack(track_id=track_id, state=state, valid_mask=valid_mask)


def extract_track_features(scenario: Mapping[str, Any], config: LabelingConfig) -> TrackFeatures:
    """
    从 scenario 的 tracks 里提取交通/拓扑与交互相关特征。
    """

    tracks_obj = scenario.get("tracks", {}) or {}
    tracks_list = list(_iter_tracks(tracks_obj))
    num_objects = int(len(tracks_list))

    vehicles: List[VehicleTrack] = []
    for tid, tr in tracks_list:
        vt = _vehicle_track_from_state(tid, tr)
        if vt is not None:
            vehicles.append(vt)

    # ----- count moving vehicles + speed stats -----
    num_moving_vehicles = 0
    all_speeds: List[float] = []
    all_headings: List[float] = []
    moving_vehicle_heading_samples: List[float] = []

    lane_change_count = 0
    merge_candidate_count = 0
    turning_tracks_count = 0

    for v in vehicles:
        valid = np.asarray(v.valid_mask, dtype=bool)
        idx = np.where(valid)[0]
        if idx.size < 2:
            continue

        pos = np.asarray(v.position_seq[:, :2], dtype=np.float64)
        vel = np.asarray(v.velocity_seq[:, :2], dtype=np.float64)
        heading = np.asarray(v.heading_seq, dtype=np.float64)

        # displacement / speed
        disp = float(np.linalg.norm(pos[idx[-1]] - pos[idx[0]]))
        speeds = np.linalg.norm(vel[idx], axis=1)
        max_speed = float(np.max(speeds)) if speeds.size else 0.0

        is_static = (disp < float(config.static_displacement_threshold_m)) and (
            max_speed < float(config.static_speed_threshold_mps)
        )
        if is_static:
            continue

        num_moving_vehicles += 1
        all_speeds.extend([float(x) for x in speeds.tolist()])

        # heading samples
        all_headings.extend([float(x) for x in heading[idx].tolist()])
        moving_vehicle_heading_samples.extend([float(x) for x in heading[idx].tolist()])

        # lane changes
        is_lane_change, _, _ = lane_change_score_from_positions(
            position_seq=pos,
            heading_seq=heading,
            valid_mask=valid,
            lat_delta_threshold=float(config.lane_change_lat_delta_threshold_m),
            min_transition_frames=int(config.lane_change_min_transition_frames),
        )
        if is_lane_change:
            lane_change_count += 1

        # merge candidates (very rough)
        # compute lateral in initial heading frame
        t0 = int(idx[0])
        h0 = float(heading[t0])
        c = math.cos(-h0)
        s = math.sin(-h0)
        dx = pos - pos[t0]
        y_lat = s * dx[:, 0] + c * dx[:, 1]
        if lane_merge_score_from_lateral(
            y_lat_seq=y_lat,
            heading_seq=heading,
            valid_idx=idx,
            merge_lat_min=float(config.merge_lat_min_m),
            merge_lat_max=float(config.merge_lat_max_m),
            monotonic_ratio_threshold=float(config.merge_monotonic_ratio_threshold),
            heading_change_max_rad=float(config.merge_heading_change_max_rad),
        ):
            merge_candidate_count += 1

        # turning tracks by heading change
        d_heading = wrap_angle_diff_abs(float(heading[idx[-1]]), float(heading[idx[0]]))
        if d_heading >= float(config.turning_heading_change_min_rad):
            turning_tracks_count += 1

    mean_speed = float(np.mean(all_speeds)) if all_speeds else 0.0
    p90_speed = float(np.percentile(np.asarray(all_speeds, dtype=np.float64), 90)) if all_speeds else 0.0
    heading_disp = circular_std(np.asarray(all_headings, dtype=np.float64)) if all_headings else 0.0

    # ----- derived parallel_lane_ratio from heading alignment -----
    parallel_lane_ratio = 0.0
    if moving_vehicle_heading_samples:
        angles = np.asarray(moving_vehicle_heading_samples, dtype=np.float64)
        sin_mean = float(np.mean(np.sin(angles)))
        cos_mean = float(np.mean(np.cos(angles)))
        mean_angle = math.atan2(sin_mean, cos_mean)
        win = float(config.parallel_heading_window_rad)
        # wrap to [-pi, pi]
        diff_wrapped = np.arctan2(np.sin(angles - mean_angle), np.cos(angles - mean_angle))
        parallel_lane_ratio = float(np.mean(np.abs(diff_wrapped) <= win))

    # ----- pairwise candidate interactions -----
    # Restrict pairwise computation by moving vehicles (can still be large; keep v1 safe)
    moving_vehicles = [v for v in vehicles if not _is_vehicle_static(v, config)]
    # Candidate interactions are computed between all moving vehicles.

    cand_pairs = 0
    crossing_cands = 0
    min_ttc_list: List[float] = []
    min_thw_list: List[float] = []

    # Pre-pack sequences for speed
    packed: List[Tuple[VehicleTrack, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for v in moving_vehicles:
        idx = np.where(np.asarray(v.valid_mask, dtype=bool))[0]
        if idx.size < 2:
            continue
        pos = np.asarray(v.position_seq[:, :2], dtype=np.float64)
        vel = np.asarray(v.velocity_seq[:, :2], dtype=np.float64)
        heading = np.asarray(v.heading_seq, dtype=np.float64)
        packed.append((v, pos, vel, heading, idx))

    pair_count_limit = int(config.pairwise_max_vehicle_count)
    if len(packed) > pair_count_limit:
        packed = packed[:pair_count_limit]

    # O(N^2 * overlap) - acceptable for v1 when vehicles count is moderate.
    for i in range(len(packed)):
        _, pos_i, vel_i, h_i, _idx_i = packed[i]
        for j in range(i + 1, len(packed)):
            _, pos_j, vel_j, h_j, _idx_j = packed[j]

            valid_overlap = find_valid_overlap_indices(
                valid_a=packed[i][0].valid_mask,
                valid_b=packed[j][0].valid_mask,
            )
            if valid_overlap.size == 0:
                continue

            # Distance over valid overlap frames
            diff_pos = pos_j[valid_overlap] - pos_i[valid_overlap]
            dists = np.linalg.norm(diff_pos, axis=1)
            min_k = int(np.argmin(dists))
            dist_min = float(dists[min_k])
            t_min = int(valid_overlap[min_k])

            if dist_min > float(config.candidate_interaction_distance_m):
                continue

            cand_pairs += 1

            # Crossing candidate: heading difference around 90 deg at closest approach time
            hd = abs(wrap_angle_diff_abs(float(h_i[t_min]), float(h_j[t_min])))
            hd_deg = math.degrees(hd)
            if float(config.crossing_heading_diff_min_deg) <= hd_deg <= float(config.crossing_heading_diff_max_deg):
                crossing_cands += 1

            # min_ttc between both ordered directions (ego=i vs other=j) aggregated.
            # Use i->j as ego for THW and also for TTC.
            ttc_ij_vals: List[float] = []
            thw_ij_vals: List[float] = []
            ttc_ji_vals: List[float] = []
            thw_ji_vals: List[float] = []

            # Vectorize TTC/THW computation where possible
            idx = valid_overlap
            ego_pos = pos_i[idx]
            ego_vel = vel_i[idx]
            other_pos = pos_j[idx]
            other_vel = vel_j[idx]

            # TTC vectorized
            r = other_pos - ego_pos  # (K,2)
            dist = np.linalg.norm(r, axis=1)  # (K,)
            eps = 1e-6
            dist_safe = np.maximum(dist, eps)
            u = r / dist_safe[:, None]
            v_rel = ego_vel - other_vel  # (K,2)
            approach_rate = np.sum(v_rel * u, axis=1)  # (K,)
            closing = approach_rate > float(config.ttc_approach_rate_eps)
            ttc_vals = np.where(
                closing,
                dist / np.maximum(approach_rate, eps),
                float(config.default_ttc_s),
            )
            ttc_vals = np.clip(ttc_vals, 0.0, float(config.max_ttc_clip_s))

            min_ttc = float(np.min(ttc_vals)) if ttc_vals.size else float(config.default_ttc_s)
            min_ttc_list.append(min_ttc)

            # THW ego=i relative to other=j: only when other is in front
            ego_speed = np.linalg.norm(ego_vel, axis=1)  # (K,)
            unit_r = r / np.maximum(dist, eps)[:, None]
            ego_dir = ego_vel / np.maximum(ego_speed, eps)[:, None]
            in_front_mask = (ego_speed > float(config.thw_min_speed_mps)) & (
                np.sum(unit_r * ego_dir, axis=1) > 0.0
            )
            if np.any(in_front_mask):
                thw_vals = dist[in_front_mask] / np.maximum(ego_speed[in_front_mask], eps)
                thw_vals = np.clip(thw_vals, 0.0, float(config.max_thw_clip_s))
                min_thw = float(np.min(thw_vals))
            else:
                min_thw = float(config.default_thw_s)
            min_thw_list.append(min_thw)

            # Also compute reverse order for more coverage
            # For v1 keep only ego=i->j as above; to reduce noise, ignore reverse.

            # Track-level feature already counts lane changes/turning, so we don't update here.

    mean_min_ttc = float(np.mean(min_ttc_list)) if min_ttc_list else 0.0
    mean_min_thw = float(np.mean(min_thw_list)) if min_thw_list else 0.0

    return TrackFeatures(
        num_objects=num_objects,
        num_moving_vehicles=num_moving_vehicles,
        mean_vehicle_speed=mean_speed,
        p90_vehicle_speed=p90_speed,
        heading_dispersion=heading_disp,
        num_candidate_interactions=cand_pairs,
        num_lane_changes=lane_change_count,
        num_merge_candidates=merge_candidate_count,
        num_crossing_candidates=crossing_cands,
        num_turning_tracks=turning_tracks_count,
        mean_min_ttc=mean_min_ttc,
        mean_min_thw=mean_min_thw,
        parallel_lane_ratio=parallel_lane_ratio,
    )


def wrap_angle_diff_abs(a: float, b: float) -> float:
    diff = wrap_angle_rad(a - b)
    return abs(diff)


def _is_vehicle_static(v: VehicleTrack, config: LabelingConfig) -> bool:
    valid = np.asarray(v.valid_mask, dtype=bool)
    idx = np.where(valid)[0]
    if idx.size < 2:
        return True
    pos = np.asarray(v.position_seq[:, :2], dtype=np.float64)
    vel = np.asarray(v.velocity_seq[:, :2], dtype=np.float64)
    disp = float(np.linalg.norm(pos[idx[-1]] - pos[idx[0]]))
    speeds = np.linalg.norm(vel[idx], axis=1)
    max_speed = float(np.max(speeds)) if speeds.size else 0.0
    return (disp < float(config.static_displacement_threshold_m)) and (max_speed < float(config.static_speed_threshold_mps))

