from __future__ import annotations

"""
extract_interaction_episodes.py
--------------------------------

从 Waymo/ScenarioNet 轨迹中自动提取“强交互 episode”，
并为每个 episode 计算 8 维驾驶风格特征向量，用于后续聚类。

用法示例：

    python -m bc_baseline.scripts.extract_interaction_episodes \\
        --waymo_dir /path/to/waymo_pkls \\
        --num_scenarios 100
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from bc_baseline.Env.expert_env import BCExpertEnv, _TrackMeta

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# -------------------------- 常量与特征定义 -------------------------- #

DEFAULT_TTC: float = 60.0
DEFAULT_PET: float = 99.0

FEATURE_NAMES: List[str] = [
    "mean_speed",
    "std_speed",
    "max_speed",
    "mean_acc",
    "min_acc",
    "jerk_peak",
    "min_ttc",
    "min_pet",
]
FEATURE_DIM: int = len(FEATURE_NAMES)

MAX_TTC_CLIP: float = 60.0
MAX_PET_CLIP: float = 99.0


@dataclass
class InteractionEpisode:
    """
    表示一次 ego 与某辆对手车之间的“交互 episode”。

    属性说明：
        scenario_index : 当前场景在数据集中的索引。
        ego_track_id   : 作为自车的轨迹 ID。
        partner_track_id: 触发该交互的对手车轨迹 ID。
        t_peak         : 该 episode 中风险峰值帧（TTC 最小）的全局时间索引。
        t_start        : episode 起始帧索引（包含）。
        t_end          : episode 结束帧索引（包含）。
        min_ttc        : 该 episode 窗口内 ego 与对手车的最小 TTC（秒）。
        min_pet        : 该 episode 窗口内 ego 与对手车的最小 PET（秒）。
        features       : shape (8,) 的特征向量，顺序参见 FEATURE_NAMES。
    """

    scenario_index: int
    ego_track_id: Any
    partner_track_id: Any
    t_peak: int
    t_start: int
    t_end: int
    min_ttc: float
    min_pet: float
    features: np.ndarray


@dataclass
class EpisodeConfig:
    """
    Episode 提取的超参数配置。

    属性：
        ttc_threshold    : 认为发生强交互的 TTC 阈值（秒）。
        window_seconds   : 以风险峰值为中心的对称时间窗口长度（秒）。
        min_episode_frames: 最小有效帧数（不足则丢弃该 episode）。
        neighbor_radius  : 搜索候选对手车的空间范围（米）。
    """

    ttc_threshold: float = 5.0
    window_seconds: float = 3.0
    min_episode_frames: int = 10
    neighbor_radius: float = 50.0


# -------------------------- 几何与安全指标计算 -------------------------- #


def compute_ttc_between(
    ego_pos: np.ndarray,
    ego_vel: np.ndarray,
    other_pos: np.ndarray,
    other_vel: np.ndarray,
) -> float:
    """
    计算两车之间的 Time-To-Collision (TTC)。

    算法：
        1. 相对位置向量 r = other_pos - ego_pos。
        2. 单位方向 u = r / ||r||。
        3. 相对速度 v_rel = ego_vel - other_vel。
        4. 接近速率 approach_rate = dot(v_rel, u)。
        5. 若 approach_rate < 1e-4，则认为未在接近，返回 DEFAULT_TTC。
        6. 否则 ttc = ||r|| / approach_rate，并裁剪到 [0, MAX_TTC_CLIP]。
    """
    r = np.asarray(other_pos, dtype=np.float64) - np.asarray(ego_pos, dtype=np.float64)
    dist = float(np.linalg.norm(r))
    if dist < 1e-6:
        return 0.0

    u = r / dist
    v_rel = np.asarray(ego_vel, dtype=np.float64) - np.asarray(other_vel, dtype=np.float64)
    approach_rate = float(np.dot(v_rel, u))
    if approach_rate < 1e-4:
        return DEFAULT_TTC

    ttc = dist / approach_rate
    if ttc < 0.0:
        ttc = 0.0
    ttc = float(np.clip(ttc, 0.0, MAX_TTC_CLIP))
    return ttc


def compute_pet_between(
    ego_pos_seq: np.ndarray,
    other_pos_seq: np.ndarray,
    dt: float,
    conflict_radius: float = 2.0,
) -> float:
    """
    计算两车之间的 Post-Encroachment Time (PET)。

    近似实现：
        1. 以最小车间距时刻 t0 作为“冲突点”参考，设冲突点为该时刻两车中点 c。
        2. 使用 d_ego(t) = ||ego_pos(t) - c||, d_other(t) = ||other_pos(t) - c||。
        3. ego 在冲突区域内：d_ego(t) < conflict_radius；partner 同理。
        4. 若 ego / partner 其中一方从未进入冲突区域，则返回 DEFAULT_PET。
        5. 令 t_ego_exit 为 ego 最后一次在冲突区域内的帧索引；
           在 t > t_ego_exit 的帧中，找到 partner 第一次进入冲突区域的帧 t_partner_enter。
        6. 若存在这样的 t_partner_enter，则 PET = (t_partner_enter - t_ego_exit) * dt；
           否则返回 DEFAULT_PET。
    """
    ego_pos_seq = np.asarray(ego_pos_seq, dtype=np.float64)
    other_pos_seq = np.asarray(other_pos_seq, dtype=np.float64)

    if ego_pos_seq.shape != other_pos_seq.shape or ego_pos_seq.ndim != 2:
        return DEFAULT_PET
    T = ego_pos_seq.shape[0]
    if T < 2:
        return DEFAULT_PET

    # 以最小车间距时刻定义冲突点
    d_pair = np.linalg.norm(ego_pos_seq - other_pos_seq, axis=1)
    # 使用最小车间距自适应调整冲突半径，避免正常车距下永远得不到 PET
    min_dist = float(np.min(d_pair))
    effective_radius = max(min_dist * 1.5, conflict_radius, 3.0)
    t0 = int(np.argmin(d_pair))
    c = 0.5 * (ego_pos_seq[t0] + other_pos_seq[t0])

    d_ego = np.linalg.norm(ego_pos_seq - c, axis=1)
    d_other = np.linalg.norm(other_pos_seq - c, axis=1)

    ego_in = np.where(d_ego < effective_radius)[0]
    other_in = np.where(d_other < effective_radius)[0]
    if ego_in.size == 0 or other_in.size == 0:
        return DEFAULT_PET

    t_ego_exit = int(ego_in.max())
    cand = other_in[other_in > t_ego_exit]
    if cand.size == 0:
        return DEFAULT_PET
    t_partner_enter = int(cand.min())

    pet = (t_partner_enter - t_ego_exit) * float(dt)
    if pet < 0.0:
        return DEFAULT_PET
    pet = float(np.clip(pet, 0.0, MAX_PET_CLIP))
    return pet


# -------------------------- Episode 特征提取 -------------------------- #


def _extract_speed_acc_jerk(
    speeds: np.ndarray,
    valid_mask: np.ndarray,
    dt: float,
) -> Tuple[
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
]:
    """
    在给定速度序列上计算速度 / 加速度 / jerk 相关统计量。

    返回：
        mean_speed, std_speed, max_speed, mean_acc, min_acc, jerk_peak
        若数据不足以计算某些量，则对应条目为 None。
    """
    speeds = np.asarray(speeds, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if speeds.ndim != 1 or valid_mask.ndim != 1 or speeds.shape[0] != valid_mask.shape[0]:
        return None, None, None, None, None, None

    idx = np.where(valid_mask)[0]
    if idx.size < 2:
        # 不足以计算加速度 / jerk
        if idx.size == 0:
            return None, None, None, None, None, None
        seg = speeds[idx]
        mean_speed = float(np.mean(seg))
        std_speed = float(np.std(seg))
        max_speed = float(np.max(seg))
        return mean_speed, std_speed, max_speed, None, None, None

    seg = speeds[idx]
    mean_speed = float(np.mean(seg))
    std_speed = float(np.std(seg))
    max_speed = float(np.max(seg))
    # 仅对时间上连续的 valid 帧计算加速度，避免跨大间隔导致数值失真
    accs_list: List[float] = []
    for i in range(len(idx) - 1):
        if idx[i + 1] != idx[i] + 1:
            continue
        dv = speeds[idx[i + 1]] - speeds[idx[i]]
        accs_list.append(float(dv / float(dt)))

    accs = np.asarray(accs_list, dtype=np.float64)
    if accs.size == 0:
        return mean_speed, std_speed, max_speed, None, None, None
    mean_acc = float(np.mean(accs))
    min_acc = float(np.min(accs))

    # jerk: 相邻加速度之差 / dt
    if accs.size < 2:
        jerk_peak = None
    else:
        jerks = np.diff(accs) / float(dt)
        jerk_peak = float(np.max(np.abs(jerks)))

    return mean_speed, std_speed, max_speed, mean_acc, min_acc, jerk_peak


def extract_episode_features(
    ego_meta: _TrackMeta,
    partner_meta: _TrackMeta,
    t_start: int,
    t_end: int,
    dt: float,
) -> Tuple[np.ndarray, float, float]:
    """
    为单个 episode 提取 8 维特征向量，并返回 episode 内的最小 TTC / PET。

    参数：
        ego_meta    : ego 轨迹的元信息。
        partner_meta: 对手车轨迹元信息。
        t_start     : episode 起始帧索引（包含）。
        t_end       : episode 结束帧索引（包含）。
        dt          : 时间步长（秒）。

    返回：
        features: shape (8,) 的特征向量。
        min_ttc : episode 内最小 TTC（秒）。
        min_pet : episode 内最小 PET（秒；若无法计算则为 DEFAULT_PET）。

    若数据不足（有效帧 < 3），
        - 速度 / 加速度 / jerk 特征置 0；
        - min_ttc 置 DEFAULT_TTC；
        - min_pet 置 DEFAULT_PET。
    """
    state_ego = ego_meta.track["state"]
    state_partner = partner_meta.track["state"]

    T_global = len(ego_meta.valid_mask)
    t_start = max(0, min(t_start, T_global - 1))
    t_end = max(t_start, min(t_end, T_global - 1))

    sl = slice(t_start, t_end + 1)

    pos_ego = np.asarray(state_ego["position"][sl, :2], dtype=np.float64)
    vel_ego = np.asarray(state_ego["velocity"][sl, :2], dtype=np.float64)
    valid_ego = np.asarray(ego_meta.valid_mask[sl], dtype=bool)

    pos_partner = np.asarray(state_partner["position"][sl, :2], dtype=np.float64)
    vel_partner = np.asarray(state_partner["velocity"][sl, :2], dtype=np.float64)
    valid_partner = np.asarray(partner_meta.valid_mask[sl], dtype=bool)

    valid_both = valid_ego & valid_partner
    num_valid = int(valid_both.sum())

    if num_valid < 3:
        features = np.zeros(FEATURE_DIM, dtype=np.float64)
        features[6] = DEFAULT_TTC
        features[7] = DEFAULT_PET
        return features, DEFAULT_TTC, DEFAULT_PET

    speeds = np.linalg.norm(vel_ego, axis=1)

    mean_speed, std_speed, max_speed, mean_acc, min_acc, jerk_peak = _extract_speed_acc_jerk(
        speeds, valid_ego, dt
    )

    if mean_speed is None:
        mean_speed = 0.0
    if std_speed is None:
        std_speed = 0.0
    if max_speed is None:
        max_speed = 0.0
    if mean_acc is None:
        mean_acc = 0.0
    if min_acc is None:
        min_acc = 0.0
    if jerk_peak is None:
        jerk_peak = 0.0

    # 逐帧计算 TTC，取最小值
    min_ttc = DEFAULT_TTC
    for k in range(pos_ego.shape[0]):
        if not (valid_ego[k] or not valid_partner[k]):
            continue
        ttc = compute_ttc_between(
            ego_pos=pos_ego[k],
            ego_vel=vel_ego[k],
            other_pos=pos_partner[k],
            other_vel=vel_partner[k],
        )
        if ttc < min_ttc:
            min_ttc = ttc
    min_ttc = float(np.clip(min_ttc, 0.0, MAX_TTC_CLIP))

    # PET：只在双方同时 valid 的帧上计算
    idx = np.where(valid_both)[0]
    if idx.size >= 2:
        ego_seq = pos_ego[idx]
        partner_seq = pos_partner[idx]
        min_pet = compute_pet_between(ego_seq, partner_seq, dt=dt)
    else:
        min_pet = DEFAULT_PET
    min_pet = float(np.clip(min_pet, 0.0, MAX_PET_CLIP))

    features = np.array(
        [
            mean_speed,
            std_speed,
            max_speed,
            mean_acc,
            min_acc,
            jerk_peak,
            min_ttc,
            min_pet,
        ],
        dtype=np.float64,
    )
    return features, min_ttc, min_pet


# -------------------------- 单场景 episode 提取 -------------------------- #


def _find_best_neighbor_at_time(
    ego_meta: _TrackMeta,
    all_tracks: Dict[Any, _TrackMeta],
    t: int,
    neighbor_radius: float,
) -> Tuple[Optional[Any], float]:
    """
    在给定时间步 t，为 ego 找到空间范围内 TTC 最小的对手车。

    返回：
        partner_id: 若找不到合适对手车则为 None。
        best_ttc  : 若 partner_id 为 None，则为 DEFAULT_TTC。
    """
    state_ego = ego_meta.track["state"]
    if t < 0 or t >= len(ego_meta.valid_mask) or not ego_meta.valid_mask[t]:
        return None, DEFAULT_TTC

    ego_pos = np.asarray(state_ego["position"][t, :2], dtype=np.float64)
    ego_vel = np.asarray(state_ego["velocity"][t, :2], dtype=np.float64)

    best_ttc = DEFAULT_TTC
    best_partner: Optional[Any] = None

    for other_id, other_meta in all_tracks.items():
        # 注意：_TrackMeta.scenario_id 在此处实际存储的是该轨迹在场景中的 track_id
        #（即 traffic_data 的 key），不是“场景索引”，因此这里用来排除自车本身。
        if other_id == ego_meta.scenario_id:
            continue
        if t < 0 or t >= len(other_meta.valid_mask) or not other_meta.valid_mask[t]:
            continue

        state_other = other_meta.track["state"]
        other_pos = np.asarray(state_other["position"][t, :2], dtype=np.float64)
        other_vel = np.asarray(state_other["velocity"][t, :2], dtype=np.float64)

        # 过滤近乎静止的对手车，避免与停放车辆的擦肩而过被误标为强交互
        other_speed = float(np.linalg.norm(other_vel))
        if other_speed < 1.0:
            continue

        vec = other_pos - ego_pos
        dist = float(np.linalg.norm(vec))
        if dist > neighbor_radius:
            continue

        # 排除对向行驶车辆（heading_diff > 150°）以及非前方大角度交叉车辆
        ego_heading = float(state_ego["heading"][t])
        other_heading = float(state_other["heading"][t])
        heading_diff = abs(ego_heading - other_heading)
        if heading_diff > np.pi:
            heading_diff = 2.0 * np.pi - heading_diff
        # 条件A：同向/小角度（< 90°，跟驰/并线）
        cond_a = heading_diff < (np.pi / 2)
        # 条件B：前方大角度交叉（< 150°，路口汇入）
        ego_dir = np.array([np.cos(ego_heading), np.sin(ego_heading)])
        unit_vec = vec / dist
        in_front = np.dot(unit_vec, ego_dir) > 0.0
        cond_b = in_front and (heading_diff < (5.0 * np.pi / 6))
        if not (cond_a or cond_b):
            continue

        ttc = compute_ttc_between(
            ego_pos=ego_pos,
            ego_vel=ego_vel,
            other_pos=other_pos,
            other_vel=other_vel,
        )
        if ttc < best_ttc:
            best_ttc = ttc
            best_partner = other_id

    return best_partner, best_ttc


def _find_ttc_segments(
    ttc_array: np.ndarray,
    partner_ids: Sequence[Optional[Any]],
    threshold: float,
) -> List[Tuple[int, int, int]]:
    """
    在时间轴上寻找 TTC < threshold 的连续片段。

    参数：
        ttc_array  : shape (T,) 的 TTC 序列。
        partner_ids: 长度为 T 的列表，对应每帧的“最危险对手车 ID”（可能为 None）。
        threshold  : TTC 阈值（秒）。

    返回：
        segments: List[(t_start, t_end, t_peak)]，其中
            - [t_start, t_end] 为连续 TTC 低于阈值的片段；
            - t_peak 为该片段内 TTC 最小的时间索引。
    """
    T = len(ttc_array)
    segments: List[Tuple[int, int, int]] = []

    in_seg = False
    seg_start = 0

    for t in range(T):
        cond = (ttc_array[t] < threshold) and (partner_ids[t] is not None)
        if cond and not in_seg:
            in_seg = True
            seg_start = t
        elif not cond and in_seg:
            in_seg = False
            seg_end = t - 1
            if seg_end >= seg_start:
                # 在 [seg_start, seg_end] 内找到 TTC 最小的 t_peak
                seg_slice = ttc_array[seg_start : seg_end + 1]
                peak_rel = int(np.argmin(seg_slice))
                t_peak = seg_start + peak_rel
                segments.append((seg_start, seg_end, t_peak))
    if in_seg:
        seg_end = T - 1
        if seg_end >= seg_start:
            seg_slice = ttc_array[seg_start : seg_end + 1]
            peak_rel = int(np.argmin(seg_slice))
            t_peak = seg_start + peak_rel
            segments.append((seg_start, seg_end, t_peak))

    return segments


def find_interaction_episodes_in_scene(
    env: BCExpertEnv,
    scenario_index: int,
    config: EpisodeConfig,
) -> List[InteractionEpisode]:
    """
    在单个场景中，为每辆动态自车提取所有交互 episodes。

    步骤：
        1. 对每条动态轨迹（ego）：
            a) 在时间轴上，针对每一帧 t，遍历 _all_tracks，
               找到 50m 范围内 TTC 最小的对手车及 TTC。
            b) 得到 per-frame 的 best_ttc[t] 与 best_partner_id[t]。
            c) 在 best_ttc 上寻找 TTC < threshold 的连续片段，
               对每个片段取 TTC 最小时刻作为 t_peak。
            d) 以 t_peak 为中心、window_seconds 为半窗长度构造 [t_start, t_end]。
            e) 计算该窗口的特征向量与 min_ttc / min_pet。
            f) 对同一 (ego, partner) 若存在时间重叠的 episodes，仅保留 min_ttc 更小者。

    返回：
        episodes: 当前场景内所有互不冲突的 InteractionEpisode 列表。
    """
    episodes: List[InteractionEpisode] = []
    dt = float(env.dt)
    window_frames = int(round(config.window_seconds / dt))
    if window_frames < 1:
        window_frames = 1

    for ego_id, ego_meta in env._dynamic_tracks.items():
        T = len(ego_meta.valid_mask)
        if T == 0:
            continue

        best_ttc = np.full(T, DEFAULT_TTC, dtype=np.float64)
        best_partner_ids: List[Optional[Any]] = [None] * T

        # 逐帧寻找最危险对手车
        for t in range(ego_meta.first_valid, ego_meta.last_valid + 1):
            partner_id, ttc = _find_best_neighbor_at_time(
                ego_meta=ego_meta,
                all_tracks=env._all_tracks,
                t=t,
                neighbor_radius=config.neighbor_radius,
            )
            best_ttc[t] = ttc
            best_partner_ids[t] = partner_id

        # 寻找 TTC 低于阈值的连续片段
        segments = _find_ttc_segments(best_ttc, best_partner_ids, config.ttc_threshold)

        # 为每个片段构造 episode，并做冲突消解
        # key: partner_id -> 已选择的 episodes（避免时间重叠）
        chosen_by_partner: Dict[Any, List[InteractionEpisode]] = {}

        for seg_start, seg_end, t_peak in segments:
            partner_id = best_partner_ids[t_peak]
            if partner_id is None:
                continue
            partner_meta = env._all_tracks.get(partner_id, None)
            if partner_meta is None:
                continue

            t_start = max(ego_meta.first_valid, t_peak - window_frames)
            t_end = min(ego_meta.last_valid, t_peak + window_frames)
            if t_end - t_start + 1 < config.min_episode_frames:
                continue

            features, min_ttc, min_pet = extract_episode_features(
                ego_meta=ego_meta,
                partner_meta=partner_meta,
                t_start=t_start,
                t_end=t_end,
                dt=dt,
            )

            episode = InteractionEpisode(
                scenario_index=scenario_index,
                ego_track_id=ego_id,
                partner_track_id=partner_id,
                t_peak=t_peak,
                t_start=t_start,
                t_end=t_end,
                min_ttc=min_ttc,
                min_pet=min_pet,
                features=features,
            )

            lst = chosen_by_partner.setdefault(partner_id, [])
            # 检查与已选 episodes 是否时间重叠
            keep_new = True
            for i, old_ep in enumerate(lst):
                overlap = not (t_end < old_ep.t_start or t_start > old_ep.t_end)
                if not overlap:
                    continue
                # 存在重叠：保留 min_ttc 更小的那一个
                if episode.min_ttc < old_ep.min_ttc:
                    lst[i] = episode
                keep_new = False
                break
            if keep_new:
                lst.append(episode)

        # 收集该 ego 的所有 episodes
        for eps in chosen_by_partner.values():
            episodes.extend(eps)

    return episodes


# -------------------------- 跨场景收集与保存 -------------------------- #


def collect_all_episodes(
    waymo_dir: str,
    num_scenarios: int,
    start_index: int,
    waymo_dt: float,
    episode_config: EpisodeConfig,
) -> Tuple[np.ndarray, List[InteractionEpisode]]:
    """
    遍历多个场景，提取并汇总所有 interaction episodes。

    参数：
        waymo_dir     : Waymo/ScenarioNet .pkl 数据目录。
        num_scenarios : 需要处理的场景数量。
        start_index   : 起始场景索引（包含）。
        waymo_dt      : Waymo 时间步长（秒），通常为 0.1。
        episode_config: EpisodeConfig 配置对象。

    返回：
        features: np.ndarray, shape (N, 8)，为所有 episodes 的特征矩阵。
        episodes: List[InteractionEpisode]，长度为 N，对应每一行特征。
    """
    waymo_dir = os.path.abspath(waymo_dir)

    all_features: List[np.ndarray] = []
    all_episodes: List[InteractionEpisode] = []

    iterator = range(start_index, start_index + num_scenarios)
    if tqdm is not None:
        iterator = tqdm(iterator, desc="场景进度", unit="场景")

    for idx in iterator:
        config = BCExpertEnv.default_config()
        config.update(
            dict(
                data_directory=waymo_dir,
                start_scenario_index=idx,
                num_scenarios=1,
                waymo_dt=float(waymo_dt),
            )
        )

        try:
            env = BCExpertEnv(config)
            env.reset(seed=idx)
        except AssertionError as e:
            msg = str(e)
            print(
                f"[extract_interaction_episodes] 场景 {idx} 加载失败: "
                f"{msg if msg else 'Insufficient scenarios!'}"
            )
            break

        episodes = find_interaction_episodes_in_scene(env, scenario_index=idx, config=episode_config)
        env.close()

        if not episodes:
            continue

        feats = np.stack([ep.features for ep in episodes], axis=0)
        all_features.append(feats)
        all_episodes.extend(episodes)

    if not all_features:
        raise RuntimeError(
            "未提取到任何 interaction episodes，请检查数据路径、场景数量或 ttc_threshold 配置。"
        )

    features_all = np.concatenate(all_features, axis=0).astype(np.float64)
    return features_all, all_episodes


def save_episodes(
    features: np.ndarray,
    episodes: List[InteractionEpisode],
    output_dir: str,
) -> None:
    """
    将 episode 特征与元信息分别保存到 .npz 与 .json 文件。

    输出：
        - episode_features.npz: 包含键 "features"，shape (N, 8)
        - episode_meta.json   : List[dict]，每个元素包含：
              {
                  "scenario_index": int,
                  "ego_track_id": Any(JSON 序列化后),
                  "partner_track_id": Any(JSON 序列化后),
                  "t_peak": int,
                  "t_start": int,
                  "t_end": int,
                  "min_ttc": float,
                  "min_pet": float,
              }
    """
    os.makedirs(output_dir, exist_ok=True)

    npz_path = os.path.join(output_dir, "episode_features.npz")
    np.savez_compressed(npz_path, features=features.astype(np.float32))

    json_path = os.path.join(output_dir, "episode_meta.json")
    meta_list: List[Dict[str, Any]] = []
    for ep in episodes:
        d = asdict(ep)
        # features 不写入 JSON
        d.pop("features", None)
        # track_id 统一转成 str 以保证可序列化
        d["ego_track_id"] = (
            d["ego_track_id"]
            if isinstance(d["ego_track_id"], (int, float, str))
            else str(d["ego_track_id"])
        )
        d["partner_track_id"] = (
            d["partner_track_id"]
            if isinstance(d["partner_track_id"], (int, float, str))
            else str(d["partner_track_id"])
        )
        meta_list.append(d)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta_list, f, ensure_ascii=False, indent=2)

    print(f"[extract_interaction_episodes] Saved features to: {npz_path}")
    print(f"[extract_interaction_episodes] Saved meta to: {json_path}")
    print(
        "[extract_interaction_episodes] 航向角过滤已启用，阈值：同向<90°或前方交叉<150°"
    )

    # 打印简单的统计摘要，方便人工检查 episode 质量
    if episodes:
        min_ttc_values = np.array([ep.min_ttc for ep in episodes], dtype=np.float64)
        min_pet_values = np.array([ep.min_pet for ep in episodes], dtype=np.float64)
        num_episodes = len(episodes)
        mean_min_ttc = float(np.mean(min_ttc_values))
        mean_min_pet = float(np.mean(min_pet_values))
        valid_pet_mask = min_pet_values < DEFAULT_PET
        valid_pet_ratio = float(np.sum(valid_pet_mask)) / float(num_episodes)
        print(
            "[extract_interaction_episodes] 统计摘要: "
            f"N={num_episodes}, mean_min_ttc={mean_min_ttc:.3f}, "
            f"mean_min_pet={mean_min_pet:.3f}, "
            f"pet_valid_ratio( < {DEFAULT_PET})={valid_pet_ratio:.3f}"
        )


# -------------------------- 命令行接口 -------------------------- #


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="从 Waymo/ScenarioNet 轨迹中提取强交互 episodes 及其特征。"
    )
    parser.add_argument(
        "--waymo_dir",
        type=str,
        required=True,
        help="包含 Waymo->ScenarioNet 转换后 .pkl 文件的目录。",
    )
    parser.add_argument(
        "--num_scenarios",
        type=int,
        required=True,
        help="要处理的场景数量。",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="起始场景索引（默认 0）。",
    )
    parser.add_argument(
        "--waymo_dt",
        type=float,
        default=0.1,
        help="Waymo 轨迹时间步长（秒），默认 0.1s。",
    )
    parser.add_argument(
        "--ttc_threshold",
        type=float,
        default=5.0,
        help="TTC 阈值（秒），低于该值视为强交互（默认 5.0）。",
    )
    parser.add_argument(
        "--window_seconds",
        type=float,
        default=3.0,
        help="以危险峰值为中心的时间窗口半长（秒，默认 3.0）。",
    )
    parser.add_argument(
        "--min_episode_frames",
        type=int,
        default=10,
        help="episode 至少包含的帧数（默认 10）。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录（默认 bc_baseline/outputs）。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        # 默认输出到 bc_baseline/outputs/interaction_episodes
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(project_root, "outputs", "interaction_episodes")
    output_dir = os.path.abspath(output_dir)

    episode_config = EpisodeConfig(
        ttc_threshold=float(args.ttc_threshold),
        window_seconds=float(args.window_seconds),
        min_episode_frames=int(args.min_episode_frames),
    )

    print("[extract_interaction_episodes] 开始提取 interaction episodes...")
    features, episodes = collect_all_episodes(
        waymo_dir=args.waymo_dir,
        num_scenarios=args.num_scenarios,
        start_index=args.start_index,
        waymo_dt=args.waymo_dt,
        episode_config=episode_config,
    )
    print(
        f"[extract_interaction_episodes] 共提取 {features.shape[0]} 个 episodes，"
        f"特征维度={features.shape[1]}"
    )

    save_episodes(features, episodes, output_dir=output_dir)


if __name__ == "__main__":
    main()

