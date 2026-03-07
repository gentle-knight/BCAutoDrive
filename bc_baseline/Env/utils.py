from __future__ import annotations

import numpy as np
from typing import Dict, Iterable, Tuple, Any

"""
本文件实现 Behavior Cloning 基线中**统一的自车中心（Ego-centric）观测提取逻辑**。

核心目标：
1. 所有特征均在自车坐标系下表示，**绝不向网络暴露全局坐标 (position_x, position_y)**。
2. 避免通过位置差分计算角速度，**yaw_rate 严格硬编码为 0.0**，从根源上消除微分噪声及分布偏移。
3. 对所有模块（数据生成、训练、评估）提供一个唯一的入口函数：
      extract_ego_observation(vehicle, map_manager, active_agents)

观测空间设计（总维度 = 45）：
    - Ego 特征（5 维）：
        1) lateral_offset      : 自车相对于当前车道中心线的横向偏移（米），左正右负（取决于地图实现）
        2) heading_error       : 自车航向与车道切向的偏差角（弧度），取值范围约 [-pi, pi]
        3) vel_longitudinal    : 自车在 **自车坐标系 X 轴方向** 的速度（纵向速度，前正后负）
        4) vel_lateral         : 自车在 **自车坐标系 Y 轴方向** 的速度（横向速度，左正右负）
        5) yaw_rate            : 自车角速度（rad/s），**本基线中强制设置为 0.0**

    - Neighbor 特征（40 维 = 10 × 4）：
        - 在自车 30 米半径范围内，选择距离最近的最多 10 辆其他车。
        - 对每辆邻居车，在自车坐标系下提取 4 维：
            [rel_x, rel_y, vel_x, vel_y]
            其中：
                rel_x, rel_y : 邻居车辆位置在自车坐标系（以自车为原点，X 轴指向自车朝向）下的坐标
                vel_x, vel_y : 邻居车辆速度在自车坐标系下的分解
        - 若少于 10 辆邻居车，则按“整车”为单位补零，始终保证 10 × 4 = 40 维。

注意：
    - 本文件中会访问 MetaDrive 的 map / lane / vehicle 的内部属性，
      仅用于**中间计算**（例如从车道坐标系求 lateral_offset），这些全局信息不会直接出现在观测向量中。
"""

# -------------------------- 观测空间维度常量 -------------------------- #

# Ego 部分维度：5
OBS_EGO_DIM: int = 5

# 邻居槽位数量：最多考虑 10 辆邻居车
OBS_NEIGHBOR_SLOTS: int = 10

# 每辆邻居车的特征维度：4 = (rel_x, rel_y, vel_x, vel_y)
OBS_NEIGHBOR_FEAT_DIM: int = 4

# 总观测维度：5 + 10 * 4 = 45
OBS_DIM: int = OBS_EGO_DIM + OBS_NEIGHBOR_SLOTS * OBS_NEIGHBOR_FEAT_DIM


# -------------------------- 基础数学工具函数 -------------------------- #

def _wrap_angle_rad(x: float) -> float:
    """
    将角度规范化到 [-pi, pi] 区间。

    说明：
        使用 atan2(sin, cos) 的方式，在数值上比直接使用取模更稳定，
        可避免在接近 ±pi 处出现数值不连续的问题。
    """
    return float(np.arctan2(np.sin(x), np.cos(x)))


def to_ego_frame_2d(
    ego_position: np.ndarray,
    ego_heading_rad: float,
    point_global: np.ndarray,
    velocity_global: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray | None]:
    """
    将**全局坐标系**下的点 / 速度投影到**自车坐标系**下。

    自车坐标系定义：
        - 原点：自车质心位置
        - X 轴：沿自车朝向方向指向前方
        - Y 轴：在平面内，左手为正（由坐标变换决定，这里采用逆时针为正的惯例）

    数学形式：
        设自车朝向为 θ（弧度，世界坐标系中相对于 X 轴逆时针为正），则：

            R_world_to_ego = R(-θ)
                = [[ cos(-θ), -sin(-θ)],
                   [ sin(-θ),  cos(-θ)]]

        对任意全局点 p_global，其相对位移为：
            delta = p_global - ego_position

        在自车系中的坐标为：
            p_ego = R_world_to_ego @ delta

        若给定全局速度向量 v_global，则：
            v_ego = R_world_to_ego @ v_global

    参数：
        ego_position    : 自车在全局坐标系下的位置 (x, y)
        ego_heading_rad : 自车在全局坐标系下的朝向角（弧度）
        point_global    : 目标点在全局坐标系下的位置 (x, y)
        velocity_global : （可选）目标点在全局坐标系下的速度向量 (vx, vy)

    返回：
        - 若 velocity_global 为 None:
              (pos_ego, None)
        - 否则:
              (pos_ego, vel_ego)
    """
    ego_position = np.asarray(ego_position[:2], dtype=np.float64)
    point_global = np.asarray(point_global[:2], dtype=np.float64)

    # 旋转角为 -heading，即从“世界系”旋转到“自车系”
    c = np.cos(-ego_heading_rad)
    s = np.sin(-ego_heading_rad)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)  # 2×2 旋转矩阵

    # 先构造从自车到目标点的位移，再乘以旋转矩阵
    delta = point_global - ego_position
    pos_ego = R @ delta

    if velocity_global is not None:
        velocity_global = np.asarray(velocity_global[:2], dtype=np.float64)
        vel_ego = R @ velocity_global
        return pos_ego, vel_ego

    return pos_ego, None


# -------------------------- 车道相关特征提取 -------------------------- #

def get_ego_lane_features(vehicle: Any, map_manager: Any) -> Tuple[float, float]:
    """
    计算 Ego 的车道相关特征：
        1) lateral_offset : 自车相对于最近车道中心线的横向偏移（米）
        2) heading_error  : 自车朝向与车道切线方向的偏差角（弧度，ego_heading - lane_heading）

    设计要点：
        - 优先使用 MetaDrive 中 vehicle.navigation.current_ref_lanes 提供的当前参考车道；
        - 若导航信息缺失，则回退为在 road_network 中搜索“最近车道”；
        - 若 map 未加载或查询失败，则返回 (0.0, 0.0)，保持观测维度稳定。

    参数：
        vehicle     : MetaDrive 自车对象，需要至少提供：
                        - position (x, y)
                        - heading_theta 或 heading
        map_manager : MetaDrive 的 map 管理器，一般为 env.engine.map_manager

    返回：
        (lateral_offset, heading_error)
    """
    lateral_offset: float = 0.0
    heading_error: float = 0.0

    if map_manager is None or getattr(map_manager, "current_map", None) is None:
        # 若地图信息缺失，则无法计算车道几何，直接返回 0
        return lateral_offset, heading_error

    try:
        # 1. 获取自车全局位置
        ego_pos = np.asarray(getattr(vehicle, "position")[:2], dtype=np.float64)

        # 2. 优先从导航模块中获取当前参考车道
        lane = None
        navigation = getattr(vehicle, "navigation", None)
        if navigation is not None and getattr(navigation, "current_ref_lanes", None):
            ref_lanes = navigation.current_ref_lanes
            if ref_lanes:
                lane = ref_lanes[0]

        # 3. 若导航未提供车道，则使用 road_network 最近车道
        if lane is None:
            road_network = map_manager.current_map.road_network
            # MetaDrive 通常提供 get_closest_lane_index(pos, return_lane=True)
            lane, _ = road_network.get_closest_lane_index(ego_pos, return_lane=True)

        if lane is None:
            # 未找到有效车道，保持 0.0
            return lateral_offset, heading_error

        # 4. 在车道局部坐标中获取 (longitudinal, lateral)，其中 lateral 即为相对中心线偏移
        longitudinal, lateral = lane.local_coordinates(ego_pos)
        lateral_offset = float(lateral)

        # 5. 自车朝向
        ego_heading = float(
            getattr(vehicle, "heading_theta", getattr(vehicle, "heading", 0.0))
        )

        # 6. 车道在该纵向位置的切线朝向
        if hasattr(lane, "heading_theta_at"):
            lane_heading = float(lane.heading_theta_at(longitudinal))
        elif hasattr(lane, "heading_at"):
            lane_heading = float(lane.heading_at(longitudinal))
        else:
            # 若车道未提供 heading 查询接口，则保守起见视为与自车同向
            lane_heading = ego_heading

        # 7. 航向误差：ego_heading - lane_heading，经过 wrap 到 [-pi, pi]
        heading_error = _wrap_angle_rad(ego_heading - lane_heading)

        return lateral_offset, heading_error
    except Exception:
        # 为了鲁棒性，任何异常都不影响整体 pipeline，直接回退到 0.0
        return 0.0, 0.0


# -------------------------- 统一 Ego 观测提取函数 -------------------------- #

def extract_ego_observation(
    vehicle: Any,
    map_manager: Any,
    active_agents: Dict[Any, Any] | Iterable[Any],
    *,
    neighbor_radius: float = 30.0,
    max_neighbors: int = OBS_NEIGHBOR_SLOTS,
) -> np.ndarray:
    """
    提取单辆自车的 **45 维 Ego-centric 观测向量**。

    观测结构：
        - 前 5 维：Ego 特征
              [lateral_offset, heading_error,
               vel_longitudinal, vel_lateral,
               yaw_rate(=0.0)]
        - 后 40 维：邻居车辆特征（最多 10 辆，每辆 4 维）
              [rel_x_1, rel_y_1, vel_x_1, vel_y_1,
               ...
               rel_x_10, rel_y_10, vel_x_10, vel_y_10]

    绝对红线：
        - yaw_rate 必须在此函数内部**硬编码为常数 0.0**，不得从位置 / 航向差分得到。
        - 所有 rel_x / rel_y / vel_x / vel_y 必须通过**显式的二维旋转矩阵**
          从全局坐标系投影到自车坐标系。

    参数：
        vehicle        : 当前 Ego 车辆（MetaDrive Vehicle 实例）
        map_manager    : 地图管理器（env.engine.map_manager），用于计算车道相关特征
        active_agents  : 当前场景中所有“动态车辆”的集合：
                            - 可以是 dict[agent_id, vehicle]
                            - 也可以是任意可迭代的 vehicle 列表
                          Ego 车辆会在函数内部被自动排除，不计入邻居。
        neighbor_radius: 邻居搜索半径（米），默认 30m
        max_neighbors  : 邻居数量上限（默认 10），即邻居槽位数

    返回：
        obs : np.ndarray, shape = (OBS_DIM,), dtype = float32
    """
    # -------------------- 1. 提取自车基础状态 -------------------- #
    ego_pos = np.asarray(getattr(vehicle, "position")[:2], dtype=np.float64)
    ego_heading = float(
        getattr(vehicle, "heading_theta", getattr(vehicle, "heading", 0.0))
    )

    # 速度（全局系）：优先使用 vehicle.velocity；若不存在则由 speed + heading 近似
    if hasattr(vehicle, "velocity"):
        ego_vel_world = np.asarray(getattr(vehicle, "velocity")[:2], dtype=np.float64)
    else:
        speed = float(getattr(vehicle, "speed", 0.0))
        ego_vel_world = np.array(
            [speed * np.cos(ego_heading), speed * np.sin(ego_heading)],
            dtype=np.float64,
        )

    # 使用自车自身的位置 + 速度，结合 to_ego_frame_2d，得到自车在自身坐标系下的速度分解
    _, ego_vel_ego = to_ego_frame_2d(
        ego_position=ego_pos,
        ego_heading_rad=ego_heading,
        point_global=ego_pos,           # 位置与自车相同，只为复用接口
        velocity_global=ego_vel_world,  # 重点是将速度向量投影到自车系
    )
    vel_longitudinal = float(ego_vel_ego[0])
    vel_lateral = float(ego_vel_ego[1])

    # 车道相关特征：lateral_offset / heading_error
    lateral_offset, heading_error = get_ego_lane_features(vehicle, map_manager)

    # 绝对红线：yaw_rate 在此处直接硬编码为 0.0，不做任何微分运算
    yaw_rate = 0.0

    ego_state = np.array(
        [lateral_offset, heading_error, vel_longitudinal, vel_lateral, yaw_rate],
        dtype=np.float32,
    )

    # -------------------- 2. 搜索邻居车辆（最近的最多 10 辆） -------------------- #
    candidates: list[Tuple[float, Any]] = []

    # 兼容两种 active_agents 形式：
    #   - dict[agent_id, vehicle]
    #   - 仅包含 vehicle 的可迭代对象（list / tuple / set 等）
    if isinstance(active_agents, dict):
        iterable = active_agents.values()
    else:
        iterable = active_agents

    for other in iterable:
        # 跳过 Ego 自身（使用对象 identity 判断）
        if other is vehicle:
            continue

        try:
            other_pos = np.asarray(getattr(other, "position")[:2], dtype=np.float64)
        except Exception:
            # 若该对象不具备 position 属性，则跳过
            continue

        dist = float(np.linalg.norm(ego_pos - other_pos))
        if dist < neighbor_radius:
            candidates.append((dist, other))

    # 按距离从近到远排序，并截取前 max_neighbors 个
    candidates.sort(key=lambda x: x[0])
    selected = candidates[: max_neighbors]

    # -------------------- 3. 构造邻居特征（投影到自车坐标系） -------------------- #
    neighbor_feats: list[float] = []

    for _, neighbor in selected:
        # 邻居位置（全局系）
        n_pos_world = np.asarray(
            getattr(neighbor, "position")[:2],
            dtype=np.float64,
        )

        # 邻居速度（全局系）：同样优先用 velocity，否则由 speed + heading 近似
        if hasattr(neighbor, "velocity"):
            n_vel_world = np.asarray(
                getattr(neighbor, "velocity")[:2],
                dtype=np.float64,
            )
        else:
            n_heading = float(
                getattr(neighbor, "heading_theta", getattr(neighbor, "heading", 0.0))
            )
            n_speed = float(getattr(neighbor, "speed", 0.0))
            n_vel_world = np.array(
                [n_speed * np.cos(n_heading), n_speed * np.sin(n_heading)],
                dtype=np.float64,
            )

        # 核心：使用显式 2D 旋转矩阵，将邻居的位置 / 速度从全局坐标系投影到自车坐标系
        pos_ego, vel_ego = to_ego_frame_2d(
            ego_position=ego_pos,
            ego_heading_rad=ego_heading,
            point_global=n_pos_world,
            velocity_global=n_vel_world,
        )

        rel_x = float(pos_ego[0])
        rel_y = float(pos_ego[1])
        vel_x = float(vel_ego[0]) if vel_ego is not None else 0.0
        vel_y = float(vel_ego[1]) if vel_ego is not None else 0.0

        neighbor_feats.extend([rel_x, rel_y, vel_x, vel_y])

    # 若邻居数量不足 max_neighbors，则按“整车”为单位补零，维度对齐到 40 维
    missing = max_neighbors - len(selected)
    if missing > 0:
        neighbor_feats.extend([0.0] * (OBS_NEIGHBOR_FEAT_DIM * missing))

    # 安全起见，强制长度裁剪 / 填充到精确的 40 维
    if len(neighbor_feats) > OBS_NEIGHBOR_SLOTS * OBS_NEIGHBOR_FEAT_DIM:
        neighbor_feats = neighbor_feats[: OBS_NEIGHBOR_SLOTS * OBS_NEIGHBOR_FEAT_DIM]
    elif len(neighbor_feats) < OBS_NEIGHBOR_SLOTS * OBS_NEIGHBOR_FEAT_DIM:
        neighbor_feats.extend(
            [0.0] * (OBS_NEIGHBOR_SLOTS * OBS_NEIGHBOR_FEAT_DIM - len(neighbor_feats))
        )

    # -------------------- 4. 拼接 Ego 与 Neighbor 特征，返回 45 维向量 -------------------- #
    obs = np.concatenate([ego_state, np.asarray(neighbor_feats, dtype=np.float32)], axis=0)

    # 最终保证输出维度为 (45,) 且 dtype 为 float32
    assert obs.shape == (OBS_DIM,), f"观测维度不匹配，期望 {OBS_DIM}，实际 {obs.shape[0]}"
    return obs.astype(np.float32)


__all__ = [
    "OBS_EGO_DIM",
    "OBS_NEIGHBOR_SLOTS",
    "OBS_NEIGHBOR_FEAT_DIM",
    "OBS_DIM",
    "extract_ego_observation",
    "get_ego_lane_features",
    "to_ego_frame_2d",
]

