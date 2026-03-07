from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.type import MetaDriveType

from bc_baseline.Env.utils import extract_ego_observation, OBS_DIM


"""
Expert 环境（BCExpertEnv）
-------------------------

职责：
    - 这是一个**离线专家数据提取环境**，用于从 MetaDrive / Waymo 轨迹中生成 Behavior Cloning 的监督数据。
    - 通过 MetaDrive 的 ScenarioEnv 加载 Waymo->ScenarioNet 的场景（.pkl），
      然后直接在轨迹层面遍历所有“动态车辆”的真实轨迹。
    - 对每一帧 t：
        1) 基于轨迹构造“虚拟车辆对象”，调用 extract_ego_observation() 提取 51 维 Ego-centric 观测 obs_t；
        2) 使用简单的**逆动力学（Inverse Dynamics）自行车模型**，从 (state_t, state_{t+1}) 反推出
           连续动作 a_t = [steering, acceleration]。

关键设计点：
    - **不运行物理仿真**：我们直接使用 Waymo/ScenarioNet 的 ground-truth 轨迹，
      不通过 env.step() 让物理引擎演化，所以不会引入额外数值误差。
    - **完全 ego-centric 观测**：观测由 bc_baseline.Env.utils.extract_ego_observation 提供，
      内部已严格遵守“禁止全局坐标、yaw_rate=0.0”等硬性约束。
    - **动态车辆过滤**：仅保留在一段时间内真正运动过的车辆（排除完全静止的背景车辆），
      以避免训练集中充斥静止样本。
"""


@dataclass
class _TrackMeta:
    """缓存每条轨迹的基本信息，便于快速遍历。"""

    scenario_id: Any
    track: Dict[str, Any]
    valid_mask: np.ndarray  # bool, shape (T,)
    first_valid: int
    last_valid: int


class InverseDynamics:
    """
    逆动力学模型：从 (state_t, state_{t+1}) 反推出连续动作 [steering, acceleration]。

    假设车辆满足简化的**自行车模型（Kinematic Bicycle Model）**：
        - 车辆质心速度 v 沿车身纵向：
              v = ||velocity|| = sqrt(v_x^2 + v_y^2)
        - 航向角为 θ（heading），前轮转角为 δ（steering）。
        - 自行车模型中的航向角变化率：
              θ_dot = v / L * tan(δ)
          其中 L 为车辆轴距（wheelbase）。

    给定离散时间步长 dt，我们有：
        - θ_dot ≈ (θ_{t+1} - θ_t) / dt
        - v_t = ||velocity_t||
        - v_{t+1} = ||velocity_{t+1}||
        - 纵向加速度：
              a = (v_{t+1} - v_t) / dt

    于是可以反求 steering（δ）：
        - rearrange: θ_dot = v_t / L * tan(δ)
              => tan(δ) = L * θ_dot / v_t
              => δ = arctan( L * θ_dot / v_t )

    为了便于神经网络训练，我们将动作归一化到 [-1, 1]：
        steering_norm   = clip( δ / max_steering, -1, 1 )
        acceleration_norm = clip( a / max_acc,     -1, 1 )
    """

    def __init__(self, max_steering: float = 0.7, max_acc: float = 8.0, length: float = 4.5):
        """
        参数：
            max_steering : 允许的最大方向盘转角（弧度），如 0.7 rad ≈ 40 度
            max_acc      : 允许的最大纵向加速度（m/s^2）
            length       : 车辆长度（m），Waymo 小车约在 4.5m 左右
        """
        self.max_steering = float(max_steering)
        self.max_acc = float(max_acc)
        # 有效轴距 L：这里简单认为是车长的 0.7 倍（经验近似）
        self.wheelbase = 0.7 * float(length)

    @staticmethod
    def _speed(vel: np.ndarray) -> float:
        """计算二维速度向量的模长 v = sqrt(vx^2 + vy^2)。"""
        return float(np.linalg.norm(vel))

    def compute_action(
        self,
        current_state: Dict[str, np.ndarray],
        next_state: Dict[str, np.ndarray],
        dt: float,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        从当前状态 current_state 和下一时刻状态 next_state 反推出动作 [steering, acceleration]。

        状态格式：
            current_state / next_state 为 dict，包含：
                - "position": np.ndarray shape (2,) or (x, y)      （此处仅用作调试，可选）
                - "heading" : float, 航向角（弧度）
                - "velocity": np.ndarray shape (2,) or (vx, vy)

        数学步骤：
            1. 计算当前/下一时刻的标量速度 v_t, v_{t+1}；
            2. 由两帧航向差求出 θ_dot；
            3. 由 v_t 和 θ_dot 反解前轮转角 δ；
            4. 由 v_t, v_{t+1} 和 dt 计算纵向加速度 a；
            5. 将 δ 和 a 归一化到 [-1, 1]。
        """
        v_curr = self._speed(np.asarray(current_state["velocity"], dtype=np.float64))
        v_next = self._speed(np.asarray(next_state["velocity"], dtype=np.float64))

        # 1. 纵向加速度：a = (v_{t+1} - v_t) / dt
        acc = (v_next - v_curr) / float(dt)

        # 2. 航向角变化率 θ_dot
        theta_curr = float(current_state["heading"])
        theta_next = float(next_state["heading"])

        # 角度差需要在 [-pi, pi] 上 wrap 一下，避免跨越 2*pi 带来的跳变
        diff = theta_next - theta_curr
        if diff > np.pi:
            diff -= 2.0 * np.pi
        elif diff < -np.pi:
            diff += 2.0 * np.pi
        theta_dot = diff / float(dt)

        # 3. 反解转角 δ：δ = atan( L * θ_dot / v )
        if v_curr < 0.1:
            # 速度太小时，航向变化难以可靠反推，直接认为转角为 0
            steering = 0.0
        else:
            steering = float(np.arctan(self.wheelbase * theta_dot / max(v_curr, 1e-3)))

        # 4. 归一化到 [-1, 1]
        norm_acc = float(np.clip(acc / self.max_acc, -1.0, 1.0))
        norm_steering = float(np.clip(steering / self.max_steering, -1.0, 1.0))

        action = np.array([norm_steering, norm_acc], dtype=np.float32)
        debug_info = {
            "raw_acc": float(acc),
            "raw_steering": float(steering),
            "v_curr": float(v_curr),
            "v_next": float(v_next),
            "theta_curr": theta_curr,
            "theta_next": theta_next,
            "theta_dot": float(theta_dot),
        }
        return action, debug_info


class _VehicleProxy:
    """
    一个极简的“车辆代理对象”，用于在不创建真实 MetaDrive 车辆的前提下，
    为 extract_ego_observation() 提供统一的访问接口。

    必要属性：
        - position      : np.ndarray shape (2,)，世界坐标系下的 (x, y)
        - velocity      : np.ndarray shape (2,)，世界坐标系下的 (vx, vy)
        - heading_theta : float，自车航向（弧度）

    说明：
        - 我们不依赖 navigation / 物理引擎，只用这些几何量即可完成 Ego-centric 特征提取。
    """

    def __init__(self, position: np.ndarray, velocity: np.ndarray, heading: float):
        self.position = np.asarray(position[:2], dtype=np.float64)
        self.velocity = np.asarray(velocity[:2], dtype=np.float64)
        self.heading_theta = float(heading)
        # 为了兼容某些函数（如 get_ego_lane_features 中的 fallback），额外提供 heading 属性
        self.heading = float(heading)
        # 不使用导航模块，这里直接置为 None
        self.navigation = None


class BCExpertEnv(ScenarioEnv):
    """
    基于 MetaDrive ScenarioEnv 的 Behavior Cloning 专家数据提取环境。

    使用方式（离线数据生成）：
        1. 创建环境：
               config = BCExpertEnv.default_config()
               config["data_directory"] = "/path/to/waymo_pkl"
               config["start_scenario_index"] = i
               config["num_scenarios"] = 1
               env = BCExpertEnv(config)

        2. reset 加载场景：
               env.reset(seed=i)

        3. 提取当前场景的所有 (obs, action)：
               obs, actions = env.collect_expert_data()
               # obs.shape     = (N, 51)
               # actions.shape = (N, 2)

    注意：
        - 本环境不用于在线交互控制，仅用于离线数据抽取；
        - 不调用 env.step() 进行物理仿真，而是直接在轨迹数组上滑动时间轴。
    """

    @classmethod
    def default_config(cls) -> Dict[str, Any]:
        config = super().default_config()
        config.update(
            dict(
                data_directory=None,  # Waymo/ScenarioNet .pkl 目录，必须由外部指定
                waymo_dt=0.1,  # Waymo 采样周期（秒），默认 0.1s，即 10Hz
                # 判断“动态车辆”的简单阈值（与 legacy 逻辑保持大致一致）：
                static_displacement_threshold=5.0,  # 总位移 < 5m 视为静止
                static_speed_threshold=1.0,  # 最大速度 < 1 m/s 视为静止
            )
        )
        return config

    def __init__(self, config: Dict[str, Any] | None = None):
        if config is None:
            config = {}
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO)

        self.dt: float = float(self.config.get("waymo_dt", 0.1))
        self.inverse_dynamics = InverseDynamics()

        # 当前场景的“全量”车辆轨迹缓存（包含静止车）：
        #   - 用于在构造邻居集合时，保证静止车辆不会“隐形”（避免感知盲区）。
        #   - 仅要求轨迹存在至少一帧 valid。
        self._all_tracks: Dict[Any, _TrackMeta] = {}

        # 当前场景的动态车辆轨迹缓存：scenario_id -> _TrackMeta
        self._dynamic_tracks: Dict[Any, _TrackMeta] = {}

    # ------------------------------------------------------------------ #
    #  环境生命周期：reset 仅负责加载场景与构建轨迹索引
    # ------------------------------------------------------------------ #

    def reset(self, seed: int | None = None, options: Dict[str, Any] | None = None):
        """
        重置环境并加载一个新的 Waymo 场景。

        - 调用父类 ScenarioEnv.reset(seed) 以确保 MetaDrive 正常构建地图和 traffic_manager；
        - 然后从 traffic_manager.current_traffic_data 中解析所有车辆轨迹：
            1) 只要该车辆轨迹存在至少一帧 valid，就存入 self._all_tracks（用于“邻居可见”）；
            2) 若该车辆满足动态条件（位移/速度阈值），再额外存入 self._dynamic_tracks（用于“作为 Ego 采样”）。
        """
        obs = super().reset(seed=seed)

        # 从 MetaDrive 中读取当前场景的 traffic 数据
        engine = getattr(self, "engine", None)
        if engine is None or not hasattr(engine, "traffic_manager"):
            raise RuntimeError("BCExpertEnv: engine.traffic_manager 未正确初始化，请检查 MetaDrive 配置。")

        traffic_data: Dict[Any, Dict[str, Any]] = engine.traffic_manager.current_traffic_data
        map_manager = getattr(engine, "map_manager", None)
        if map_manager is None or getattr(map_manager, "current_map", None) is None:
            raise RuntimeError("BCExpertEnv: engine.map_manager.current_map 缺失，无法进行车道相关计算。")

        # 构建轨迹索引：
        #   - self._all_tracks    : 所有 valid 车辆（包含静止车），用于邻居集合
        #   - self._dynamic_tracks: 动态车辆（用于作为 Ego 生成训练样本）
        self._all_tracks.clear()
        self._dynamic_tracks.clear()
        static_disp_th = float(self.config.get("static_displacement_threshold", 5.0))
        static_speed_th = float(self.config.get("static_speed_threshold", 1.0))

        num_total = 0
        num_vehicle = 0
        num_dynamic = 0

        for scenario_id, track in traffic_data.items():
            num_total += 1
            if track.get("type", None) != MetaDriveType.VEHICLE:
                continue
            num_vehicle += 1

            state = track["state"]
            valid = np.asarray(state["valid"], dtype=bool)
            if not valid.any():
                continue

            # 记录 valid 的首尾索引，便于后续遍历时设定时间范围
            first_valid = int(np.argmax(valid))
            last_valid = int(len(valid) - 1 - np.argmax(valid[::-1]))

            meta = _TrackMeta(
                scenario_id=scenario_id,
                track=track,
                valid_mask=valid,
                first_valid=first_valid,
                last_valid=last_valid,
            )

            # 关键修复：全量 valid 车辆都存入 _all_tracks，确保静止车辆作为邻居“可见”
            self._all_tracks[scenario_id] = meta

            positions = np.asarray(state["position"][valid], dtype=np.float64)
            velocities = np.asarray(state["velocity"][valid], dtype=np.float64)

            # 总位移：轨迹起点到终点的欧氏距离
            total_displacement = float(np.linalg.norm(positions[-1] - positions[0])) if len(positions) > 1 else 0.0
            # 最大速度：所有有效帧速度模长的最大值
            speeds = np.linalg.norm(velocities, axis=1) if len(velocities) > 0 else np.zeros(1)
            max_speed = float(np.max(speeds))

            is_static = (total_displacement < static_disp_th) and (max_speed < static_speed_th)
            if is_static:
                continue

            # 满足动态条件的车辆，额外进入 _dynamic_tracks，用于作为 Ego 采样生成训练样本
            self._dynamic_tracks[scenario_id] = meta
            num_dynamic += 1

        self.logger.info(
            "[BCExpertEnv] Loaded scenario: total_objects=%d, vehicles=%d, dynamic_vehicles=%d",
            num_total,
            num_vehicle,
            num_dynamic,
        )
        return obs

    # ------------------------------------------------------------------ #
    #  核心接口：收集当前场景的所有 (obs, action) 样本
    # ------------------------------------------------------------------ #

    def _build_vehicle_proxies_at_time(self, t: int) -> Dict[Any, _VehicleProxy]:
        """
        在给定时间步 t，为所有“在该帧有效”的车辆构造 _VehicleProxy。

        重要说明（感知盲区修复）：
            - 邻居集合必须包含静止车辆，否则网络在训练/评估时会出现“静止障碍物隐形”的分布偏移。
            - 因此这里遍历的是 self._all_tracks（所有 valid 车辆），而不是 self._dynamic_tracks。

        返回：
            proxies: dict[scenario_id, _VehicleProxy]
        """
        proxies: Dict[Any, _VehicleProxy] = {}
        for scenario_id, meta in self._all_tracks.items():
            # 时间索引需要在轨迹长度范围内，且该帧 valid 为 True
            state = meta.track["state"]
            if t < 0 or t >= len(meta.valid_mask):
                continue
            if not meta.valid_mask[t]:
                continue

            pos = np.asarray(state["position"][t, :2], dtype=np.float64)
            vel = np.asarray(state["velocity"][t, :2], dtype=np.float64)
            heading = float(state["heading"][t])

            proxies[scenario_id] = _VehicleProxy(pos, vel, heading)
        return proxies

    def collect_expert_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        遍历当前场景的所有时间步和所有“动态车辆”，生成 Behavior Cloning 训练所需的
        (obs, action) 数据对。

        逻辑：
            对每条动态轨迹 meta：
                对每个时间步 t = first_valid ... (last_valid - 1):
                    - 要求该车辆在 t 和 t+1 帧均 valid（避免越界和空洞）；
                    - 在时间 t，为场景中所有“该帧 valid 的动态车辆”构造 _VehicleProxy 集合，
                      并将当前轨迹对应的 proxy 作为 ego 车辆；
                    - 使用 extract_ego_observation(ego, map_manager, active_agents)，
                      计算当前帧的 51 维 Ego-centric 观测 obs_t；
                    - 基于 (state_t, state_{t+1}) 调用 InverseDynamics.compute_action()，
                      得到连续动作 a_t = [steering, acceleration]。

        返回：
            obs     : np.ndarray, shape = (N, 51)
            actions : np.ndarray, shape = (N, 2)
        """
        engine = getattr(self, "engine", None)
        if engine is None:
            raise RuntimeError("BCExpertEnv.collect_expert_data: engine 未初始化，请先调用 reset()。")
        map_manager = getattr(engine, "map_manager", None)
        if map_manager is None or getattr(map_manager, "current_map", None) is None:
            raise RuntimeError("BCExpertEnv.collect_expert_data: map_manager.current_map 缺失。")

        obs_list: List[np.ndarray] = []
        action_list: List[np.ndarray] = []

        # 为提高效率，可以先确定整个场景的最大时间步长度
        max_T = 0
        for meta in self._dynamic_tracks.values():
            state = meta.track["state"]
            T = int(len(state["position"]))
            if T > max_T:
                max_T = T

        # 对每条动态轨迹逐一遍历时间
        for scenario_id, meta in self._dynamic_tracks.items():
            state = meta.track["state"]
            valid = meta.valid_mask

            # 时间范围：从 first_valid 到 last_valid - 1（因为要用到 t+1）
            for t in range(meta.first_valid, meta.last_valid):
                t_next = t + 1
                if t_next >= len(valid):
                    continue
                # 需要确保该车辆在 t 与 t+1 都是 valid
                if not (valid[t] and valid[t_next]):
                    continue

                # 1) 构造当前帧 t 所有 active 车辆的 proxy（用于邻居特征）
                active_proxies = self._build_vehicle_proxies_at_time(t)
                ego_vehicle = active_proxies.get(scenario_id, None)
                if ego_vehicle is None:
                    # 理论上不应该发生，但为鲁棒起见直接跳过
                    continue

                # 2) 提取 Ego-centric 观测 obs_t（51 维）
                obs_t = extract_ego_observation(
                    vehicle=ego_vehicle,
                    map_manager=map_manager,
                    active_agents=active_proxies.values(),
                )

                # 3) 构造当前帧与下一帧的状态，进行逆动力学计算
                pos_t = np.asarray(state["position"][t, :2], dtype=np.float64)
                pos_next = np.asarray(state["position"][t_next, :2], dtype=np.float64)
                vel_t = np.asarray(state["velocity"][t, :2], dtype=np.float64)
                vel_next = np.asarray(state["velocity"][t_next, :2], dtype=np.float64)
                heading_t = float(state["heading"][t])
                heading_next = float(state["heading"][t_next])

                current_state = {
                    "position": pos_t,
                    "heading": heading_t,
                    "velocity": vel_t,
                }
                next_state = {
                    "position": pos_next,
                    "heading": heading_next,
                    "velocity": vel_next,
                }

                action_t, _info = self.inverse_dynamics.compute_action(
                    current_state=current_state,
                    next_state=next_state,
                    dt=self.dt,
                )

                obs_list.append(obs_t)
                action_list.append(action_t)

        if not obs_list:
            # 当前场景可能没有任何动态车辆，返回空数组（方便上层脚本跳过）
            return (
                np.zeros((0, OBS_DIM), dtype=np.float32),
                np.zeros((0, 2), dtype=np.float32),
            )

        obs_array = np.stack(obs_list, axis=0).astype(np.float32)
        actions_array = np.stack(action_list, axis=0).astype(np.float32)
        return obs_array, actions_array


__all__ = ["BCExpertEnv", "InverseDynamics"]

