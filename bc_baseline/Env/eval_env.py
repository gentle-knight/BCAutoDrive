from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
from metadrive.envs.scenario_env import ScenarioEnv

from bc_baseline.Env.utils import extract_ego_observation


"""
BCEvalEnv: 用于 Behavior Cloning 模型的闭环验证环境
--------------------------------------------------

设计目标：
    - 复用 MetaDrive 的 ScenarioEnv 机制，让**所有背景车辆严格按真实轨迹进行 Log-replay**；
    - 只将 SDC / Ego 车辆交给外部策略（例如训练好的 BCActor）控制；
    - 将环境对外暴露的观测统一为 Phase 1 定义的 51 维 Ego-centric 向量。

关键点：
    - 不改动 MetaDrive 的 traffic_manager / background replay 逻辑；
    - 仅重写 reset() 与 step() 的观测输出，将其替换为 extract_ego_observation 的结果；
    - 默认控制对象为 MetaDrive 中的 "default_agent"（SDC 车辆）。
"""


class BCEvalEnv(ScenarioEnv):
    """
    Behavior Cloning 闭环验证环境。

    使用方式示例：
        config = BCEvalEnv.default_config()
        config.update(
            dict(
                data_directory="/path/to/waymo_pkls",
                start_scenario_index=i,
                num_scenarios=1,
            )
        )
        env = BCEvalEnv(config)
        obs = env.reset(seed=i)  # obs 为 51 维 np.ndarray
        done = False
        while not done:
            action = policy(obs)  # action: [steering, acceleration] in [-1, 1]
            obs, reward, done, info = env.step(action)
    """

    @classmethod
    def default_config(cls) -> Dict[str, Any]:
        config = super().default_config()
        # 保持与 BCExpertEnv 一致的关键字段（data_directory / start_scenario_index / num_scenarios）
        config.update(
            dict(
                data_directory=None,
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

        # 缓存一次选定的 Ego 车辆 ID，避免每步都重复搜索
        self._ego_agent_id: str | None = None

    # ------------------------------------------------------------------ #
    #  工具函数：基于当前引擎状态构造 51 维 Ego 观测
    # ------------------------------------------------------------------ #

    def _build_ego_obs(self) -> np.ndarray:
        """
        使用 Phase 1 的 extract_ego_observation 构造 51 维自车观测。

        约定：
            - 控制的自车为 MetaDrive 默认的 "default_agent"；
            - 邻居集合使用 engine.agent_manager.active_agents 中的全部车辆（包括背景车）。
        """
        engine = getattr(self, "engine", None)
        if engine is None:
            raise RuntimeError("BCEvalEnv: engine 未初始化。")

        agent_manager = engine.agent_manager
        map_manager = getattr(engine, "map_manager", None)
        if map_manager is None or getattr(map_manager, "current_map", None) is None:
            raise RuntimeError("BCEvalEnv: map_manager.current_map 缺失。")

        active_agents = agent_manager.active_agents
        if not active_agents:
            raise RuntimeError("BCEvalEnv: 当前场景中不存在任何 active_agents。")

        # ---------- 1. 若之前已选定 Ego ID，优先复用 ----------
        if self._ego_agent_id is not None and self._ego_agent_id in active_agents:
            ego_id = self._ego_agent_id
            ego_vehicle = active_agents[ego_id]
        else:
            ego_id = None
            ego_vehicle = None

            # ---------- 2. 优先尝试硬编码的 'default_agent'（MetaDrive 默认 SDC ID） ----------
            if "default_agent" in active_agents:
                ego_id = "default_agent"
                ego_vehicle = active_agents[ego_id]

            # ---------- 3. 若不存在 'default_agent'，尝试基于 sdc_scenario_id 做一次匹配 ----------
            if ego_vehicle is None and hasattr(engine, "traffic_manager"):
                sdc_sid = getattr(engine.traffic_manager, "sdc_scenario_id", None)
                if sdc_sid is not None:
                    sdc_sid_str = str(sdc_sid)

                    # 3.1 直接匹配 key 等于 scenario_id
                    if sdc_sid_str in active_agents:
                        ego_id = sdc_sid_str
                        ego_vehicle = active_agents[ego_id]
                    else:
                        # 3.2 兼容常见前缀形式，例如 "controlled_<sid>"、"sdc_<sid>" 等
                        candidate_ids = []
                        for aid in active_agents.keys():
                            aid_str = str(aid)
                            if (
                                aid_str.endswith(sdc_sid_str)
                                or f"_{sdc_sid_str}" in aid_str
                                or "sdc" in aid_str.lower()
                            ):
                                candidate_ids.append(aid_str)
                        if candidate_ids:
                            # 简单选择字典序最短/最小的一个，作为稳定的 Ego ID
                            ego_id = sorted(candidate_ids, key=len)[0]
                            ego_vehicle = active_agents[ego_id]

            # ---------- 4. 仍未找到时，使用第一个 active agent 作为最终 Fallback ----------
            if ego_vehicle is None:
                ego_id, ego_vehicle = next(iter(active_agents.items()))
                self.logger.warning(
                    "BCEvalEnv: 未找到 'default_agent' 或基于 sdc_scenario_id 匹配的 Ego，"
                    "回退为 active_agents 中的第一辆车作为 Ego（id=%s）。",
                    ego_id,
                )

            # 记录下来，后续步骤直接复用
            self._ego_agent_id = str(ego_id)

        # active_agents.values() 中包含 ego + 所有背景车辆，满足“邻居包含背景车”的要求
        obs = extract_ego_observation(
            vehicle=ego_vehicle,
            map_manager=map_manager,
            active_agents=active_agents.values(),
        )
        return obs

    # ------------------------------------------------------------------ #
    #  对外接口：reset / step 均返回 51 维观测
    # ------------------------------------------------------------------ #

    def reset(self, seed: int | None = None, options: Dict[str, Any] | None = None):
        """
        重置场景：
            - 由父类 ScenarioEnv 完成地图和 traffic replay 的初始化；
            - 然后用 extract_ego_observation 生成 51 维观测返回。
        """
        super().reset(seed=seed)
        obs = self._build_ego_obs()
        return obs

    def step(self, action):
        """
        单步仿真：
            - 外部提供的 action 应为长度 2 的向量 [steering, acceleration]，范围 [-1, 1]；
            - 直接交由 ScenarioEnv 进行一步仿真（默认控制 "default_agent"）；
            - 然后基于更新后的引擎状态计算下一时刻 51 维观测。

        注意：
            - 这里假设 MetaDrive 配置的动作空间与 BCActor 输出对齐（二维连续、范围 [-1, 1]）。
            - 若后续发现动作维度不同，可在此处加一层简单映射（例如 acc -> throttle/brake）。
        """
        # 调用父类的 step 执行真实仿真
        # 注意：不同版本的 MetaDrive / Gym 可能返回 4 元组或 5 元组：
        #   - 旧版 Gym 接口:      obs, reward, done, info
        #   - 新版 Gymnasium 接口: obs, reward, terminated, truncated, info
        step_ret = super().step(action)
        if isinstance(step_ret, tuple) and len(step_ret) == 4:
            _obs_raw, reward, done, info = step_ret
        elif isinstance(step_ret, tuple) and len(step_ret) == 5:
            _obs_raw, reward, terminated, truncated, info = step_ret
            done = bool(terminated or truncated)
        else:
            raise RuntimeError(
                f"BCEvalEnv.step: 不支持的父类 step 返回格式（len={len(step_ret)}），"
                "请检查 MetaDrive / Gym 版本。"
            )

        # 忽略父类的观测，统一返回 51 维 Ego-centric 观测
        obs = self._build_ego_obs()
        return obs, reward, done, info


__all__ = ["BCEvalEnv"]

