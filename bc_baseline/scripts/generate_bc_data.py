import argparse
import os
from typing import List

import numpy as np

from bc_baseline.Env.expert_env import BCExpertEnv


"""
generate_bc_data.py
-------------------

命令行工具：从 Waymo/ScenarioNet 转换后的 .pkl 轨迹数据中，批量生成 Behavior Cloning 训练数据。

数据流回顾：
    - 输入阶段：MetaDrive / ScenarioEnv 仍然按照原流程读取 ScenarioNet 生成的 .pkl 场景文件，
      本脚本不会也不应该修改这部分逻辑；
    - 输出阶段：我们基于 BCExpertEnv + extract_ego_observation，直接在轨迹层面构造
      (obs, action) 数值矩阵：
          obs     : shape = (N, 45)
          actions : shape = (N, 2)
      最终使用 np.savez_compressed 以 .npz 格式持久化，键为 'obs' 与 'actions'。

用法示例：
    python -m bc_baseline.scripts.generate_bc_data \\
        --waymo_dir /path/to/waymo_pkls \\
        --output_dir ./bc_baseline/outputs/data \\
        --num_scenarios 100

生成结果：
    - 在 output_dir 下创建文件：
          bc_training_data.npz
      内容包含两个键：
          'obs'     -> np.ndarray, shape = (N, 45), dtype = float32
          'actions' -> np.ndarray, shape = (N, 2),  dtype = float32
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Behavioral Cloning expert dataset from Waymo/ScenarioNet pkls.")
    parser.add_argument(
        "--waymo_dir",
        type=str,
        required=True,
        help="包含 Waymo->ScenarioNet 转换后 .pkl 文件的目录（MetaDrive data_directory）。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出目录，将在此目录下生成 bc_training_data.npz。",
    )
    parser.add_argument(
        "--num_scenarios",
        type=int,
        required=True,
        help="要处理的场景数量（从 0 开始按索引顺序读取）。",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="起始场景索引（默认为 0）。",
    )
    parser.add_argument(
        "--waymo_dt",
        type=float,
        default=0.1,
        help="Waymo 轨迹时间步长（秒），默认 0.1s (10Hz)。",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    waymo_dir = os.path.abspath(args.waymo_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    all_obs: List[np.ndarray] = []
    all_actions: List[np.ndarray] = []

    # 逐场景处理：每个场景单独构建一个 BCExpertEnv，避免 MetaDrive 内部 map/traffic 状态交叉污染
    # 若用户请求的场景数量超过实际可用数量，我们在遇到 MetaDrive 抛出的
    # “Insufficient scenarios!” 断言时停止循环，并给出提示。
    for idx in range(args.start_index, args.start_index + args.num_scenarios):
        print(f"[generate_bc_data] Processing scenario index {idx} ...")

        # 构造该场景专用的环境配置
        config = BCExpertEnv.default_config()
        config.update(
            dict(
                data_directory=waymo_dir,
                start_scenario_index=idx,
                num_scenarios=1,
                waymo_dt=float(args.waymo_dt),
            )
        )

        try:
            # 创建并重置环境，加载指定场景
            env = BCExpertEnv(config)
            env.reset(seed=idx)
        except AssertionError as e:
            # MetaDrive 在 ScenarioDataManager 中若发现 start_scenario_index 超出范围，
            # 会触发 "Insufficient scenarios!" 的断言。
            msg = str(e)
            print(
                "[generate_bc_data] WARNING: 遇到 MetaDrive 抛出的断言错误："
                f"{msg if msg else 'Insufficient scenarios!'}"
            )
            print(
                "[generate_bc_data] 已处理到可用场景的末尾，将停止继续转换。"
            )
            break

        # 收集该场景内所有 (obs, action) 对
        obs, actions = env.collect_expert_data()
        env.close()

        if obs.shape[0] == 0:
            print(f"[generate_bc_data]   Scenario {idx} has no dynamic vehicles, skip.")
            continue

        print(
            f"[generate_bc_data]   Scenario {idx}: collected {obs.shape[0]} samples "
            f"(obs_dim={obs.shape[1]}, action_dim={actions.shape[1]})."
        )
        all_obs.append(obs)
        all_actions.append(actions)

    if not all_obs:
        raise RuntimeError(
            "generate_bc_data: 所有场景均未产生任何 (obs, action) 样本，请检查 Waymo 数据或过滤条件。"
        )

    # 拼接所有场景的数据
    obs_all = np.concatenate(all_obs, axis=0).astype(np.float32)
    actions_all = np.concatenate(all_actions, axis=0).astype(np.float32)

    # 以 .npz 格式保存，键为 'obs' 和 'actions'
    output_path = os.path.join(output_dir, "bc_training_data.npz")
    np.savez_compressed(output_path, obs=obs_all, actions=actions_all)

    print(
        f"[generate_bc_data] Done. Saved dataset to: {output_path}\n"
        f"    obs.shape     = {obs_all.shape}\n"
        f"    actions.shape = {actions_all.shape}"
    )


if __name__ == "__main__":
    main()

