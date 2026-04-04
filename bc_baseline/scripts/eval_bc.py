import argparse
import os
from typing import Dict

import numpy as np
import torch

from bc_baseline.Algorithm.bc_net import BCActor
from bc_baseline.Env.eval_env import BCEvalEnv


"""
eval_bc.py
----------

闭环评估脚本：加载训练好的 BC 模型，在 MetaDrive+Waymo 场景中让策略接管 Ego 车辆，
并统计：Success Rate / Collision Rate / Out of Road Rate。

数据流：
    - 输入：Waymo->ScenarioNet 的 .pkl 场景（由 MetaDrive 在 ScenarioEnv 中读取）；
    - 策略：BCActor (51 -> 2)，输出 [steering, acceleration]，范围 [-1, 1]；
    - 环境：BCEvalEnv，背景车 log-replay，Ego 用策略输出的 action 控制。
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained BC policy in MetaDrive+Waymo scenarios.")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="训练完成后的模型权重路径（.pth），例如 bc_baseline/outputs/checkpoints/best_bc.pth。",
    )
    parser.add_argument(
        "--waymo_dir",
        type=str,
        required=True,
        help="Waymo->ScenarioNet .pkl 数据目录（MetaDrive data_directory）。",
    )
    parser.add_argument(
        "--num_scenarios",
        type=int,
        required=True,
        help="评估的测试场景数量。",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="起始场景索引（默认为 0）。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="推理设备：'cuda' 或 'cpu'。",
    )
    return parser.parse_args()


def load_policy(ckpt_path: str, device: torch.device) -> BCActor:
    """
    加载训练好的 BCActor。

    Checkpoint 结构参考 train.py 中的 save_checkpoint：
        {
            "epoch": int,
            "model_state_dict": state_dict,
            "optimizer_state_dict": ...,
            "best_val_loss": float,
            "config": dict,
        }
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)

    model = BCActor(obs_dim=51, hidden_dim=256, action_dim=2)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def extract_episode_flags(info: Dict) -> Dict[str, bool]:
    """
    从 MetaDrive 的 info 字典中抽取我们关心的事件标志：
        - 'success'       : 是否成功到达终点（arrive_dest）
        - 'collision'     : 是否发生碰撞（crash / crash_vehicle / crash_object / crash_building 等）
        - 'out_of_road'   : 是否偏离道路（out_of_road）

    说明：
        - MetaDrive 不同版本可能在 info 中使用略微不同的 key，这里做了尽量鲁棒的合并判断。
    """
    # 成功到达终点
    success = bool(
        info.get("arrive_dest", False)
        or info.get("arrive_destination", False)
        or info.get("success", False)
    )

    # 各种碰撞事件
    collision = bool(
        info.get("crash", False)
        or info.get("crash_vehicle", False)
        or info.get("crash_object", False)
        or info.get("crash_building", False)
        or info.get("crash_human", False)
    )

    # 偏离车道 / 出界
    out_of_road = bool(
        info.get("out_of_road", False)
        or info.get("out_of_lane", False)
    )

    return {
        "success": success,
        "collision": collision,
        "out_of_road": out_of_road,
    }


def main():
    args = parse_args()
    device = torch.device(args.device)

    ckpt_path = os.path.abspath(args.ckpt_path)
    waymo_dir = os.path.abspath(args.waymo_dir)

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"未找到模型权重文件：{ckpt_path}")
    if not os.path.isdir(waymo_dir):
        raise NotADirectoryError(f"Waymo 数据目录不存在：{waymo_dir}")

    # 1) 加载策略模型
    policy = load_policy(ckpt_path, device)

    # 评估统计量
    n_episodes = 0
    n_success = 0
    n_collision = 0
    n_out_of_road = 0

    # 2) 按场景索引依次评估
    for idx in range(args.start_index, args.start_index + args.num_scenarios):
        print(f"[eval_bc] Evaluating scenario index {idx} ...")

        # 为避免 ScenarioEnv 在多场景下的 map/traffic 交叉污染，建议每个场景单独构建 env
        config = BCEvalEnv.default_config()
        config.update(
            dict(
                data_directory=waymo_dir,
                start_scenario_index=idx,
                num_scenarios=1,
            )
        )
        env = BCEvalEnv(config)

        obs = env.reset(seed=idx)
        done = False
        last_info = {}

        while not done:
            # 将 51 维观测送入 BCActor，得到动作 [steering, acceleration]
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)  # (1, 51)
            with torch.no_grad():
                action_tensor = policy(obs_tensor)  # (1, 2)，范围 [-1, 1]
            action = action_tensor.squeeze(0).cpu().numpy().astype(np.float32)  # (2,)

            # 与环境交互一步
            obs, reward, done, info = env.step(action)
            last_info = info

        env.close()

        flags = extract_episode_flags(last_info)
        n_episodes += 1
        if flags["success"]:
            n_success += 1
        if flags["collision"]:
            n_collision += 1
        if flags["out_of_road"]:
            n_out_of_road += 1

        print(
            f"[eval_bc]   Episode result: "
            f"success={flags['success']}, "
            f"collision={flags['collision']}, "
            f"out_of_road={flags['out_of_road']}"
        )

    # 3) 汇总统计结果
    if n_episodes == 0:
        print("[eval_bc] 没有成功评估任何场景。")
        return

    success_rate = n_success / n_episodes
    collision_rate = n_collision / n_episodes
    out_of_road_rate = n_out_of_road / n_episodes

    print("\n========== BC Evaluation Summary ==========")
    print(f"Number of episodes          : {n_episodes}")
    print(f"Success Rate                : {success_rate:.4f}")
    print(f"Collision Rate              : {collision_rate:.4f}")
    print(f"Out of Road Rate            : {out_of_road_rate:.4f}")
    print("===========================================")


if __name__ == "__main__":
    main()

