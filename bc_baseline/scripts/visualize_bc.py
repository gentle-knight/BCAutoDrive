from __future__ import annotations

import argparse
import os
import time
from typing import Dict

import numpy as np
import torch

from bc_baseline.Algorithm.bc_net import BCActor
from bc_baseline.Env.eval_env import BCEvalEnv


"""
visualize_bc.py
----------------

用途：
    对单个 Waymo 场景进行**可视化闭环回放**，让训练好的 BC 策略接管 Ego 车辆，
    并通过 3D 渲染（和可选的 top-down 视角）肉眼观察 Ego 的驾驶行为。

特点：
    - 仅查看单个场景：通过 --scenario_index 指定（默认 0）；
    - 支持开启 MetaDrive 原生 3D 渲染窗口（config["use_render"] = True）；
    - 可选上帝视角：--top_down 时在循环内调用 env.render(mode="topdown") 并减慢帧率；
    - 在终端中实时单行刷新打印 step、网络动作和 Ego 速度。
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize BC policy behavior in a single MetaDrive+Waymo scenario.")
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
        "--scenario_index",
        type=int,
        default=0,
        help="要可视化的场景索引（默认 0）。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="推理设备：'cuda' 或 'cpu'。",
    )
    parser.add_argument(
        "--top_down",
        action="store_true",
        help="若指定，则在每个 step 内调用 env.render(mode='topdown') 并减慢帧率，方便观察轨迹。",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.05,
        help="可视化时每个 step 的暂停时间（秒），默认 0.05。",
    )
    return parser.parse_args()


def load_policy(ckpt_path: str, device: torch.device) -> BCActor:
    """
    从 checkpoint 加载 BCActor。

    Checkpoint 结构参考 train.py 中的 save_checkpoint：
        {
            "epoch": int,
            "model_state_dict": state_dict,
            ...
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
        - 'collision'     : 是否发生碰撞
        - 'out_of_road'   : 是否偏离道路
    """
    success = bool(
        info.get("arrive_dest", False)
        or info.get("arrive_destination", False)
        or info.get("success", False)
    )

    collision = bool(
        info.get("crash", False)
        or info.get("crash_vehicle", False)
        or info.get("crash_object", False)
        or info.get("crash_building", False)
        or info.get("crash_human", False)
    )

    out_of_road = bool(
        info.get("out_of_road", False)
        or info.get("out_of_lane", False)
    )

    return {
        "success": success,
        "collision": collision,
        "out_of_road": out_of_road,
    }


def get_ego_speed(env: BCEvalEnv) -> float:
    """
    获取当前 Ego 车辆的速度标量。

    策略：
        - 优先通过 env._ego_agent_id 拿到 Ego 对象；
        - 若不存在 _ego_agent_id 或对应 agent 丢失，则返回 0.0。
    """
    engine = getattr(env, "engine", None)
    if engine is None or not hasattr(engine, "agent_manager"):
        return 0.0
    ego_id = getattr(env, "_ego_agent_id", None)
    if ego_id is None:
        return 0.0
    ego_vehicle = engine.agent_manager.active_agents.get(ego_id, None)
    if ego_vehicle is None:
        return 0.0

    # MetaDrive Vehicle 通常有 speed 属性；若不存在则由 velocity 范数近似
    speed = getattr(ego_vehicle, "speed", None)
    if speed is None:
        vel = np.asarray(getattr(ego_vehicle, "velocity", np.zeros(2)), dtype=np.float64)
        speed = float(np.linalg.norm(vel))
    return float(speed)


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

    # 2) 构造单场景可视化环境（开启 3D 渲染）
    config = BCEvalEnv.default_config()
    config.update(
        dict(
            data_directory=waymo_dir,
            start_scenario_index=int(args.scenario_index),
            num_scenarios=1,
            use_render=True,  # 打开 MetaDrive 3D 渲染窗口
        )
    )
    env = BCEvalEnv(config)

    # 3) 运行单个 episode，并在终端和窗口中实时可视化
    obs = env.reset(seed=int(args.scenario_index))
    done = False
    step_count = 0
    last_info: Dict = {}

    print(f"[visualize_bc] Start scenario index {args.scenario_index}")

    while not done:
        step_count += 1

        # 将 51 维观测送入 BCActor，得到动作 [steering, acceleration]
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)  # (1, 51)
        with torch.no_grad():
            action_tensor = policy(obs_tensor)  # (1, 2)，范围 [-1, 1]
        action = action_tensor.squeeze(0).cpu().numpy().astype(np.float32)  # (2,)

        # 与环境交互一步（BCEvalEnv.step 已将父类多种返回格式统一为 obs, reward, done, info）
        obs, reward, done, info = env.step(action)
        last_info = info

        # 计算当前 Ego 速度
        ego_speed = get_ego_speed(env)

        # 终端单行刷新打印信息
        msg = (
            f"[visualize_bc] Step {step_count:04d} | "
            f"Steering: {float(action[0]): .3f} | Accel: {float(action[1]): .3f} | "
            f"Speed: {ego_speed: .3f} m/s"
        )
        print(msg, end="\r", flush=True)

        # 3D 渲染由 MetaDrive 内部负责；若用户启用 top_down，则额外绘制上帝视角
        if args.top_down:
            try:
                # top-down 渲染模式，具体字符串取决于 MetaDrive 版本，一般为 "topdown"
                env.render(mode="topdown")
            except TypeError:
                # 某些版本 render(mode=...) 签名不同，简单 fallback 为无参 render
                env.render()

        # 减慢仿真速度，便于人眼观察
        time.sleep(max(float(args.sleep), 0.0))

    env.close()

    # 换行，避免最后一行覆盖终端提示
    print()

    # 4) Episode 结束后的结果反馈
    flags = extract_episode_flags(last_info if isinstance(last_info, dict) else {})
    result = "Success"
    if flags["collision"]:
        result = "Collision"
    elif flags["out_of_road"]:
        result = "Out of Road"

    print("========== BC Visualization Result ==========")
    print(f"Scenario Index    : {args.scenario_index}")
    print(f"Total Steps       : {step_count}")
    print(f"Episode Result    : {result}")
    print(f"  - success       : {flags['success']}")
    print(f"  - collision     : {flags['collision']}")
    print(f"  - out_of_road   : {flags['out_of_road']}")
    print("=============================================")


if __name__ == "__main__":
    main()

