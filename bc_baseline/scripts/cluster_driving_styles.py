"""
cluster_driving_styles.py
-------------------------

驾驶风格聚类脚本：从 Waymo/ScenarioNet 数据集中提取动态车辆轨迹的高级特征，
使用 K-Means 进行驾驶风格聚类。

支持两种模式：
    --mode elbow  : 测试 K=2~10，绘制肘部曲线，保存 elbow_curve.png、elbow_table.csv、elbow_table.png
    --mode cluster: 按指定 K 聚类，保存 style_labels.json

用法示例：
    # 肘部法确定最优 K
    python -m bc_baseline.scripts.cluster_driving_styles \\
        --waymo_dir /path/to/waymo_pkls \\
        --num_scenarios 100 \\
        --mode elbow

    # 正式聚类
    python -m bc_baseline.scripts.cluster_driving_styles \\
        --waymo_dir /path/to/waymo_pkls \\
        --num_scenarios 100 \\
        --mode cluster --k 4
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np

from bc_baseline.Env.expert_env import BCExpertEnv

# 可选依赖：进度条与聚类
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    raise ImportError(
        "cluster_driving_styles 需要 sklearn 和 matplotlib，请安装: pip install scikit-learn matplotlib"
    ) from e


# 特征维度顺序：[max_acc, min_acc, max_speed, min_ttc, avg_thw]
FEATURE_NAMES = ["max_acc", "min_acc", "max_speed", "min_ttc", "avg_thw"]
FEATURE_DIM = len(FEATURE_NAMES)

# 默认填充值：当无法计算 TTC/THW 时使用（如前方无车）
DEFAULT_MIN_TTC = 60.0  # 秒，表示无碰撞风险
DEFAULT_AVG_THW = 5.0   # 秒，典型跟车时距

# 正前方判定：与自车航向夹角小于此阈值（弧度）视为正前方
AHEAD_ANGLE_THRESHOLD = np.pi / 3  # 60 度

# 除零保护：最小速度/距离阈值
EPS_SPEED = 1e-3
EPS_DISTANCE = 1e-3
EPS_APPROACH_RATE = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 Waymo/ScenarioNet 轨迹中提取驾驶风格特征并进行 K-Means 聚类。"
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
        "--mode",
        type=str,
        required=True,
        choices=["elbow", "cluster"],
        help="运行模式：elbow 绘制肘部曲线，cluster 执行聚类并保存标签。",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="聚类数 K（仅 cluster 模式必需，例如 3 或 4）。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录，默认 bc_baseline/outputs。",
    )
    return parser.parse_args()


def _speed(vel: np.ndarray) -> float:
    """计算二维速度向量的模长。"""
    return float(np.linalg.norm(vel))


def _wrap_angle(x: float) -> float:
    """将角度规范化到 [-pi, pi]。"""
    return float(np.arctan2(np.sin(x), np.cos(x)))


def compute_trajectory_features(
    env: BCExpertEnv,
    scenario_index: int,
    track_id: Any,
    meta: Any,
    dt: float,
) -> np.ndarray | None:
    """
    对单条动态轨迹计算高级特征标量。

    特征：
        max_acc, min_acc : 基于相邻帧速度求导
        max_speed        : 轨迹最大速度
        min_ttc, avg_thw : 每帧寻找正前方最近车辆，计算 TTC 和 THW 的 min/avg

    返回：
        shape (5,) 的特征向量，若轨迹过短无法计算则返回 None。
    """
    state = meta.track["state"]
    valid = meta.valid_mask

    positions = np.asarray(state["position"], dtype=np.float64)
    velocities = np.asarray(state["velocity"], dtype=np.float64)
    headings = np.asarray(state["heading"], dtype=np.float64)

    # 确保 2D
    if positions.ndim == 1:
        positions = positions.reshape(-1, 2)
    if velocities.ndim == 1:
        velocities = velocities.reshape(-1, 2)
    if headings.ndim > 1:
        headings = headings.ravel()

    T = len(valid)
    if T < 2:
        return None

    # ----- 1. 加速度：a = (v_{t+1} - v_t) / dt -----
    speeds = np.linalg.norm(velocities, axis=1)
    accs = []
    for t in range(T - 1):
        if not (valid[t] and valid[t + 1]):
            continue
        v_curr = speeds[t]
        v_next = speeds[t + 1]
        if dt < 1e-6:
            continue
        acc = (v_next - v_curr) / dt
        accs.append(acc)

    if not accs:
        return None
    max_acc = float(np.max(accs))
    min_acc = float(np.min(accs))

    # ----- 2. 最大速度 -----
    valid_speeds = speeds[valid]
    if len(valid_speeds) == 0:
        return None
    max_speed = float(np.max(valid_speeds))

    # ----- 3. TTC 与 THW：每帧寻找正前方最近有效车辆 -----
    ttc_list: List[float] = []
    thw_list: List[float] = []

    for t in range(T):
        if not valid[t]:
            continue

        ego_pos = positions[t, :2]
        ego_vel = velocities[t, :2]
        ego_heading = headings[t]
        ego_speed = _speed(ego_vel)

        # 自车朝向单位向量（车头方向）
        ego_dir = np.array([np.cos(ego_heading), np.sin(ego_heading)], dtype=np.float64)

        best_dist = np.inf
        best_ttc = None
        best_thw = None

        # 遍历所有其他车辆（_all_tracks 中排除自车）
        for other_id, other_meta in env._all_tracks.items():
            if other_id == track_id:
                continue
            if t >= len(other_meta.valid_mask) or not other_meta.valid_mask[t]:
                continue

            other_state = other_meta.track["state"]
            other_pos = np.asarray(other_state["position"][t, :2], dtype=np.float64)
            other_vel = np.asarray(other_state["velocity"][t, :2], dtype=np.float64)

            vec_to_other = other_pos - ego_pos
            dist = float(np.linalg.norm(vec_to_other))
            if dist < EPS_DISTANCE:
                continue

            # 判定是否在正前方：与自车航向夹角小于阈值（夹角较小）
            unit_vec = vec_to_other / dist
            cos_angle = float(np.dot(unit_vec, ego_dir))
            if cos_angle < np.cos(AHEAD_ANGLE_THRESHOLD):
                continue
            # 更严格：夹角 < 60 度，即 cos > 0.5
            # 等价于 angle < pi/3

            # 在正前方，且距离更近则更新
            if dist >= best_dist:
                continue

            # 计算 TTC：相对距离 / 接近速率
            # 接近速率 = ego 向 other 方向的速度分量 - other 向 ego 方向的速度分量
            # 简化：approach_rate = dot(ego_vel - other_vel, unit_vec_to_other)
            rel_vel = ego_vel - other_vel
            approach_rate = float(np.dot(rel_vel, unit_vec))
            if approach_rate < EPS_APPROACH_RATE:
                # 未在接近，TTC 无效
                continue

            ttc = dist / approach_rate
            if ttc <= 0 or ttc > 120:  # 过滤异常值
                continue

            # 计算 THW：相对距离 / 本车速度
            if ego_speed < EPS_SPEED:
                thw = DEFAULT_AVG_THW
            else:
                thw = dist / ego_speed
                if thw > 30:  # 过滤过大的 THW
                    thw = DEFAULT_AVG_THW

            best_dist = dist
            best_ttc = ttc
            best_thw = thw

        if best_ttc is not None and best_thw is not None:
            ttc_list.append(best_ttc)
            thw_list.append(best_thw)

    # 汇总 TTC 与 THW
    if ttc_list:
        min_ttc = float(np.min(ttc_list))
        avg_thw = float(np.mean(thw_list))
    else:
        min_ttc = DEFAULT_MIN_TTC
        avg_thw = DEFAULT_AVG_THW

    return np.array([max_acc, min_acc, max_speed, min_ttc, avg_thw], dtype=np.float64)


def collect_all_features(
    waymo_dir: str,
    num_scenarios: int,
    start_index: int,
    waymo_dt: float,
) -> Tuple[np.ndarray, List[Tuple[int, Any]]]:
    """
    遍历指定数据集，收集所有动态轨迹的特征向量。

    返回：
        features: np.ndarray, shape (N, 5)
        meta_list: List[(scenario_index, track_id)]，与 features 行一一对应
    """
    waymo_dir = os.path.abspath(waymo_dir)
    features_list: List[np.ndarray] = []
    meta_list: List[Tuple[int, Any]] = []

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
                f"[cluster_driving_styles] 场景 {idx} 加载失败: "
                f"{msg if msg else 'Insufficient scenarios!'}"
            )
            break

        for track_id, meta in env._dynamic_tracks.items():
            feat = compute_trajectory_features(env, idx, track_id, meta, env.dt)
            if feat is not None:
                features_list.append(feat)
                meta_list.append((idx, track_id))

        env.close()

    if not features_list:
        raise RuntimeError(
            "未提取到任何有效轨迹特征，请检查数据路径和场景数量。"
        )

    features = np.stack(features_list, axis=0)
    return features, meta_list


def run_elbow_mode(
    features: np.ndarray,
    output_path: str,
) -> None:
    """测试 K=2~10，绘制肘部曲线并保存；同时输出对应的 K-SSE 表格（CSV + 表格图）。"""
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    k_range = list(range(2, 11))
    sse_list: List[float] = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        sse_list.append(km.inertia_)

    output_dir = os.path.dirname(output_path)

    # 1. 保存肘部曲线图
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, sse_list, "bo-", linewidth=2, markersize=8)
    plt.xlabel("Number of Clusters (K)", fontsize=12)
    plt.ylabel("SSE (Sum of Squared Errors)", fontsize=12)
    plt.title("Elbow Curve: Driving Style Clustering", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(k_range)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[cluster_driving_styles] 肘部曲线已保存至: {output_path}")

    # 2. 保存 K-SSE 表格为 CSV
    table_csv_path = os.path.join(output_dir, "elbow_table.csv")
    with open(table_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["K", "SSE"])
        for k, sse in zip(k_range, sse_list):
            writer.writerow([k, f"{sse:.2f}"])
    print(f"[cluster_driving_styles] K-SSE 表格(CSV)已保存至: {table_csv_path}")

    # 3. 保存表格为图片
    table_png_path = os.path.join(output_dir, "elbow_table.png")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    table_data = [[str(k), f"{sse:.2f}"] for k, sse in zip(k_range, sse_list)]
    table = ax.table(
        colLabels=["K", "SSE"],
        cellText=table_data,
        loc="center",
        cellLoc="center",
        colColours=["#f0f0f0", "#f0f0f0"],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)
    plt.title("Elbow Method: K vs SSE", fontsize=12)
    plt.tight_layout()
    plt.savefig(table_png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[cluster_driving_styles] K-SSE 表格图已保存至: {table_png_path}")


def run_cluster_mode(
    features: np.ndarray,
    meta_list: List[Tuple[int, Any]],
    k: int,
    output_path: str,
) -> None:
    """按指定 K 聚类，构建嵌套字典并保存 JSON。"""
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    # 构建 {scenario_index: {track_id: cluster_label}}
    result: Dict[int, Dict[Any, int]] = {}
    for (scenario_index, track_id), label in zip(meta_list, labels):
        if scenario_index not in result:
            result[scenario_index] = {}
        # 确保 track_id 可 JSON 序列化
        tid = track_id if isinstance(track_id, (int, str, float)) else str(track_id)
        result[scenario_index][tid] = int(label)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"[cluster_driving_styles] 聚类标签已保存至: {output_path}")
    print(f"    聚类数 K={k}, 轨迹总数={len(meta_list)}")


def main():
    args = parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "outputs",
        )
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if args.mode == "cluster" and args.k is None:
        raise ValueError("cluster 模式下必须指定 --k 参数（例如 --k 3 或 --k 4）")

    print("[cluster_driving_styles] 开始提取轨迹特征...")
    features, meta_list = collect_all_features(
        waymo_dir=args.waymo_dir,
        num_scenarios=args.num_scenarios,
        start_index=args.start_index,
        waymo_dt=args.waymo_dt,
    )
    print(f"[cluster_driving_styles] 共提取 {features.shape[0]} 条轨迹特征")

    if args.mode == "elbow":
        output_path = os.path.join(output_dir, "elbow_curve.png")
        run_elbow_mode(features, output_path)
    else:
        output_path = os.path.join(output_dir, "style_labels.json")
        run_cluster_mode(features, meta_list, args.k, output_path)


if __name__ == "__main__":
    main()
