"""
cluster_driving_styles.py
--------------------------
从预先提取的 interaction episode 特征文件中，
使用 K-Means 对驾驶风格进行聚类分析。

依赖上游输出（需先运行 extract_interaction_episodes.py）：
  episode_features.npz  shape (N, 8)，8维特征矩阵
  episode_meta.json     List[Dict]，每条 episode 的元信息

特征顺序（8维）：
  0: mean_acc           全程纵向加速度均值
  1: min_acc            全程最大制动
  2: jerk_peak          全程 jerk 峰值绝对值
  3: response_mean_acc  t_peak 窗口内平均加速度（最能反映风格）
  4: response_min_acc   t_peak 窗口内最小加速度
  5: mean_thw           平均跟车时间距
  6: mean_speed_ratio   ego/partner 速度比
  7: relative_speed     ego 均值速度 - partner 均值速度

聚类使用 3 维核心特征（indices [3,5,6]）：response_mean_acc、mean_thw、
mean_speed_ratio；并对物理不可能离群点做过滤后再聚类。

支持两种模式：
  --mode elbow  : 测试 K=2~10，输出三联评估图（SSE/Silhouette/DBI）
  --mode cluster: 按指定 K 聚类，输出标签文件和聚类中心报告

用法示例：
  # 第一步：肘部法确定最优 K
  python -m bc_baseline.scripts.cluster_driving_styles \\
      --mode elbow \\
      --features_path bc_baseline/outputs/interaction_episodes/episode_features.npz \\
      --meta_path     bc_baseline/outputs/interaction_episodes/episode_meta.json

  # 第二步：正式聚类
  python -m bc_baseline.scripts.cluster_driving_styles \\
      --mode cluster --k 3 \\
      --features_path bc_baseline/outputs/interaction_episodes/episode_features.npz \\
      --meta_path     bc_baseline/outputs/interaction_episodes/episode_meta.json

输出文件（保存在 output_dir）：
  elbow 模式：
    elbow_analysis.png       SSE/Silhouette/DBI 三联图
    elbow_table.csv          K, SSE, Silhouette, DBI 四列表格
  cluster 模式：
    style_labels.json        {scenario_index: {track_id: cluster_label}}
                             同一辆车多个 episode 冲突时取众数
    episode_labels.json      每条 episode 附加 cluster_label 字段
    cluster_centers.json     聚类中心物理值 + 语义标签
    cluster_report.txt       人类可读的聚类分析报告
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    raise ImportError(
        "cluster_driving_styles 需要 sklearn 和 matplotlib，请安装: pip install scikit-learn matplotlib"
    ) from e


# -------------------------- 常量定义 -------------------------- #

# 全量特征名（与 extract_interaction_episodes.py 中定义一致）
ALL_FEATURE_NAMES: List[str] = [
    "mean_acc",
    "min_acc",
    "jerk_peak",
    "response_mean_acc",
    "response_min_acc",
    "mean_thw",
    "mean_speed_ratio",
    "relative_speed",
]

# 聚类使用的特征列索引（3 维，降低 jerk/峰值等噪声敏感高阶量对 K-Means 的绑架）
#   - index 3 (response_mean_acc)：博弈窗口内平均纵向响应
#   - index 5 (mean_thw)           ：跟车时间距
#   - index 6 (mean_speed_ratio) ：ego/partner 速度比
CLUSTER_FEATURE_INDICES: List[int] = [3, 5, 6]
CLUSTER_FEATURE_NAMES: List[str] = [
    ALL_FEATURE_NAMES[i] for i in CLUSTER_FEATURE_INDICES
]

# 物理边界过滤（全量 8 维矩阵上的列索引，在聚类前剔除不可能离群点）
IDX_RESPONSE_MEAN_ACC = 3
IDX_MEAN_SPEED_RATIO = 6
RESPONSE_MEAN_ACC_MIN = -15.0
RESPONSE_MEAN_ACC_MAX = 5.0
MEAN_SPEED_RATIO_MIN = 0.0
MEAN_SPEED_RATIO_MAX = 5.0


def _normalize_feature_names(raw_feature_names: np.ndarray) -> List[str]:
    """将 npz 中读取到的 feature_names 统一转换为 Python 字符串列表。"""
    names: List[str] = []
    for name in np.asarray(raw_feature_names).tolist():
        if isinstance(name, bytes):
            names.append(name.decode("utf-8"))
        else:
            names.append(str(name))
    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 interaction episode 特征文件进行驾驶风格 K-Means 聚类。"
    )
    parser.add_argument(
        "--features_path",
        type=str,
        required=True,
        help="episode_features.npz 路径（shape N×8）。",
    )
    parser.add_argument(
        "--meta_path",
        type=str,
        required=True,
        help="episode_meta.json 路径（List[Dict]）。",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["elbow", "cluster"],
        help="elbow：K=2~10 三联评估图；cluster：按 K 聚类并输出标签与报告。",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="聚类数 K（仅 cluster 模式必需）。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录，默认 bc_baseline/outputs/driving_style。",
    )
    return parser.parse_args()


def load_episode_data(
    features_path: str,
    meta_path: str,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    加载 episode_features.npz 与 episode_meta.json。

    返回：
        features: shape (N, 8)，float32
        meta_list: 长度为 N 的 List[Dict]
    """
    features_path = os.path.abspath(features_path)
    meta_path = os.path.abspath(meta_path)

    if not os.path.isfile(features_path):
        raise FileNotFoundError(f"未找到特征文件: {features_path}")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"未找到元信息文件: {meta_path}")

    data = np.load(features_path)
    if "features" not in data:
        raise KeyError(f"npz 中需包含键 'features'，当前键: {list(data.keys())}")
    if "feature_names" not in data:
        data.close()
        raise KeyError(
            "npz 中缺少键 'feature_names'，无法确认列语义。"
            "请使用当前版本的 extract_interaction_episodes.py 重新导出 features。"
        )

    feature_names = _normalize_feature_names(data["feature_names"])
    features = np.asarray(data["features"], dtype=np.float64)
    data.close()

    if features.ndim != 2 or features.shape[1] != 8:
        raise ValueError(
            f"特征矩阵期望 shape (N, 8)，实际为 {features.shape}"
        )
    if len(feature_names) != len(ALL_FEATURE_NAMES):
        raise ValueError(
            "feature_names 长度与当前脚本预期不一致："
            f"读取到 {len(feature_names)} 项，预期 {len(ALL_FEATURE_NAMES)} 项。"
        )
    if feature_names != ALL_FEATURE_NAMES:
        raise ValueError(
            "episode_features.npz 的 feature_names 与当前聚类脚本不一致。\n"
            f"读取到: {feature_names}\n"
            f"预期为: {ALL_FEATURE_NAMES}\n"
            "请重新运行 extract_interaction_episodes.py 导出与当前脚本一致的特征文件。"
        )

    with open(meta_path, encoding="utf-8") as f:
        meta_list = json.load(f)

    if len(meta_list) != features.shape[0]:
        raise ValueError(
            f"episode 数量不一致: meta 长度={len(meta_list)}, features 行数={features.shape[0]}"
        )

    return features, meta_list


def filter_episodes_by_physical_bounds(
    features: np.ndarray,
    meta_list: List[Dict[str, Any]],
) -> Tuple[np.ndarray, List[Dict[str, Any]], int]:
    """
    在聚类前剔除物理上不可能的离群点（避免绑架 K-Means）。
    依据全量特征中的 response_mean_acc 与 mean_speed_ratio。
    """
    rma = features[:, IDX_RESPONSE_MEAN_ACC]
    msr = features[:, IDX_MEAN_SPEED_RATIO]
    mask = (
        (rma >= RESPONSE_MEAN_ACC_MIN)
        & (rma <= RESPONSE_MEAN_ACC_MAX)
        & (msr >= MEAN_SPEED_RATIO_MIN)
        & (msr <= MEAN_SPEED_RATIO_MAX)
    )
    keep = np.flatnonzero(mask)
    n_drop = int(features.shape[0] - keep.size)
    if n_drop == 0:
        return features, meta_list, 0
    features_f = features[keep]
    meta_f = [meta_list[int(i)] for i in keep]
    return features_f, meta_f, n_drop


def get_cluster_features(features: np.ndarray) -> np.ndarray:
    """从 8 维特征中取出聚类用子矩阵（维度由 CLUSTER_FEATURE_INDICES 决定）。"""
    return features[:, CLUSTER_FEATURE_INDICES].astype(np.float64)


def assign_semantic_labels(centers_physical: np.ndarray, k: int) -> List[str]:
    """
    根据聚类中心的相对排名分配语义标签，而非固定阈值。
    路口博弈语义：
      - conservative：response_mean_acc 偏小（制动）且 mean_speed_ratio 偏小（相对对手更慢）
      - aggressive：response_mean_acc 偏正或趋近 0（少刹/加速）且 mean_speed_ratio 偏大（相对更快）

    实现：对第 0 列 rma、第 2 列 msr 分别做 0..K-1 升序秩，综合得分
    rank_rma + rank_msr 最小者为 conservative，最大者为 aggressive；
    其余中间簇按 mean_thw（第 1 列）从大到小标为 normal / normal_1, ...

    参数：
      centers_physical: shape (K, 3)，列顺序与 CLUSTER_FEATURE_NAMES 一致：
                        [response_mean_acc, mean_thw, mean_speed_ratio]
      k: 簇数量

    返回：
      List[str] 长度为 K
    """
    rma_values = centers_physical[:, 0]
    thw_values = centers_physical[:, 1]
    msr_values = centers_physical[:, 2]

    # 升序秩：rma/msr 越小秩越小 → 综合秩和最小 ≈ 最保守
    rank_rma = np.argsort(np.argsort(rma_values))
    rank_msr = np.argsort(np.argsort(msr_values))
    composite = rank_rma.astype(np.float64) + rank_msr.astype(np.float64)

    cons_idx = int(np.argmin(composite))
    agg_idx = int(np.argmax(composite))
    if cons_idx == agg_idx and k > 1:
        order_rma = np.argsort(rma_values)
        cons_idx = int(order_rma[0])
        agg_idx = int(order_rma[-1])

    semantic = ["normal"] * k
    semantic[cons_idx] = "conservative"
    semantic[agg_idx] = "aggressive"

    middle_indices = [i for i in range(k) if i not in (cons_idx, agg_idx)]
    if len(middle_indices) == 1:
        semantic[middle_indices[0]] = "normal"
    elif len(middle_indices) > 1:
        middle_sorted = sorted(
            middle_indices,
            key=lambda i: thw_values[i],
            reverse=True,
        )
        for rank, idx in enumerate(middle_sorted):
            semantic[idx] = f"normal_{rank + 1}"

    return semantic


def resolve_label_for_track(
    episode_indices: List[int],
    all_labels: List[int],
    meta_list: List[Dict[str, Any]],
) -> int:
    """
    给同一辆车的多个 episode 确定最终风格标签。

    规则：
      1. 取所有 episode 标签的众数
      2. 若出现平票（多个标签出现次数相同），
         在平票候选标签对应的 episode 中，
         取 min_ttc 最小的那个 episode 的标签
         （最危险时刻的风格最能代表这辆车）

    参数：
      episode_indices : 该车所有 episode 在全局列表中的索引
      all_labels      : 全局 labels 列表（长度 = 总 episode 数）
      meta_list       : 全局 meta_list（每条含 min_ttc 字段）

    返回：
      int，最终确定的 cluster_label
    """
    track_labels = [all_labels[i] for i in episode_indices]
    label_counts = Counter(track_labels)
    max_count = max(label_counts.values())
    candidates = [lbl for lbl, cnt in label_counts.items() if cnt == max_count]

    if len(candidates) == 1:
        return int(candidates[0])

    # 平票：在候选标签对应的 episode 中取 min_ttc 最小的
    best_idx = min(
        (i for i in episode_indices if all_labels[i] in candidates),
        key=lambda i: float(meta_list[i].get("min_ttc", 99.0)),
    )
    return int(all_labels[best_idx])


def run_elbow_mode(
    X: np.ndarray,
    output_dir: str,
) -> None:
    """
    测试 K=2~10，计算 SSE、Silhouette、DBI，保存三联图与 CSV 表。
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k_range = list(range(2, 11))
    sse_list: List[float] = []
    sil_list: List[float] = []
    dbi_list: List[float] = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sse_list.append(float(km.inertia_))
        sil_list.append(
            float(
                silhouette_score(
                    X_scaled,
                    labels,
                    sample_size=min(5000, len(X_scaled)),
                    random_state=42,
                )
            )
        )
        dbi_list.append(float(davies_bouldin_score(X_scaled, labels)))

    # 三联图
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(k_range, sse_list, "bo-", linewidth=2, markersize=8)
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("SSE")
    axes[0].set_title("Sum of Squared Errors")
    axes[0].set_xticks(k_range)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(k_range, sil_list, "go-", linewidth=2, markersize=8)
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("Silhouette")
    axes[1].set_title("Silhouette Score (higher better)")
    axes[1].set_xticks(k_range)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(k_range, dbi_list, "ro-", linewidth=2, markersize=8)
    axes[2].set_xlabel("K")
    axes[2].set_ylabel("DBI")
    axes[2].set_title("Davies-Bouldin Index (lower better)")
    axes[2].set_xticks(k_range)
    axes[2].grid(True, alpha=0.3)

    best_sil_k = k_range[int(np.argmax(sil_list))]
    best_dbi_k = k_range[int(np.argmin(dbi_list))]
    # 使用 ASCII 标题：默认 DejaVu Sans 不含中文字形，中文 suptitle 会触发大量 Glyph missing 警告
    fig.suptitle(
        f"Suggested K = {best_sil_k} (max Silhouette) "
        f"or K = {best_dbi_k} (min DBI)",
        fontsize=12,
        y=1.03,
    )

    plt.tight_layout()
    analysis_path = os.path.join(output_dir, "elbow_analysis.png")
    plt.savefig(analysis_path, dpi=150)
    plt.close()
    print(f"[cluster_driving_styles] 三联图已保存: {analysis_path}")

    # CSV 表：K, SSE, Silhouette, DBI
    table_path = os.path.join(output_dir, "elbow_table.csv")
    with open(table_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["K", "SSE", "Silhouette", "DBI"])
        for k, sse, sil, dbi in zip(k_range, sse_list, sil_list, dbi_list):
            writer.writerow([k, f"{sse:.4f}", f"{sil:.4f}", f"{dbi:.4f}"])
        writer.writerow([f"# 推荐: Silhouette最大 K={best_sil_k}, DBI最小 K={best_dbi_k}"])
    print(f"[cluster_driving_styles] 表格已保存: {table_path}")


def run_cluster_mode(
    X: np.ndarray,
    meta_list: List[Dict[str, Any]],
    k: int,
    output_dir: str,
) -> None:
    """
    按 K 聚类，输出 style_labels.json、episode_labels.json、
    cluster_centers.json、cluster_report.txt。
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    labels = labels.tolist()

    # 聚类中心（缩放空间）与反变换到物理空间
    centers_scaled = km.cluster_centers_
    centers_physical = scaler.inverse_transform(centers_scaled)

    # ---------- style_labels.json：scenario_index -> track_id -> 最终 label ----------
    # 收集每辆车的 episode 全局索引；平票时用 resolve_label_for_track 按 min_ttc 打破
    track_episode_indices: Dict[Tuple[int, str], List[int]] = defaultdict(list)
    for i, meta in enumerate(meta_list):
        si = int(meta["scenario_index"])
        tid = meta["ego_track_id"]
        if not isinstance(tid, (int, str, float)):
            tid = str(tid)
        track_episode_indices[(si, str(tid))].append(i)

    style_labels_serializable: Dict[str, Dict[str, int]] = {}
    for (si, tid), ep_indices in track_episode_indices.items():
        final_label = resolve_label_for_track(ep_indices, labels, meta_list)
        si_str = str(si)
        if si_str not in style_labels_serializable:
            style_labels_serializable[si_str] = {}
        style_labels_serializable[si_str][tid] = final_label

    style_path = os.path.join(output_dir, "style_labels.json")
    with open(style_path, "w", encoding="utf-8") as f:
        json.dump(style_labels_serializable, f, indent=2, ensure_ascii=False)
    print(f"[cluster_driving_styles] style_labels 已保存: {style_path}")

    # ---------- episode_labels.json：每条 meta 附加 cluster_label ----------
    episode_labels = []
    for i, meta in enumerate(meta_list):
        row = dict(meta)
        row["cluster_label"] = int(labels[i])
        episode_labels.append(row)

    episode_path = os.path.join(output_dir, "episode_labels.json")
    with open(episode_path, "w", encoding="utf-8") as f:
        json.dump(episode_labels, f, indent=2, ensure_ascii=False)
    print(f"[cluster_driving_styles] episode_labels 已保存: {episode_path}")

    # ---------- cluster_centers.json：物理中心 + 语义标签 ----------
    semantic_labels = assign_semantic_labels(centers_physical, k)
    # ---------- episode_labels.json：每条 meta 附加 cluster_label + semantic_label ----------
    episode_labels = []
    for i, meta in enumerate(meta_list):
        row = dict(meta)
        row["cluster_label"] = int(labels[i])
        row["semantic_label"] = semantic_labels[int(labels[i])]  # 新增
        episode_labels.append(row)

    episode_path = os.path.join(output_dir, "episode_labels.json")
    with open(episode_path, "w", encoding="utf-8") as f:
        json.dump(episode_labels, f, indent=2, ensure_ascii=False)
    print(f"[cluster_driving_styles] episode_labels 已保存: {episode_path}")

    centers_data: List[Dict[str, Any]] = []
    for c in range(k):
        phys = centers_physical[c].tolist()
        name = f"cluster_{c}"
        centers_data.append({
            "cluster_id": c,
            "name": name,
            "semantic": semantic_labels[c],
            "center_physical": [round(x, 4) for x in phys],
            "feature_names": CLUSTER_FEATURE_NAMES,
        })

    centers_path = os.path.join(output_dir, "cluster_centers.json")
    with open(centers_path, "w", encoding="utf-8") as f:
        json.dump(centers_data, f, indent=2, ensure_ascii=False)
    print(f"[cluster_driving_styles] cluster_centers 已保存: {centers_path}")

    # ---------- cluster_report.txt ----------
    report_lines: List[str] = [
        "========== Driving Style Cluster Report ==========",
        f"K = {k}",
        f"Total episodes = {len(meta_list)}",
        f"Features used (3-dim): {CLUSTER_FEATURE_NAMES}",
        "",
    ]
    for c in range(k):
        count = sum(1 for L in labels if L == c)
        report_lines.append(f"--- Cluster {c} ---")
        report_lines.append(f"  Count: {count} ({100.0 * count / len(labels):.1f}%)")
        report_lines.append(
            f"  Center (physical): {[round(x, 4) for x in centers_physical[c].tolist()]}"
        )
        report_lines.append(
            f"  Semantic: {centers_data[c]['semantic']}"
        )
        report_lines.append("")

    # 统计多 episode 车辆的标签一致率
    # 一致率高 = 聚类稳定，同一辆车在不同交互里风格一致
    # 一致率低 = 需要关注，可能同一辆车在不同场景下风格差异大
    multi_track_label_lists = [
        [labels[i] for i in ep_indices]
        for ep_indices in track_episode_indices.values()
        if len(ep_indices) >= 2
    ]
    if multi_track_label_lists:
        consistent_count = sum(
            1 for ls in multi_track_label_lists if len(set(ls)) == 1
        )
        total_multi = len(multi_track_label_lists)
        consistency_rate = consistent_count / total_multi
        report_lines.append("--- 标签一致性分析 ---")
        report_lines.append(
            f"  拥有 >=2 个 episode 的车辆数: {total_multi}"
        )
        report_lines.append(
            f"  所有 episode 标签相同的车辆数: {consistent_count}"
        )
        report_lines.append(
            f"  标签一致率: {consistency_rate:.1%}"
        )
        report_lines.append(
            "  （一致率 >70% 说明聚类稳定，<50% 建议检查聚类质量）"
        )
        report_lines.append("")
    else:
        report_lines.append("  （所有车辆均只有1个 episode，无法计算一致率）")
        report_lines.append("")

    report_path = os.path.join(output_dir, "cluster_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"[cluster_driving_styles] cluster_report 已保存: {report_path}")


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(project_root, "outputs", "driving_style")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if args.mode == "cluster" and args.k is None:
        raise ValueError("cluster 模式下必须指定 --k")

    print("[cluster_driving_styles] 加载 episode 特征与元信息...")
    features, meta_list = load_episode_data(args.features_path, args.meta_path)
    features, meta_list, n_drop = filter_episodes_by_physical_bounds(
        features, meta_list
    )
    if n_drop > 0:
        print(
            f"[cluster_driving_styles] 物理边界过滤剔除 {n_drop} 条 episode "
            f"(response_mean_acc∈[{RESPONSE_MEAN_ACC_MIN},{RESPONSE_MEAN_ACC_MAX}], "
            f"mean_speed_ratio∈[{MEAN_SPEED_RATIO_MIN},{MEAN_SPEED_RATIO_MAX}])"
        )
    X = get_cluster_features(features)
    print(f"[cluster_driving_styles] 共 {X.shape[0]} 条 episode，聚类特征维度={X.shape[1]}")

    if args.mode == "elbow":
        run_elbow_mode(X, output_dir)
    else:
        run_cluster_mode(X, meta_list, args.k, output_dir)


if __name__ == "__main__":
    main()
