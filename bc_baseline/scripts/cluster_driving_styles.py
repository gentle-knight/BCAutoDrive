"""
cluster_driving_styles.py
--------------------------
从预先提取的 interaction episode 特征文件中，
使用 K-Means 对驾驶风格进行聚类分析。

依赖上游输出（需先运行 extract_interaction_episodes.py）：
  episode_features.npz  shape (N, 8)，8维特征矩阵
  episode_meta.json     List[Dict]，每条 episode 的元信息

须与 extract 时 --feature_schema 一致：
  v1：原 8 维行为特征；聚类默认子空间列 3,5,6（response_mean_acc, mean_thw, mean_speed_ratio）
  v2：速度/加速度各 mean,max,min,std；聚类使用全部 8 维

使用 --feature_schema v1|v2 指定；聚类前按 schema 做物理边界过滤。

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

from bc_baseline.scripts.interaction_feature_schema import (
    FeatureSchemaId,
    cluster_feature_indices,
    feature_names,
    npz_version_string,
    parse_feature_schema_arg,
)

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

# v2 物理边界过滤（聚类前剔除离群点）
SPEED_LO = 0.0
SPEED_HI = 120.0
SPEED_STD_MAX = 60.0
ACC_LO = -35.0
ACC_HI = 25.0
ACC_STD_MAX = 60.0

# v1 物理边界（全量 8 维中的列索引）
IDX_V1_RESPONSE_MEAN_ACC = 3
IDX_V1_MEAN_SPEED_RATIO = 6
V1_RESPONSE_MEAN_ACC_MIN = -15.0
V1_RESPONSE_MEAN_ACC_MAX = 5.0
V1_MEAN_SPEED_RATIO_MIN = 0.0
V1_MEAN_SPEED_RATIO_MAX = 5.0


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
    parser.add_argument(
        "--feature_schema",
        type=str,
        default="v2",
        choices=["v1", "v2"],
        help="须与 episode_features.npz 一致：v1=原 8 维+聚类列 3,5,6；"
        "v2=速度/加速度 8 统计+聚类 8 维（默认 v2）。",
    )
    return parser.parse_args()


def load_episode_data(
    features_path: str,
    meta_path: str,
    feature_schema: FeatureSchemaId,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    加载 episode_features.npz 与 episode_meta.json。
    feature_schema 须与提取时 --feature_schema 及 npz 内列名一致。
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

    names_in_file = _normalize_feature_names(data["feature_names"])
    features = np.asarray(data["features"], dtype=np.float64)
    fv_raw = data["feature_version"] if "feature_version" in data.files else None
    data.close()
    expected = feature_names(feature_schema)

    if features.ndim != 2 or features.shape[1] != 8:
        raise ValueError(
            f"特征矩阵期望 shape (N, 8)，实际为 {features.shape}"
        )
    if len(names_in_file) != len(expected):
        raise ValueError(
            "feature_names 长度与当前 schema 不一致："
            f"读取到 {len(names_in_file)} 项，预期 {len(expected)} 项。"
        )
    if names_in_file != expected:
        raise ValueError(
            "episode_features.npz 的 feature_names 与 --feature_schema 不匹配。\n"
            f"读取到: {names_in_file}\n"
            f"预期为 ({feature_schema}): {expected}\n"
            "请使用 extract_interaction_episodes.py --feature_schema "
            f"{feature_schema} 重新导出，或调整聚类时的 --feature_schema。"
        )

    if fv_raw is not None:
        fv = str(np.asarray(fv_raw).item())
        exp_ver = npz_version_string(feature_schema)
        if fv != exp_ver:
            print(
                f"[cluster_driving_styles] 警告: npz feature_version={fv!r}，"
                f"与 schema 对应 {exp_ver!r} 不一致，已以列名为准继续。"
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
    feature_schema: FeatureSchemaId,
) -> Tuple[np.ndarray, List[Dict[str, Any]], int]:
    """
    在聚类前剔除物理上不可能的离群点（避免绑架 K-Means）。
    v1：response_mean_acc、mean_speed_ratio；v2：8 维速度与加速度统计。
    """
    mask = np.isfinite(features).all(axis=1)
    if feature_schema == "v1":
        rma = features[:, IDX_V1_RESPONSE_MEAN_ACC]
        msr = features[:, IDX_V1_MEAN_SPEED_RATIO]
        mask &= (rma >= V1_RESPONSE_MEAN_ACC_MIN) & (rma <= V1_RESPONSE_MEAN_ACC_MAX)
        mask &= (msr >= V1_MEAN_SPEED_RATIO_MIN) & (msr <= V1_MEAN_SPEED_RATIO_MAX)
    else:
        mask &= (features[:, 0] >= SPEED_LO) & (features[:, 0] <= SPEED_HI)
        mask &= (features[:, 1] >= SPEED_LO) & (features[:, 1] <= SPEED_HI)
        mask &= (features[:, 2] >= SPEED_LO) & (features[:, 2] <= SPEED_HI)
        mask &= (features[:, 3] >= 0.0) & (features[:, 3] <= SPEED_STD_MAX)
        mask &= (features[:, 4] >= ACC_LO) & (features[:, 4] <= ACC_HI)
        mask &= (features[:, 5] >= ACC_LO) & (features[:, 5] <= ACC_HI)
        mask &= (features[:, 6] >= ACC_LO) & (features[:, 6] <= ACC_HI)
        mask &= (features[:, 7] >= 0.0) & (features[:, 7] <= ACC_STD_MAX)
    keep = np.flatnonzero(mask)
    n_drop = int(features.shape[0] - keep.size)
    if n_drop == 0:
        return features, meta_list, 0
    features_f = features[keep]
    meta_f = [meta_list[int(i)] for i in keep]
    return features_f, meta_f, n_drop


def get_cluster_features(
    features: np.ndarray,
    cluster_indices: List[int],
) -> np.ndarray:
    """从 8 维全量特征中取出聚类子矩阵。"""
    return features[:, cluster_indices].astype(np.float64)


def assign_semantic_labels(
    centers_physical: np.ndarray,
    k: int,
    feature_schema: FeatureSchemaId,
) -> List[str]:
    """
    语义标签规则随 feature_schema 变化。
    v1：子空间列为 response_mean_acc, mean_thw, mean_speed_ratio；
    v2：子空间为 8 维全量，使用 speed_mean、acc_mean、acc_std。
    """
    if feature_schema == "v1":
        rma_values = centers_physical[:, 0]
        thw_values = centers_physical[:, 1]
        msr_values = centers_physical[:, 2]
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

    speed_mean = centers_physical[:, 0]
    acc_mean = centers_physical[:, 4]
    acc_std = centers_physical[:, 7]
    rank_s = np.argsort(np.argsort(speed_mean))
    rank_a = np.argsort(np.argsort(acc_mean))
    composite = rank_s.astype(np.float64) + rank_a.astype(np.float64)
    cons_idx = int(np.argmin(composite))
    agg_idx = int(np.argmax(composite))
    if cons_idx == agg_idx and k > 1:
        order_s = np.argsort(speed_mean)
        cons_idx = int(order_s[0])
        agg_idx = int(order_s[-1])
    semantic = ["normal"] * k
    semantic[cons_idx] = "conservative"
    semantic[agg_idx] = "aggressive"
    middle_indices = [i for i in range(k) if i not in (cons_idx, agg_idx)]
    if len(middle_indices) == 1:
        semantic[middle_indices[0]] = "normal"
    elif len(middle_indices) > 1:
        middle_sorted = sorted(
            middle_indices,
            key=lambda i: acc_std[i],
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
    feature_schema: FeatureSchemaId,
    cluster_feature_names: List[str],
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
    semantic_labels = assign_semantic_labels(
        centers_physical, k, feature_schema
    )
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
            "feature_names": cluster_feature_names,
        })

    centers_path = os.path.join(output_dir, "cluster_centers.json")
    with open(centers_path, "w", encoding="utf-8") as f:
        json.dump(centers_data, f, indent=2, ensure_ascii=False)
    print(f"[cluster_driving_styles] cluster_centers 已保存: {centers_path}")

    # ---------- cluster_report.txt ----------
    dim_desc = f"{X.shape[1]}-dim"
    report_lines: List[str] = [
        "========== Driving Style Cluster Report ==========",
        f"feature_schema = {feature_schema}",
        f"K = {k}",
        f"Total episodes = {len(meta_list)}",
        f"Features used ({dim_desc}): {cluster_feature_names}",
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
    feature_schema = parse_feature_schema_arg(args.feature_schema)
    full_names = feature_names(feature_schema)
    c_indices = cluster_feature_indices(feature_schema)
    cluster_feature_names = [full_names[i] for i in c_indices]

    output_dir = args.output_dir
    if output_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(project_root, "outputs", "driving_style")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if args.mode == "cluster" and args.k is None:
        raise ValueError("cluster 模式下必须指定 --k")

    print(
        f"[cluster_driving_styles] 加载 episode 特征与元信息 "
        f"(feature_schema={feature_schema})..."
    )
    features, meta_list = load_episode_data(
        args.features_path, args.meta_path, feature_schema
    )
    features, meta_list, n_drop = filter_episodes_by_physical_bounds(
        features, meta_list, feature_schema
    )
    if n_drop > 0:
        if feature_schema == "v1":
            print(
                f"[cluster_driving_styles] 物理边界过滤剔除 {n_drop} 条 episode "
                f"(response_mean_acc∈[{V1_RESPONSE_MEAN_ACC_MIN},{V1_RESPONSE_MEAN_ACC_MAX}], "
                f"mean_speed_ratio∈[{V1_MEAN_SPEED_RATIO_MIN},{V1_MEAN_SPEED_RATIO_MAX}])"
            )
        else:
            print(
                f"[cluster_driving_styles] 物理边界过滤剔除 {n_drop} 条 episode "
                f"(speed∈[{SPEED_LO},{SPEED_HI}] m/s, std≤{SPEED_STD_MAX}; "
                f"acc∈[{ACC_LO},{ACC_HI}] m/s², acc_std≤{ACC_STD_MAX}; 且 finite)"
            )
    X = get_cluster_features(features, c_indices)
    print(
        f"[cluster_driving_styles] 共 {X.shape[0]} 条 episode，"
        f"聚类特征维度={X.shape[1]} ({feature_schema})"
    )

    if args.mode == "elbow":
        run_elbow_mode(X, output_dir)
    else:
        run_cluster_mode(
            X,
            meta_list,
            args.k,
            output_dir,
            feature_schema,
            cluster_feature_names,
        )


if __name__ == "__main__":
    main()
