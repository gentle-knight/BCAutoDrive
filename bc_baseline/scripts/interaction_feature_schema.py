"""
interaction_feature_schema.py
------------------------------
Interaction episode 特征 schema 的共享定义（v1 / v2），供
extract_interaction_episodes、cluster_driving_styles、可视化脚本等导入，
避免列名与聚类列索引重复硬编码。
"""

from __future__ import annotations

from typing import List, Literal

FeatureSchemaId = Literal["v1", "v2"]

# ---------- v1：原 8 维（行为/交互统计），聚类默认用列 3,5,6 ----------
FEATURE_NAMES_V1: List[str] = [
    "mean_acc",
    "min_acc",
    "jerk_peak",
    "response_mean_acc",
    "response_min_acc",
    "mean_thw",
    "mean_speed_ratio",
    "relative_speed",
]

NPZ_VERSION_V1 = "interaction_episode_v1"

# 与旧版 cluster_driving_styles 一致：3 维子空间
CLUSTER_FEATURE_INDICES_V1: List[int] = [3, 5, 6]

# ---------- v2：速度/加速度各 mean,max,min,std，聚类用全部 8 维 ----------
FEATURE_NAMES_V2: List[str] = [
    "speed_mean",
    "speed_max",
    "speed_min",
    "speed_std",
    "acc_mean",
    "acc_max",
    "acc_min",
    "acc_std",
]

NPZ_VERSION_V2 = "interaction_episode_v2"

CLUSTER_FEATURE_INDICES_V2: List[int] = list(range(8))


def feature_names(schema: FeatureSchemaId) -> List[str]:
    return FEATURE_NAMES_V1 if schema == "v1" else FEATURE_NAMES_V2


def npz_version_string(schema: FeatureSchemaId) -> str:
    return NPZ_VERSION_V1 if schema == "v1" else NPZ_VERSION_V2


def cluster_feature_indices(schema: FeatureSchemaId) -> List[int]:
    return (
        list(CLUSTER_FEATURE_INDICES_V1)
        if schema == "v1"
        else list(CLUSTER_FEATURE_INDICES_V2)
    )


def parse_feature_schema_arg(s: str) -> FeatureSchemaId:
    key = s.strip().lower()
    if key not in ("v1", "v2"):
        raise ValueError(
            f"feature_schema 须为 v1 或 v2，收到: {s!r}"
        )
    return key  # type: ignore[return-value]
