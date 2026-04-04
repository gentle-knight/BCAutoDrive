from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

from bc_baseline.scene_labeling.schemas import ControlType, InteractionType, RoadType, RuleTraceItem, SceneFeatures, SceneLabel


@dataclass
class LabelingConfig:
    """
    所有阈值集中管理，便于后续调参。
    """

    # ---------- Control rules (traffic control) ----------
    # 若存在动态 lane state（转换后通常来自 traffic light）且其 object_state 中存在非 None，则认为 signalized
    dynamic_lane_state_min_valid_frames: int = 1
    # stop sign 触发
    stop_sign_min_count: int = 1

    # ---------- Road type rules ----------
    crossing_candidates_high: int = 2
    merge_candidates_high: int = 1

    # freeway 近似条件：并行车道比例 + 高速 + 少交叉
    parallel_lane_ratio_threshold: float = 0.70
    p90_vehicle_speed_threshold: float = 12.0  # m/s（仅作软指标/置信度参考）
    crossing_candidates_few: int = 1

    # ---------- Interaction type rules ----------
    following_max_lane_changes: int = 0
    following_max_merge_candidates: int = 0
    following_max_crossing_candidates: int = 1

    lane_change_min_lane_changes: int = 1
    lane_change_max_crossing_candidates: int = 1

    merge_min_merge_candidates: int = 2
    merge_max_crossing_candidates: int = 1

    crossing_conflict_min_candidates: int = 3

    turning_conflict_min_turning_tracks: int = 2
    turning_conflict_max_ttc: float = 8.0  # s，越小越冲突

    # ---------- Mixed decision ----------
    mixed_min_hit_count: int = 3

    # ---------- Confidence shaping ----------
    # 对“命中强度”做温和的置信度映射
    softmax_scale: float = 4.0

    # 默认：未计算到时的置信度
    unknown_confidence: float = 0.2

    # ---------- Track parsing thresholds (must stay in config zone) ----------
    # determine moving vs static
    static_displacement_threshold_m: float = 5.0
    static_speed_threshold_mps: float = 1.0

    # lane change detection (track-level)
    lane_change_lat_delta_threshold_m: float = 1.0
    lane_change_min_transition_frames: int = 5

    # merge candidate detection (track-level, rough)
    merge_lat_min_m: float = 1.0
    merge_lat_max_m: float = 4.0
    merge_monotonic_ratio_threshold: float = 0.7
    merge_heading_change_max_rad: float = math.radians(15.0)

    # turning track detection (track-level)
    turning_heading_change_min_rad: float = math.radians(20.0)

    # derived parallel lane ratio
    parallel_heading_window_rad: float = math.radians(10.0)

    # pairwise interaction computation
    candidate_interaction_distance_m: float = 8.0
    pairwise_max_vehicle_count: int = 30

    # crossing candidate: heading difference around 90 deg
    crossing_heading_diff_min_deg: float = 60.0
    crossing_heading_diff_max_deg: float = 120.0

    # TTC/THW computation knobs
    default_ttc_s: float = 60.0
    max_ttc_clip_s: float = 60.0
    ttc_approach_rate_eps: float = 1e-4

    default_thw_s: float = 10.0
    max_thw_clip_s: float = 10.0
    thw_min_speed_mps: float = 1.0


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _confidence_from_hit_strength(hit_strength: float) -> float:
    """
    将命中强度（一般为非负数）映射到 [0,1]。
    """
    if hit_strength <= 0.0:
        return 0.0
    # 软映射：hit_strength 越大，置信度越接近 1
    return _clamp01(1.0 - math.exp(-hit_strength))


def _make_trace(step: str, rule: str, condition: str, result: str, details: Dict[str, Any]) -> RuleTraceItem:
    # Confidence 待外部再注入，这里先放占位值
    return RuleTraceItem(
        step=step,
        rule=rule,
        condition=condition,
        result=result,
        details=details,
        confidence=0.0,
    )


def label_scene(scene_features: SceneFeatures, config: LabelingConfig) -> Dict[str, Any]:
    """
    按规则顺序：control_type -> road_type -> interaction_type
    返回包含：
      - control_type / road_type / interaction_type
      - 每个标签的 confidence
      - rule_trace（按步骤聚合）
    """

    global_trace: List[RuleTraceItem] = []

    # -------------------------- 1) control_type -------------------------- #
    has_dyn = bool(scene_features.get("has_dynamic_lane_state", False))
    # 若地图解析器已经给出 has_dynamic_lane_state，则这里仅按布尔触发
    if has_dyn:
        control_label: ControlType = "signalized"
        control_conf = 0.9
        trace_item = _make_trace(
            step="control_type",
            rule="signalized",
            condition="has_dynamic_lane_state == True",
            result="signalized",
            details={"has_dynamic_lane_state": has_dyn},
        )
        trace_item.confidence = control_conf
        global_trace.append(trace_item)
    else:
        stop_cnt = int(scene_features.get("num_stop_signs", 0) or 0)
        if stop_cnt >= int(config.stop_sign_min_count):
            control_label = "stop_controlled"
            control_conf = 0.75
            trace_item = _make_trace(
                step="control_type",
                rule="stop_controlled",
                condition=f"num_stop_signs >= {config.stop_sign_min_count}",
                result="stop_controlled",
                details={"num_stop_signs": stop_cnt},
            )
            trace_item.confidence = control_conf
            global_trace.append(trace_item)
        else:
            control_label = "none"
            control_conf = 0.5
            trace_item = _make_trace(
                step="control_type",
                rule="none",
                condition="else",
                result="none",
                details={"num_stop_signs": stop_cnt, "has_dynamic_lane_state": has_dyn},
            )
            trace_item.confidence = control_conf
            global_trace.append(trace_item)

    control_obj = SceneLabel(label=control_label, confidence=control_conf, rule_trace=global_trace[:])

    # -------------------------- 2) road_type -------------------------- #
    crossing_cnt = int(scene_features.get("num_crossing_candidates", 0) or 0)
    merge_cnt = int(scene_features.get("num_merge_candidates", 0) or 0)
    parallel_lane_ratio = float(scene_features.get("parallel_lane_ratio", 0.0) or 0.0)
    p90_speed = float(scene_features.get("p90_vehicle_speed", 0.0) or 0.0)
    num_crosswalks = int(scene_features.get("num_crosswalks", 0) or 0)
    num_stop_signs = int(scene_features.get("num_stop_signs", 0) or 0)
    control_is_signalized = str(control_label) == "signalized"

    road_trace_items: List[RuleTraceItem] = []
    road_label: RoadType = "urban_road"
    road_conf: float = 0.6

    # 新优先级（按顺序遇到即停止）：
    # 1) intersection（路口）
    # 2) freeway（高速）
    # 3) merge_ramp（匝道汇入）
    # 4) urban_road（兜底）

    # -------- Priority 1: intersection --------
    intersection_reasons: List[str] = []
    if num_crosswalks > 0:
        intersection_reasons.append("num_crosswalks > 0")
    if num_stop_signs > 0:
        intersection_reasons.append("num_stop_signs > 0")
    if control_is_signalized:
        intersection_reasons.append("control_type == signalized")
    if crossing_cnt >= int(config.crossing_candidates_high):
        intersection_reasons.append("num_crossing_candidates >= crossing_candidates_high")

    intersection_hit = len(intersection_reasons) > 0

    if intersection_hit:
        road_label = "intersection"
        # 置信度：把“地图证据/控制证据/轨迹冲突 proxy”综合成强度
        strength = 0.0
        if num_crosswalks > 0:
            strength += 1.0
        if num_stop_signs > 0:
            strength += 1.0
        if control_is_signalized:
            strength += 1.0
        # crossing 证据按比例叠加
        strength += float(crossing_cnt) / max(1.0, float(config.crossing_candidates_high))

        road_conf = 0.6 + 0.35 * _confidence_from_hit_strength(strength - 1.0)
        road_trace_items.append(
            _make_trace(
                step="road_type",
                rule="intersection_by_map_control_or_crossing_proxy",
                condition=(
                    "num_crosswalks > 0 OR num_stop_signs > 0 OR "
                    "control_type == signalized OR num_crossing_candidates >= crossing_candidates_high"
                ),
                result="intersection",
                details={
                    "num_crosswalks": num_crosswalks,
                    "num_stop_signs": num_stop_signs,
                    "control_type": str(control_label),
                    "num_crossing_candidates": crossing_cnt,
                    "crossing_candidates_high": int(config.crossing_candidates_high),
                    "trigger_reasons": intersection_reasons,
                },
            )
        )

    # -------- Priority 2: freeway --------
    elif (
        parallel_lane_ratio >= float(config.parallel_lane_ratio_threshold)
        and num_crosswalks == 0
        and (not control_is_signalized)
        and (crossing_cnt <= int(config.crossing_candidates_few))
    ):
        road_label = "freeway"

        # 速度不参与硬阈值（拥堵仍可能是 freeway）；只用于置信度软参考。
        lane_strength = parallel_lane_ratio / max(1e-3, float(config.parallel_lane_ratio_threshold))
        speed_strength = p90_speed / max(1e-3, float(config.p90_vehicle_speed_threshold))
        # lane 强权重，speed 轻权重
        hit_strength = 0.85 * float(lane_strength) + 0.15 * float(speed_strength)

        road_conf = 0.55 + 0.4 * _confidence_from_hit_strength(hit_strength - 1.0)
        road_trace_items.append(
            _make_trace(
                step="road_type",
                rule="freeway_by_parallel_lane_no_crosswalk_no_signal_and_crossing_few",
                condition=(
                    f"parallel_lane_ratio >= {config.parallel_lane_ratio_threshold} AND "
                    f"num_crosswalks == 0 AND control_type != signalized AND "
                    f"num_crossing_candidates <= {config.crossing_candidates_few}"
                ),
                result="freeway",
                details={
                    "parallel_lane_ratio": parallel_lane_ratio,
                    "parallel_lane_ratio_threshold": float(config.parallel_lane_ratio_threshold),
                    "num_crosswalks": num_crosswalks,
                    "control_type": str(control_label),
                    "num_crossing_candidates": crossing_cnt,
                    "crossing_candidates_few": int(config.crossing_candidates_few),
                    "p90_vehicle_speed": p90_speed,
                    "speed_strength_soft": speed_strength,
                    "lane_strength": lane_strength,
                },
            )
        )

    # -------- Priority 3: merge_ramp --------
    elif merge_cnt >= int(config.merge_candidates_high):
        road_label = "merge_ramp"
        strength = float(merge_cnt) / max(1.0, float(config.merge_candidates_high))
        road_conf = 0.6 + 0.35 * _confidence_from_hit_strength(strength - 1.0)
        road_trace_items.append(
            _make_trace(
                step="road_type",
                rule="merge_ramp_by_merge_candidates",
                condition=f"num_merge_candidates >= {config.merge_candidates_high}",
                result="merge_ramp",
                details={
                    "num_merge_candidates": merge_cnt,
                    "merge_candidates_high": int(config.merge_candidates_high),
                },
            )
        )

    # -------- Priority 4: urban_road (fallback) --------
    else:
        road_label = "urban_road"
        road_conf = 0.55
        road_trace_items.append(
            _make_trace(
                step="road_type",
                rule="default_urban_road",
                condition="fallback_else",
                result="urban_road",
                details={
                    "num_crosswalks": num_crosswalks,
                    "num_stop_signs": num_stop_signs,
                    "control_type": str(control_label),
                    "parallel_lane_ratio": parallel_lane_ratio,
                    "p90_vehicle_speed": p90_speed,
                    "num_crossing_candidates": crossing_cnt,
                    "num_merge_candidates": merge_cnt,
                    "intersection_reasons": intersection_reasons,
                },
            )
        )

    for it in road_trace_items:
        it.confidence = road_conf
        global_trace.append(it)

    road_obj = SceneLabel(label=road_label, confidence=road_conf, rule_trace=road_trace_items)

    # -------------------------- 3) interaction_type -------------------------- #
    num_lane_changes = int(scene_features.get("num_lane_changes", 0) or 0)
    num_merge_candidates = int(scene_features.get("num_merge_candidates", 0) or 0)
    num_crossing_candidates = int(scene_features.get("num_crossing_candidates", 0) or 0)
    num_turning_tracks = int(scene_features.get("num_turning_tracks", 0) or 0)
    num_candidate_interactions = int(scene_features.get("num_candidate_interactions", 0) or 0)
    mean_min_ttc = float(scene_features.get("mean_min_ttc", 0.0) or 0.0)

    following_hit = (
        num_lane_changes <= int(config.following_max_lane_changes)
        and num_merge_candidates <= int(config.following_max_merge_candidates)
        and num_crossing_candidates <= int(config.following_max_crossing_candidates)
        and num_candidate_interactions > 0
    )

    lane_change_hit = (
        num_lane_changes >= int(config.lane_change_min_lane_changes)
        and num_crossing_candidates <= int(config.lane_change_max_crossing_candidates)
    )

    merge_hit = (
        num_merge_candidates >= int(config.merge_min_merge_candidates)
        and num_crossing_candidates <= int(config.merge_max_crossing_candidates)
    )

    crossing_conflict_hit = num_crossing_candidates >= int(config.crossing_conflict_min_candidates)

    turning_conflict_hit = (
        num_turning_tracks >= int(config.turning_conflict_min_turning_tracks)
        and mean_min_ttc <= float(config.turning_conflict_max_ttc)
    )

    hits: List[Tuple[str, bool, float]] = [
        ("following", following_hit, 1.0 + 0.1 * float(max(0, -num_lane_changes))),
        ("lane_change", lane_change_hit, 1.0 + 0.1 * float(num_lane_changes)),
        ("merge", merge_hit, 1.0 + 0.2 * float(num_merge_candidates)),
        ("crossing_conflict", crossing_conflict_hit, 1.0 + 0.2 * float(num_crossing_candidates)),
        ("turning_conflict", turning_conflict_hit, 1.0 + 0.2 * float(num_turning_tracks)),
    ]

    hit_labels = [name for name, ok, _ in hits if ok]
    hit_count = len(hit_labels)

    if hit_count >= int(config.mixed_min_hit_count):
        interaction_label: InteractionType = "mixed"
        interaction_conf = 0.55 + 0.35 * _confidence_from_hit_strength(float(hit_count) - 1.0)
        global_trace.append(
            RuleTraceItem(
                step="interaction_type",
                rule="mixed_multi_rule_strong_hit",
                condition=f"num_hits >= {config.mixed_min_hit_count}",
                result="mixed",
                details={
                    "hits": hit_labels,
                    "num_lane_changes": num_lane_changes,
                    "num_merge_candidates": num_merge_candidates,
                    "num_crossing_candidates": num_crossing_candidates,
                    "num_turning_tracks": num_turning_tracks,
                    "num_candidate_interactions": num_candidate_interactions,
                    "mean_min_ttc": mean_min_ttc,
                },
                confidence=interaction_conf,
            )
        )
    else:
        # 命中数量 < mixed_min_hit_count：按照强关联的“单规则”
        if following_hit:
            interaction_label = "following"
        elif lane_change_hit:
            interaction_label = "lane_change"
        elif merge_hit:
            interaction_label = "merge"
        elif crossing_conflict_hit:
            interaction_label = "crossing_conflict"
        elif turning_conflict_hit:
            interaction_label = "turning_conflict"
        else:
            # 若没有任何强命中，给一个温和的默认（更可解释：走背景）
            interaction_label = "following"

        # 置信度由命中的规则强度粗略估计
        strength_map = {name: strength for name, ok, strength in hits}
        interaction_conf = _confidence_from_hit_strength(float(strength_map.get(interaction_label, 1.0)))
        interaction_conf = max(0.45, interaction_conf)

        global_trace.append(
            RuleTraceItem(
                step="interaction_type",
                rule="interaction_single_rule_or_default",
                condition="mixed_hit=False",
                result=interaction_label,
                details={
                    "following_hit": following_hit,
                    "lane_change_hit": lane_change_hit,
                    "merge_hit": merge_hit,
                    "crossing_conflict_hit": crossing_conflict_hit,
                    "turning_conflict_hit": turning_conflict_hit,
                    "hit_labels": hit_labels,
                    "mean_min_ttc": mean_min_ttc,
                },
                confidence=interaction_conf,
            )
        )

    interaction_obj = SceneLabel(
        label=interaction_label,
        confidence=interaction_conf,
        rule_trace=[it for it in global_trace if it.step == "interaction_type"],
    )

    return {
        "control_type": control_obj,
        "road_type": road_obj,
        "interaction_type": interaction_obj,
        "rule_trace": global_trace,
    }

