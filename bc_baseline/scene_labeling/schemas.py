from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, TypedDict


RoadType = Literal["freeway", "urban_road", "merge_ramp", "intersection"]
InteractionType = Literal[
    "following",
    "lane_change",
    "merge",
    "crossing_conflict",
    "turning_conflict",
    "mixed",
]
ControlType = Literal["none", "stop_controlled", "signalized", "unknown"]


@dataclass
class RuleTraceItem:
    """
    单条规则命中/排除的可解释记录。
    """

    step: str
    rule: str
    condition: str
    result: str
    details: Dict[str, Any]
    confidence: float


class SceneFeatures(TypedDict, total=False):
    # ---------- Map features ----------
    num_lanes: int
    num_lane_boundaries: int
    num_road_boundaries: int
    num_crosswalks: int
    num_stop_signs: int
    num_speed_bumps: int
    has_dynamic_lane_state: bool

    # ---------- Traffic features ----------
    num_objects: int
    num_moving_vehicles: int
    mean_vehicle_speed: float
    p90_vehicle_speed: float
    heading_dispersion: float

    # ---------- Topology & interaction features ----------
    num_candidate_interactions: int
    num_lane_changes: int
    num_merge_candidates: int
    num_crossing_candidates: int
    num_turning_tracks: int
    mean_min_ttc: float
    mean_min_thw: float

    # Derived (used by rules; optional for debugging)
    parallel_lane_ratio: float


@dataclass
class SceneLabel:
    label: str
    confidence: float
    rule_trace: List[RuleTraceItem]

    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "confidence": float(self.confidence),
            "rule_trace": [
                {
                    "step": it.step,
                    "rule": it.rule,
                    "condition": it.condition,
                    "result": it.result,
                    "details": it.details,
                    "confidence": float(it.confidence),
                }
                for it in self.rule_trace
            ],
        }


@dataclass
class SceneLabelingResult:
    scene_id: str
    control_type: SceneLabel
    road_type: SceneLabel
    interaction_type: SceneLabel
    # 全局 trace（按 rule 顺序聚合，便于整体调试）
    global_rule_trace: List[RuleTraceItem]
    # 为了追溯，保留 features
    features: SceneFeatures

    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "control_type": self.control_type.to_jsonable(),
            "road_type": self.road_type.to_jsonable(),
            "interaction_type": self.interaction_type.to_jsonable(),
            "global_rule_trace": [
                {
                    "step": it.step,
                    "rule": it.rule,
                    "condition": it.condition,
                    "result": it.result,
                    "details": it.details,
                    "confidence": float(it.confidence),
                }
                for it in self.global_rule_trace
            ],
            "features": dict(self.features),
        }


def dataclass_asdict(obj: Any) -> Any:
    """
    Avoid importing dataclasses.asdict at call sites.
    """

    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    return obj

