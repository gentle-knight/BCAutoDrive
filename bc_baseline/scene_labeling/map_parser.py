from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

from bc_baseline.scene_labeling.schemas import SceneFeatures


@dataclass
class MapFeatures:
    num_lanes: int
    num_lane_boundaries: int
    num_road_boundaries: int
    num_crosswalks: int
    num_stop_signs: int
    num_speed_bumps: int
    has_dynamic_lane_state: bool
    # for debugging
    num_dynamic_objects: int = 0

    def to_scene_features_updates(self) -> Dict[str, Any]:
        return {
            "num_lanes": int(self.num_lanes),
            "num_lane_boundaries": int(self.num_lane_boundaries),
            "num_road_boundaries": int(self.num_road_boundaries),
            "num_crosswalks": int(self.num_crosswalks),
            "num_stop_signs": int(self.num_stop_signs),
            "num_speed_bumps": int(self.num_speed_bumps),
            "has_dynamic_lane_state": bool(self.has_dynamic_lane_state),
        }


def _iter_dict_values(maybe_dict: Any) -> Iterable[Any]:
    if isinstance(maybe_dict, dict):
        return maybe_dict.values()
    if isinstance(maybe_dict, list):
        return maybe_dict
    return []


def extract_map_features(scenario: Mapping[str, Any]) -> MapFeatures:
    """
    解析地图特征与动态控制信息，并输出 Step2 里要求的 map features。

    说明：
    - ScenarioNet/MetaDrive 转换后，Waymo 的 lane 类特征的 `type` 通常包含 `LANE_` 前缀。
    - lane 的左右边界在 `left_boundaries` / `right_boundaries` 字段中，数量可直接累加。
    - stop/crosswalk/speed_bump 在 map_features 的条目里会以 type 字符串区分。
    """

    map_features_obj = scenario.get("map_features", {}) or {}
    dynamic_map_states_obj = scenario.get("dynamic_map_states", {}) or {}

    num_lanes = 0
    num_lane_boundaries = 0
    num_road_boundaries = 0
    num_crosswalks = 0
    num_stop_signs = 0
    num_speed_bumps = 0

    # ----- parse map_features -----
    for feat in _iter_dict_values(map_features_obj):
        if not isinstance(feat, dict):
            continue
        feat_type = feat.get("type", None)
        if not isinstance(feat_type, str):
            continue

        # LANE features
        if "LANE_" in feat_type:
            # 大多数 lane center 会带 left/right_boundaries 字段
            # 同时可能也存在 entry_lanes 字段，用于进一步区分。
            num_lanes += 1
            left_b = feat.get("left_boundaries", None) or []
            right_b = feat.get("right_boundaries", None) or []
            if isinstance(left_b, list):
                num_lane_boundaries += len(left_b)
            if isinstance(right_b, list):
                num_lane_boundaries += len(right_b)
        # Road edge boundaries (physical road boundaries)
        elif feat_type.startswith("ROAD_EDGE_"):
            num_road_boundaries += 1

        # Crosswalk / Stop sign / Speed bump
        if feat_type == "CROSSWALK":
            num_crosswalks += 1
        elif feat_type == "STOP_SIGN":
            num_stop_signs += 1
        elif feat_type == "SPEED_BUMP":
            num_speed_bumps += 1

    # ----- parse dynamic_map_states -----
    # dynamic_map_states 转换后通常表示交通灯（TrafficLight），其 state.object_state 序列可能包含非 None 状态
    has_dynamic_lane_state = False
    num_dynamic_objects = 0

    for dyn_obj in _iter_dict_values(dynamic_map_states_obj):
        if not isinstance(dyn_obj, dict):
            continue
        num_dynamic_objects += 1
        state = dyn_obj.get("state", None)
        if not isinstance(state, dict):
            continue
        object_state_seq = state.get("object_state", None)
        lane = dyn_obj.get("lane", None)
        if lane is None:
            continue
        if isinstance(object_state_seq, (list, tuple)):
            # 若存在非 None 的 object_state，说明该动态控制在场景片段内有效
            if any(v is not None for v in object_state_seq):
                has_dynamic_lane_state = True
                break

    return MapFeatures(
        num_lanes=num_lanes,
        num_lane_boundaries=num_lane_boundaries,
        num_road_boundaries=num_road_boundaries,
        num_crosswalks=num_crosswalks,
        num_stop_signs=num_stop_signs,
        num_speed_bumps=num_speed_bumps,
        has_dynamic_lane_state=has_dynamic_lane_state,
        num_dynamic_objects=num_dynamic_objects,
    )

