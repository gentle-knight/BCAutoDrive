from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import random
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from bc_baseline.scene_labeling.map_parser import extract_map_features
from bc_baseline.scene_labeling.scene_rules import LabelingConfig, label_scene
from bc_baseline.scene_labeling.schemas import RuleTraceItem
from bc_baseline.scene_labeling.track_parser import extract_track_features


FEATURE_COLUMNS: List[str] = [
    "scene_id",
    "num_lanes",
    "num_lane_boundaries",
    "num_road_boundaries",
    "num_crosswalks",
    "num_stop_signs",
    "num_speed_bumps",
    "has_dynamic_lane_state",
    "num_objects",
    "num_moving_vehicles",
    "mean_vehicle_speed",
    "p90_vehicle_speed",
    "heading_dispersion",
    "num_candidate_interactions",
    "num_lane_changes",
    "num_merge_candidates",
    "num_crossing_candidates",
    "num_turning_tracks",
    "mean_min_ttc",
    "mean_min_thw",
    "parallel_lane_ratio",
]


def _trace_items_to_jsonable(items: Sequence[RuleTraceItem]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for it in items:
        out.append(
            {
                "step": it.step,
                "rule": it.rule,
                "condition": it.condition,
                "result": it.result,
                "details": it.details,
                "confidence": float(it.confidence),
            }
        )
    return out


def _load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        # ScenarioNet pkls usually are plain pickle
        return pickle.load(f)


def _extract_scenario_dict(obj: Any) -> Dict[str, Any]:
    # Most likely: dict already
    if isinstance(obj, dict):
        if {"id", "tracks", "dynamic_map_states", "map_features"}.issubset(set(obj.keys())):
            return obj
        # Some wrappers
        if "scenario" in obj and isinstance(obj["scenario"], dict):
            return _extract_scenario_dict(obj["scenario"])
        if "data" in obj and isinstance(obj["data"], dict):
            return _extract_scenario_dict(obj["data"])
    raise ValueError("Unsupported pkl format: cannot find required scenario dict.")


def _discover_pkl_files(input_path: str) -> List[str]:
    input_path = os.path.abspath(input_path)
    if os.path.isfile(input_path):
        return [input_path]
    if os.path.isdir(input_path):
        # 1) If scenario pkls already exist directly/recursively, prefer them.
        #    (Some dataset folders contain only dataset_mapping/summary without real scenario pkls.)
        direct_pkl: List[str] = []
        for root, _dirs, files in os.walk(input_path):
            for fn in files:
                if not fn.endswith(".pkl"):
                    continue
                if fn in ("dataset_mapping.pkl", "dataset_summary.pkl"):
                    continue
                direct_pkl.append(os.path.join(root, fn))
        direct_pkl.sort()
        if direct_pkl:
            return direct_pkl

        # 2) Legacy index mode: dataset_mapping.pkl maps scenario pkl filename -> scenario folder.
        mapping_path = os.path.join(input_path, "dataset_mapping.pkl")
        if os.path.isfile(mapping_path):
            try:
                mapping_obj = _load_pickle(mapping_path)
                if isinstance(mapping_obj, dict) and mapping_obj:
                    out: List[str] = []
                    for scenario_fn, scenario_dir in mapping_obj.items():
                        if not isinstance(scenario_fn, str):
                            continue
                        if not isinstance(scenario_fn, str) or not scenario_fn.endswith(".pkl"):
                            continue

                        scenario_root_abs: Optional[str] = None
                        if isinstance(scenario_dir, str):
                            scenario_root_abs = os.path.abspath(os.path.join(input_path, scenario_dir))
                        elif isinstance(scenario_dir, dict):
                            # best-effort for alternative formats
                            # try a few common keys
                            for k in ("path", "scenario_dir", "dir", "root"):
                                v = scenario_dir.get(k, None)
                                if isinstance(v, str):
                                    scenario_root_abs = os.path.abspath(os.path.join(input_path, v))
                                    break

                        if scenario_root_abs is None:
                            continue

                        candidate = os.path.join(scenario_root_abs, scenario_fn)
                        if os.path.isfile(candidate):
                            out.append(candidate)

                    out.sort()
                    if out:
                        return out
            except Exception:
                # fall back to generic discovery
                pass

        # 3) Generic fallback: load all pkl except mapping/summary (or empty).
        out: List[str] = []
        for root, _dirs, files in os.walk(input_path):
            for fn in files:
                if fn.endswith(".pkl") and fn not in ("dataset_mapping.pkl", "dataset_summary.pkl"):
                    out.append(os.path.join(root, fn))
        out.sort()
        return out
    raise FileNotFoundError(f"input not found: {input_path}")


def _scene_id_from_scenario(scenario: Mapping[str, Any], fallback_path: str) -> str:
    sid = scenario.get("id", None)
    if sid is not None:
        return str(sid)
    meta = scenario.get("metadata", {}) or {}
    # metadrive converter uses metadata["scenario_id"]
    scen_id = meta.get("scenario_id", None)
    if scen_id is not None:
        return str(scen_id)
    return os.path.splitext(os.path.basename(fallback_path))[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rule-based scene labeling for ScenarioNet->Waymo Motion pkls (v1).")
    parser.add_argument("--input", type=str, required=True, help="单个 scenario_xxx.pkl 或整个场景目录")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录（默认 bc_baseline/scene_labeling/outputs）",
    )
    parser.add_argument("--samples_per_class", type=int, default=20, help="每类随机抽样数量（用于人工复核）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = os.path.abspath(args.input)
    pkl_files = _discover_pkl_files(input_path)
    if not pkl_files:
        raise RuntimeError(f"No pkl files found under input: {input_path}")

    if args.output_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(project_root, "scene_labeling", "outputs")
    else:
        output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Output paths
    scene_labels_path = os.path.join(output_dir, "scene_labels.json")
    scene_features_path = os.path.join(output_dir, "scene_features.csv")
    review_samples_path = os.path.join(output_dir, "review_samples.json")

    config = LabelingConfig()

    scene_labels: Dict[str, Any] = {}
    features_rows: List[Dict[str, Any]] = []

    # For review sampling grouping
    by_control: Dict[str, List[str]] = {"none": [], "stop_controlled": [], "signalized": [], "unknown": []}
    by_road: Dict[str, List[str]] = {"freeway": [], "urban_road": [], "merge_ramp": [], "intersection": []}
    by_interaction: Dict[str, List[str]] = {
        "following": [],
        "lane_change": [],
        "merge": [],
        "crossing_conflict": [],
        "turning_conflict": [],
        "mixed": [],
    }

    errors: List[Dict[str, str]] = []

    for pkl_path in pkl_files:
        try:
            obj = _load_pickle(pkl_path)
            scenario = _extract_scenario_dict(obj)
            scene_id = _scene_id_from_scenario(scenario, fallback_path=pkl_path)

            map_feats = extract_map_features(scenario)
            track_feats = extract_track_features(scenario, config=config)

            scene_features: Dict[str, Any] = {}
            scene_features.update(map_feats.to_scene_features_updates())
            scene_features.update(track_feats.to_scene_features_updates())

            labeled = label_scene(scene_features=scene_features, config=config)

            control_obj = labeled["control_type"]
            road_obj = labeled["road_type"]
            interaction_obj = labeled["interaction_type"]
            global_trace = labeled["rule_trace"]

            # Store feature row (CSV)
            row = {k: scene_features.get(k, 0) for k in FEATURE_COLUMNS}
            row["scene_id"] = scene_id
            features_rows.append(row)

            # Store json label output
            scene_labels[scene_id] = {
                "control_type": control_obj.to_jsonable(),
                "road_type": road_obj.to_jsonable(),
                "interaction_type": interaction_obj.to_jsonable(),
                "rule_trace": _trace_items_to_jsonable(global_trace),
                "features": scene_features,
            }

            by_control[str(scene_labels[scene_id]["control_type"]["label"])].append(scene_id)
            by_road[str(scene_labels[scene_id]["road_type"]["label"])].append(scene_id)
            by_interaction[str(scene_labels[scene_id]["interaction_type"]["label"])].append(scene_id)

        except Exception as e:
            errors.append({"path": pkl_path, "error": str(e)})
            continue

    # Write CSV
    with open(scene_features_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FEATURE_COLUMNS)
        writer.writeheader()
        for row in features_rows:
            writer.writerow(row)

    # Write labels json
    payload = {
        "version": "rule_labeler_v1",
        "num_scenes": len(scene_labels),
        "scene_labels": scene_labels,
        "errors": errors,
        "config": asdict(config),
    }
    with open(scene_labels_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # Review sampling
    rng = random.Random(int(args.seed))
    def sample_ids(group: List[str], k: int) -> List[str]:
        if not group:
            return []
        if len(group) <= k:
            return sorted(group)
        return sorted(rng.sample(group, k))

    review_payload = {
        "version": "rule_labeler_v1",
        "seed": int(args.seed),
        "samples_per_class": int(args.samples_per_class),
        "by_control_type": {k: sample_ids(v, args.samples_per_class) for k, v in by_control.items()},
        "by_road_type": {k: sample_ids(v, args.samples_per_class) for k, v in by_road.items()},
        "by_interaction_type": {k: sample_ids(v, args.samples_per_class) for k, v in by_interaction.items()},
    }
    with open(review_samples_path, "w", encoding="utf-8") as f:
        json.dump(review_payload, f, ensure_ascii=False, indent=2)

    print(f"[scene_labeling] Done. Scenes labeled: {len(scene_labels)}")
    print(f"[scene_labeling] scene_labels.json -> {scene_labels_path}")
    print(f"[scene_labeling] scene_features.csv -> {scene_features_path}")
    print(f"[scene_labeling] review_samples.json -> {review_samples_path}")
    if errors:
        print(f"[scene_labeling] WARNING: {len(errors)} files failed, see errors in scene_labels.json")


if __name__ == "__main__":
    main()

