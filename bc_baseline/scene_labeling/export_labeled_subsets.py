from __future__ import annotations

import argparse
import json
import os
import pickle
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple


def _load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def _ensure_dir_empty(path: str, *, exist_ok: bool) -> None:
    if os.path.exists(path):
        if not exist_ok:
            # If dir exists but is empty, allow.
            if os.listdir(path):
                raise FileExistsError(f"Output dir exists and is not empty: {path}")
    os.makedirs(path, exist_ok=True)


def _scenario_file_name_from_scene_id(
    scene_id_to_scenario_file: Mapping[str, str],
    scene_id: str,
) -> Optional[str]:
    return scene_id_to_scenario_file.get(scene_id, None)


@dataclass
class ExportPlan:
    # category: "control_type" | "road_type" | "interaction_type"
    category: str
    label_value: str
    scenario_files: List[str]


def build_scene_id_to_scenario_file_map(
    dataset_summary: Mapping[str, Any],
    scene_id_regex: str,
) -> Dict[str, str]:
    """
    dataset_summary.pkl 的 keys 通常是 scenario pkl 文件名，
    形如: sd_waymo_v1.2_<scene_id>.pkl
    """
    rx = re.compile(scene_id_regex)
    out: Dict[str, str] = {}
    for scenario_file in dataset_summary.keys():
        m = rx.match(str(scenario_file))
        if not m:
            continue
        scene_id = m.group(1)
        out[scene_id] = str(scenario_file)
    return out


def build_export_plan(
    scene_labels: Mapping[str, Any],
    scene_id_to_scenario_file: Mapping[str, str],
    *,
    categories: Sequence[str],
    subset_limit_per_label: Optional[int] = None,
    allowed_scene_ids_by_category_label: Optional[Dict[str, Dict[str, set[str]]]] = None,
) -> Tuple[List[ExportPlan], Dict[str, int]]:
    """
    返回 ExportPlan 列表，以及丢失统计（scene_id -> scenario_file 找不到）。
    """
    missing_scene_id = 0
    # category -> label_value -> scenario_files
    grouped: Dict[str, Dict[str, List[str]]] = {c: {} for c in categories}

    for scene_id, obj in scene_labels.items():
        for c in categories:
            label_obj = obj.get(c, None)
            if not isinstance(label_obj, dict):
                continue
            label_value = label_obj.get("label", None)
            if label_value is None:
                continue
            label_value = str(label_value)

            if allowed_scene_ids_by_category_label is not None:
                allowed_for_cat = allowed_scene_ids_by_category_label.get(c, {})
                allowed_set = allowed_for_cat.get(label_value, None)
                # 若该 (category,label) 不在 allowed 中，则跳过整个组
                if allowed_set is None:
                    continue
                if scene_id not in allowed_set:
                    continue

            scenario_file = _scenario_file_name_from_scene_id(scene_id_to_scenario_file, scene_id)
            if scenario_file is None:
                # 不同 category 会重复统计，这里简化：只要一个 category 缺就计一次
                missing_scene_id += 1
                continue
            grouped[c].setdefault(label_value, []).append(scenario_file)

    plans: List[ExportPlan] = []
    for c in categories:
        for label_value, scenario_files in grouped[c].items():
            if subset_limit_per_label is not None:
                scenario_files = scenario_files[: int(subset_limit_per_label)]
            if len(scenario_files) == 0:
                continue
            plans.append(ExportPlan(category=c, label_value=label_value, scenario_files=scenario_files))

    # missing_scene_id 指的是“至少一次 category 缺失”的次数（粗略）
    return plans, {"missing_scene_id": missing_scene_id}


def _compute_new_relative_mapping_dir(
    *,
    original_dataset_dir: str,
    original_mapping_value: str,
    scenario_file_name: str,
    new_dataset_root: str,
) -> str:
    # original_mapping_value: scenario_file_name 所在“目录”相对路径（相对于 original_dataset_dir）
    src_abs_pkl = os.path.join(original_dataset_dir, original_mapping_value, scenario_file_name)
    src_abs_dir = os.path.dirname(src_abs_pkl)
    rel_dir = os.path.relpath(src_abs_dir, new_dataset_root)
    # scenarionet/utils.py 用 os.path.join(file_folder, mapping[file], file)
    # 这里给 '.' 也可以被 join 正常处理。
    if rel_dir == "":
        rel_dir = "."
    return rel_dir


def export_dataset_root(
    *,
    plan: ExportPlan,
    output_root: str,
    original_dataset_dir: str,
    original_dataset_summary: Mapping[str, Any],
    original_dataset_mapping: Mapping[str, str],
    dry_run: bool,
    exist_ok: bool,
) -> None:
    # New dataset root directory
    category_folder = {
        "control_type": "by_control_type",
        "road_type": "by_road_type",
        "interaction_type": "by_interaction_type",
    }.get(plan.category, f"by_{plan.category}")

    new_dataset_root = os.path.join(output_root, category_folder, plan.label_value)
    _ensure_dir_empty(new_dataset_root, exist_ok=exist_ok)

    # Build subset summary + mapping
    subset_summary: Dict[str, Any] = {}
    subset_mapping: Dict[str, str] = {}

    for scenario_file in plan.scenario_files:
        if scenario_file not in original_dataset_summary:
            continue
        subset_summary[scenario_file] = original_dataset_summary[scenario_file]

        original_rel_dir = original_dataset_mapping.get(scenario_file, "")
        if not isinstance(original_rel_dir, str):
            original_rel_dir = str(original_rel_dir)
        rel_dir_in_new = _compute_new_relative_mapping_dir(
            original_dataset_dir=original_dataset_dir,
            original_mapping_value=original_rel_dir,
            scenario_file_name=scenario_file,
            new_dataset_root=new_dataset_root,
        )
        subset_mapping[scenario_file] = rel_dir_in_new

    summary_file_path = os.path.join(new_dataset_root, "dataset_summary.pkl")
    mapping_file_path = os.path.join(new_dataset_root, "dataset_mapping.pkl")

    if dry_run:
        print(
            f"[dry_run] {plan.category}/{plan.label_value}: "
            f"scenarios={len(subset_summary)} -> {new_dataset_root}"
        )
        return

    # Prefer scenarionet helper for consistent serialization
    try:
        from scenarionet.common_utils import save_summary_and_mapping
    except Exception:
        save_summary_and_mapping = None

    if save_summary_and_mapping is None:
        # Fallback: write pickle directly (may contain numpy arrays).
        with open(summary_file_path, "wb") as f:
            pickle.dump(subset_summary, f)
        with open(mapping_file_path, "wb") as f:
            pickle.dump(subset_mapping, f)
    else:
        save_summary_and_mapping(summary_file_path, mapping_file_path, subset_summary, subset_mapping)

    print(
        f"[exported] {plan.category}/{plan.label_value}: "
        f"scenarios={len(subset_summary)} -> {new_dataset_root}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export ScenarioNet dataset subsets grouped by rule-based scene labels."
    )
    parser.add_argument(
        "--scene_labels_json",
        type=str,
        default="bc_baseline/scene_labeling/outputs/scene_labels.json",
        help="输入 scene_labels.json（来自 scene_labeling/label_scenes.py）。",
    )
    parser.add_argument(
        "--dataset_dir",
        "-d",
        type=str,
        default="legacy_magail/data/exp_filtered/",
        help="原始 ScenarioNet 数据集根目录（包含 dataset_summary.pkl/dataset_mapping.pkl）。",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="bc_baseline/scene_labeling/outputs/labeled_subsets/",
        help="输出根目录。",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="仅打印将导出哪些标签子集，不生成 dataset_summary.pkl/dataset_mapping.pkl。",
    )
    parser.add_argument(
        "--exist_ok",
        action="store_true",
        help="允许输出目录已存在（会覆盖写 dataset_summary/dataset_mapping）。",
    )
    parser.add_argument(
        "--scene_id_regex",
        type=str,
        default=r"^sd_waymo_v1\.2_(.*)\.pkl$",
        help="用于从 dataset_summary.pkl 的 scenario 文件名中抽取 scene_id 的正则。",
    )
    parser.add_argument(
        "--subset_limit_per_label",
        type=int,
        default=None,
        help="每个 label 最多导出多少个 scenario_file_name（调试用）。",
    )
    parser.add_argument(
        "--use_review_samples",
        action="store_true",
        help="若启用，则只导出 review_samples.json 里包含的 scene_id（复核抽样子集）。",
    )
    parser.add_argument(
        "--review_samples_json",
        type=str,
        default="bc_baseline/scene_labeling/outputs/review_samples.json",
        help="review_samples.json 路径（仅在 --use_review_samples 启用时生效）。",
    )
    parser.add_argument(
        "--report_path",
        type=str,
        default=None,
        help="报告文件路径（JSON）。默认写入 output_root/export_report.json。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    scene_labels_json = os.path.abspath(args.scene_labels_json)
    original_dataset_dir = os.path.abspath(args.dataset_dir)
    output_root = os.path.abspath(args.output_root)

    scene_labels_payload = json.load(open(scene_labels_json, "r", encoding="utf-8"))
    scene_labels = scene_labels_payload.get("scene_labels", {})
    if not isinstance(scene_labels, dict) or not scene_labels:
        raise ValueError(f"scene_labels_json has no scene_labels: {scene_labels_json}")

    summary_path = os.path.join(original_dataset_dir, "dataset_summary.pkl")
    mapping_path = os.path.join(original_dataset_dir, "dataset_mapping.pkl")
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(f"dataset_summary.pkl not found: {summary_path}")
    if not os.path.isfile(mapping_path):
        raise FileNotFoundError(f"dataset_mapping.pkl not found: {mapping_path}")

    original_dataset_summary = _load_pickle(summary_path)
    original_dataset_mapping = _load_pickle(mapping_path)
    if not isinstance(original_dataset_summary, dict) or not isinstance(original_dataset_mapping, dict):
        raise TypeError(
            f"dataset_summary.pkl/mapping.pkl should be dict, got: "
            f"{type(original_dataset_summary)} / {type(original_dataset_mapping)}"
        )

    scene_id_to_scenario_file = build_scene_id_to_scenario_file_map(
        dataset_summary=original_dataset_summary,
        scene_id_regex=args.scene_id_regex,
    )
    if not scene_id_to_scenario_file:
        raise RuntimeError(
            "Failed to build scene_id_to_scenario_file mapping. "
            "Try adjusting --scene_id_regex."
        )

    categories = ["control_type", "road_type", "interaction_type"]

    allowed_scene_ids_by_category_label: Optional[Dict[str, Dict[str, set[str]]]] = None
    export_mode = "all"
    review_samples_payload: Optional[Dict[str, Any]] = None

    if args.use_review_samples:
        export_mode = "review"
        review_samples_json = os.path.abspath(args.review_samples_json)
        if not os.path.isfile(review_samples_json):
            raise FileNotFoundError(f"review_samples.json not found: {review_samples_json}")
        review_samples_payload = json.load(open(review_samples_json, "r", encoding="utf-8"))

        allowed_scene_ids_by_category_label = {"control_type": {}, "road_type": {}, "interaction_type": {}}
        by_control = review_samples_payload.get("by_control_type", {}) or {}
        by_road = review_samples_payload.get("by_road_type", {}) or {}
        by_inter = review_samples_payload.get("by_interaction_type", {}) or {}

        if isinstance(by_control, dict):
            for label_value, ids in by_control.items():
                if isinstance(ids, list):
                    allowed_scene_ids_by_category_label["control_type"][str(label_value)] = set([str(x) for x in ids])
        if isinstance(by_road, dict):
            for label_value, ids in by_road.items():
                if isinstance(ids, list):
                    allowed_scene_ids_by_category_label["road_type"][str(label_value)] = set([str(x) for x in ids])
        if isinstance(by_inter, dict):
            for label_value, ids in by_inter.items():
                if isinstance(ids, list):
                    allowed_scene_ids_by_category_label["interaction_type"][str(label_value)] = set([str(x) for x in ids])

    plans, missing_info = build_export_plan(
        scene_labels=scene_labels,
        scene_id_to_scenario_file=scene_id_to_scenario_file,
        categories=categories,
        subset_limit_per_label=args.subset_limit_per_label,
        allowed_scene_ids_by_category_label=allowed_scene_ids_by_category_label,
    )

    print(
        f"[export_labeled_subsets] total plans: {len(plans)}; "
        f"missing_scene_id={missing_info.get('missing_scene_id', 0)}"
    )

    os.makedirs(output_root, exist_ok=True)

    # ----------------- report ----------------- #
    counts_by_category_and_label: Dict[str, Dict[str, int]] = {c: {} for c in categories}
    for p in plans:
        counts_by_category_and_label[p.category][p.label_value] = len(p.scenario_files)

    report: Dict[str, Any] = {
        "version": "export_labeled_subsets_v2",
        "export_mode": export_mode,
        "scene_labels_json": scene_labels_json,
        "dataset_dir": original_dataset_dir,
        "output_root": output_root,
        "categories": categories,
        "subset_limit_per_label": args.subset_limit_per_label,
        "counts_by_category_and_label": counts_by_category_and_label,
        "total_scenarios_in_plans": int(sum(len(p.scenario_files) for p in plans)),
        "missing_info": missing_info,
    }
    if args.use_review_samples:
        report["review_samples_json"] = os.path.abspath(args.review_samples_json)

    report_path = args.report_path
    if report_path is None:
        report_path = os.path.join(output_root, "export_report.json")
    report_path = os.path.abspath(report_path)

    if not args.dry_run:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[export_labeled_subsets] report -> {report_path}")

    # Execute
    for plan in plans:
        export_dataset_root(
            plan=plan,
            output_root=output_root,
            original_dataset_dir=original_dataset_dir,
            original_dataset_summary=original_dataset_summary,
            original_dataset_mapping=original_dataset_mapping,
            dry_run=args.dry_run,
            exist_ok=args.exist_ok,
        )

    print("[export_labeled_subsets] Done.")


if __name__ == "__main__":
    main()

