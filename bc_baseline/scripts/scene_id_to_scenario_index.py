from __future__ import annotations

import argparse
import os
import pickle
import re
from typing import Any, Dict, Optional, Tuple


DEFAULT_PATTERN = r"^sd_waymo_v1\.2_(.*)\.pkl$"


def _load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def _find_index_from_summary(
    dataset_dir: str,
    scene_id: str,
    pattern: str,
) -> Tuple[int, str]:
    """
    返回:
      - scenario_index: scenarionet.sim 里要传的整数索引
      - matching_key: dataset_summary.pkl 中匹配到的键名

    注意：scenario_index 的定义依赖 dataset_summary.pkl 里 keys 的遍历顺序。
    """

    summary_path = os.path.join(dataset_dir, "dataset_summary.pkl")
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(f"dataset_summary.pkl not found: {summary_path}")

    summary = _load_pickle(summary_path)
    if not isinstance(summary, dict):
        raise TypeError(f"dataset_summary.pkl expected dict, got: {type(summary)}")

    keys = list(summary.keys())
    rx = re.compile(pattern)
    for i, k in enumerate(keys):
        m = rx.match(str(k))
        if m and m.group(1) == scene_id:
            return i, str(k)
    raise KeyError(f"scene_id '{scene_id}' not found in dataset_summary.pkl under: {dataset_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ScenarioNet scene_id(hash) to scenarionet.sim --scenario_index (int)."
    )
    parser.add_argument(
        "--dataset_dir",
        "-d",
        type=str,
        default="legacy_magail/data/exp_filtered/",
        help="dataset 根目录（包含 dataset_summary.pkl）。",
    )
    # 与用户习惯对齐：输入参数叫 --id
    parser.add_argument(
        "--scene_id",
        "--id",
        type=str,
        required=True,
        help="scene_labels.json 里的 scene_id（例如 a7545087f82dafeb）。",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=DEFAULT_PATTERN,
        help=f"用于匹配 dataset_summary.pkl 的 key 的正则（默认: {DEFAULT_PATTERN}）。",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="输出 JSON（方便下游脚本读取），否则只输出整数 scenario_index。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = os.path.abspath(args.dataset_dir)
    scene_id = str(args.scene_id)

    idx, key = _find_index_from_summary(
        dataset_dir=dataset_dir,
        scene_id=scene_id,
        pattern=args.pattern,
    )

    if args.json:
        print({"dataset_dir": dataset_dir, "scene_id": scene_id, "scenario_index": idx, "key": key})
    else:
        # 与 scenarionet.sim 的类型对齐：int
        print(int(idx))


if __name__ == "__main__":
    main()

