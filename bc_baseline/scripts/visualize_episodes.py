import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from bc_baseline.scripts.interaction_feature_schema import (
    feature_names,
    parse_feature_schema_arg,
)


def load_features_with_schema_check(features_path: str, expected_names: list) -> np.ndarray:
    data = np.load(features_path)
    if "features" not in data:
        raise KeyError(f"npz 中缺少键 'features'：{features_path}")
    if "feature_names" not in data:
        data.close()
        raise KeyError(
            "npz 中缺少键 'feature_names'，无法确认列语义。"
            "请使用当前版本的 extract_interaction_episodes.py 重新导出 features。"
        )

    names = [str(x) for x in np.asarray(data["feature_names"]).tolist()]
    features = np.asarray(data["features"])
    data.close()

    if names != expected_names:
        raise ValueError(
            f"feature_names 不匹配，读取到 {names}，预期 {expected_names}。"
            "请使用 --feature_schema 与 npz 一致，或重新提取。"
        )
    return features


def main() -> None:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_ie_dir = os.path.join(project_root, "outputs", "interaction_episodes")
    parser = argparse.ArgumentParser(description="绘制 episode 特征分布直方图")
    parser.add_argument(
        "--features_path",
        default=os.path.join(default_ie_dir, "episode_features.npz"),
        help="episode_features.npz 路径",
    )
    parser.add_argument(
        "--feature_schema",
        type=str,
        default="v2",
        choices=["v1", "v2"],
        help="须与提取时一致（默认 v2）",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help="输出 PNG 路径（默认与 npz 同目录下 feature_distributions.png）",
    )
    args = parser.parse_args()
    schema = parse_feature_schema_arg(args.feature_schema)
    expected = feature_names(schema)

    features_path = os.path.abspath(args.features_path)
    features = load_features_with_schema_check(features_path, expected)
    names = expected

    out = args.output_path
    if out is None:
        out = os.path.join(os.path.dirname(features_path), "feature_distributions.png")

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, (ax, name) in enumerate(zip(axes.flat, names)):
        col = features[:, i]
        ax.hist(col, bins=50, color="steelblue", edgecolor="none", alpha=0.8)
        ax.set_title(f"{name}\n(schema={schema})")
        ax.set_xlabel("value")
        ax.set_ylabel("count")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[visualize_episodes] 已保存: {out}")


if __name__ == "__main__":
    main()
