import os
import numpy as np
import matplotlib.pyplot as plt
import json

from bc_baseline.scripts.extract_interaction_episodes import FEATURE_NAMES


def load_features_with_schema_check(features_path):
    data = np.load(features_path)
    if "features" not in data:
        raise KeyError(f"npz 中缺少键 'features'：{features_path}")
    if "feature_names" not in data:
        data.close()
        raise KeyError(
            "npz 中缺少键 'feature_names'，无法确认列语义。"
            "请使用当前版本的 extract_interaction_episodes.py 重新导出 features。"
        )

    feature_names = [str(x) for x in np.asarray(data["feature_names"]).tolist()]
    features = np.asarray(data["features"])
    data.close()

    if feature_names != FEATURE_NAMES:
        raise ValueError(
            f"feature_names 不匹配，读取到 {feature_names}，预期 {FEATURE_NAMES}"
        )
    return features


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
interaction_dir = os.path.join(project_root, "outputs", "interaction_episodes")

features_path = os.path.join(interaction_dir, "episode_features.npz")
features = load_features_with_schema_check(features_path)
# shape (N, 8)，列顺序见 FEATURE_NAMES

names = FEATURE_NAMES

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i, (ax, name) in enumerate(zip(axes.flat, names)):
    data = features[:, i]
    ax.hist(data, bins=50, color="steelblue", edgecolor="none", alpha=0.8)
    ax.set_title(name)
    ax.set_xlabel("value")
    ax.set_ylabel("count")
plt.tight_layout()
output_path = os.path.join(interaction_dir, "feature_distributions.png")
plt.savefig(output_path, dpi=150)
