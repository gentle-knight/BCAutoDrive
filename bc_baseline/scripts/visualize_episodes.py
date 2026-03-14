import os
import numpy as np
import matplotlib.pyplot as plt
import json


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
interaction_dir = os.path.join(project_root, "outputs", "interaction_episodes")

features_path = os.path.join(interaction_dir, "episode_features.npz")
features = np.load(features_path)["features"]
# shape (21601, 8)
# 列顺序: mean_speed, std_speed, max_speed, mean_acc, min_acc, jerk_peak, min_ttc, min_pet

names = ["mean_speed", "std_speed", "max_speed",
         "mean_acc", "min_acc", "jerk_peak", "min_ttc", "min_pet"]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i, (ax, name) in enumerate(zip(axes.flat, names)):
    data = features[:, i]
    if name == "min_pet":
        data = data[data < 99.0]  # 过滤掉默认值
    ax.hist(data, bins=50, color="steelblue", edgecolor="none", alpha=0.8)
    ax.set_title(name)
    ax.set_xlabel("value")
    ax.set_ylabel("count")
plt.tight_layout()
output_path = os.path.join(interaction_dir, "feature_distributions.png")
plt.savefig(output_path, dpi=150)
