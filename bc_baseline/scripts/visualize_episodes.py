import os
import numpy as np
import matplotlib.pyplot as plt
import json

from bc_baseline.scripts.extract_interaction_episodes import FEATURE_NAMES


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
interaction_dir = os.path.join(project_root, "outputs", "interaction_episodes")

features_path = os.path.join(interaction_dir, "episode_features.npz")
features = np.load(features_path)["features"]
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
