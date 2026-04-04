import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from bc_baseline.Env.expert_env import BCExpertEnv
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


def visualize_episode(waymo_dir, episode_meta, episode_features, save_path=None):
    """
    可视化单个 interaction episode 的轨迹。
    episode_meta: dict，来自 episode_meta.json 的一条记录
    """
    scenario_idx = episode_meta["scenario_index"]
    ego_id = episode_meta["ego_track_id"]
    partner_id = episode_meta["partner_track_id"]
    t_start = episode_meta["t_start"]
    t_end = episode_meta["t_end"]
    t_peak = episode_meta["t_peak"]
    min_ttc = episode_meta["min_ttc"]

    # 加载场景
    config = BCExpertEnv.default_config()
    config.update({"data_directory": waymo_dir,
                   "start_scenario_index": scenario_idx,
                   "num_scenarios": 1})
    env = BCExpertEnv(config)
    env.reset(seed=scenario_idx)

    ego_meta = env._all_tracks.get(ego_id)
    partner_meta = env._all_tracks.get(partner_id)
    if ego_meta is None or partner_meta is None:
        print("track not found"); env.close(); return

    sl = slice(t_start, t_end + 1)
    ego_pos = np.array(ego_meta.track["state"]["position"][sl, :2])
    partner_pos = np.array(partner_meta.track["state"]["position"][sl, :2])
    ego_valid = np.array(ego_meta.valid_mask[sl])
    partner_valid = np.array(partner_meta.valid_mask[sl])

    # 其他背景车（灰色）
    fig, ax = plt.subplots(figsize=(10, 10))
    for tid, tmeta in env._all_tracks.items():
        if tid in (ego_id, partner_id): continue
        pos = np.array(tmeta.track["state"]["position"][sl, :2])
        valid = np.array(tmeta.valid_mask[sl])
        if valid.any():
            ax.plot(pos[valid, 0], pos[valid, 1],
                    color="lightgray", linewidth=0.8, alpha=0.5)

    # partner（橙色）
    ax.plot(partner_pos[partner_valid, 0], partner_pos[partner_valid, 1],
            color="orange", linewidth=2, label=f"partner ({partner_id})")
    ax.scatter(partner_pos[partner_valid, 0][0],
               partner_pos[partner_valid, 1][0],
               color="orange", s=80, zorder=5)

    # ego（蓝色）
    ax.plot(ego_pos[ego_valid, 0], ego_pos[ego_valid, 1],
            color="royalblue", linewidth=2.5, label=f"ego ({ego_id})")
    ax.scatter(ego_pos[ego_valid, 0][0], ego_pos[ego_valid, 1][0],
               color="royalblue", s=80, zorder=5)

    # 标出 t_peak 的位置
    peak_offset = t_peak - t_start
    if 0 <= peak_offset < len(ego_pos) and ego_valid[peak_offset]:
        ax.scatter(ego_pos[peak_offset, 0], ego_pos[peak_offset, 1],
                   color="red", s=150, zorder=6, marker="*",
                   label=f"t_peak (TTC={min_ttc:.2f}s)")

    ax.set_aspect("equal")
    ax.legend(fontsize=9)
    feat_str = "\n".join(
        f"{FEATURE_NAMES[i]}={episode_features[i]:.3f}"
        for i in range(len(FEATURE_NAMES))
    )
    ax.set_title(f"Scenario {scenario_idx} | Episode\n{feat_str}", fontsize=9)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
        plt.close()
    else:
        plt.show()
    env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--waymo_dir", required=True)

    # 默认从 bc_baseline/outputs/interaction_episodes 读取 episode 元信息与特征
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_ie_dir = os.path.join(project_root, "outputs", "interaction_episodes")

    parser.add_argument(
        "--meta",
        default=os.path.join(default_ie_dir, "episode_meta.json"),
        help="episode 元信息 JSON 文件路径（默认 bc_baseline/outputs/interaction_episodes/episode_meta.json）",
    )
    parser.add_argument(
        "--features",
        default=os.path.join(default_ie_dir, "episode_features.npz"),
        help="episode 特征 NPZ 文件路径（默认 bc_baseline/outputs/interaction_episodes/episode_features.npz）",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=20,
        help="可视化前 n 个 episode",
    )
    parser.add_argument(
        "--sort_by",
        default="min_ttc",
        choices=["min_ttc", "min_pet", "jerk_peak", "random"],
        help="按哪个维度排序（取极端样本）",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(default_ie_dir, "episode_vis"),
        help="可视化结果输出目录（默认 bc_baseline/outputs/interaction_episodes/episode_vis）",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.meta, encoding="utf-8") as f:
        metas = json.load(f)
    feats = load_features_with_schema_check(args.features)

    # 排序选取极端样本（min_ttc/min_pet 来自 meta，jerk_peak 来自特征列 2）
    if args.sort_by == "random":
        import random
        indices = random.sample(range(len(metas)), args.n)
    elif args.sort_by in ("min_ttc", "min_pet"):
        vals = np.array([metas[i][args.sort_by] for i in range(len(metas))])
        if args.sort_by == "min_pet":
            valid = vals < 99.0
            vals_masked = np.where(valid, vals, 999.0)
            indices = np.argsort(vals_masked)[: args.n].tolist()
        else:
            indices = np.argsort(vals)[: args.n].tolist()
    else:
        # jerk_peak 对应特征列 2
        col = 2
        vals = feats[:, col]
        indices = np.argsort(vals)[: args.n].tolist()

    for rank, idx in enumerate(indices):
        save_path = os.path.join(args.output_dir,
                                 f"rank{rank:03d}_ep{idx}.png")
        visualize_episode(args.waymo_dir, metas[idx], feats[idx], save_path)
        print(f"saved {save_path}")
