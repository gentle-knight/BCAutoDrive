# 交互 Episode 提取、可视化与驾驶风格聚类工作流

本文档按**流程顺序**说明：如何从 Waymo 场景中提取“强交互片段”（Interaction Episodes）、如何对特征做可视化、如何进行驾驶风格聚类，以及聚类后的输出如何用于后续分析。便于写入科研日志或作为实验记录参考。

---

## 一、整体流程概览

```text
Waymo .pkl 场景
    → [1] extract_interaction_episodes.py  提取强交互片段 + 8 维特征
    → episode_features.npz + episode_meta.json
    → [2] visualize_episodes.py            可选：8 维特征分布直方图
    → [3] visualize_episode_trajectory.py  可选：单 episode 轨迹图（按 min_ttc/min_pet/jerk 等排序）
    → [4] cluster_driving_styles.py       elbow 选 K → cluster 正式聚类
    → style_labels.json、episode_labels.json、cluster_centers.json、cluster_report.txt
    → 后续：按风格标签筛选/分析 episode，或用于 IRL 等
```

数据与脚本约定（相对项目根）：

- 输入：Waymo 经 ScenarioNet 转换后的 `.pkl` 场景目录（如 `data/exp_filtered`）。
- 中间/输出：`bc_baseline/outputs/interaction_episodes/`（特征与 meta）、`bc_baseline/outputs/driving_style/`（聚类结果）。

---

## 二、步骤 1：提取 Interaction Episodes

**脚本**：`bc_baseline/scripts/extract_interaction_episodes.py`

**作用**：从多段 Waymo 场景中，自动找出“自车（ego）与某一辆对手车（partner）发生强交互”的时间片段，并为每个片段计算一条 **8 维特征向量** 和元信息。这些片段称为 **Interaction Episode**。

### 2.1 什么叫“强交互”与 Episode

- **强交互**：在某一帧，自车与周围某辆车之间的 **TTC（Time-To-Collision）** 低于给定阈值（默认 5 s），即认为该时刻存在明显的碰撞风险或接近事件。
- **Episode 定义**：以“TTC 最小的那一帧”为 **t_peak**，向前后各取一段时间（默认半窗 3 s），得到时间窗口 [t_start, t_end]。在这段窗口内，自车与**同一辆**对手车（该帧 TTC 最小的那辆）的轨迹构成一个 **Interaction Episode**。
- 若同一对 (ego, partner) 在时间上有重叠的多个候选 episode，只保留 **min_ttc 更小** 的那一个，避免重复计数。

### 2.2 单场景内的提取逻辑（简述）

对每一个场景：

1. 对每条**动态轨迹**视为 ego：
   - 在每一帧 t，在 50 m 范围内找 **TTC 最小的对手车**（排除静止车、对向车等），得到每帧的 best_ttc[t] 和 best_partner_id[t]。
2. 在 best_ttc 序列上找 **TTC < 阈值** 的**连续片段**，每个片段取 TTC 最小的时刻作为 **t_peak**。
3. 以 t_peak 为中心、`window_seconds` 为半窗长，得到 [t_start, t_end]。
4. 在该窗口内调用 `extract_episode_features`，计算 8 维特征及该窗口内的 min_ttc、min_pet。
5. 对同一 (ego, partner) 重叠的 episode 做冲突消解，只保留 min_ttc 更小者。

因此，**一个 episode = 一个 (ego, partner, 时间窗)**，对应一段“有风险峰值”的交互。

### 2.3 8 维特征含义（仅描述 ego 行为，不含 min_ttc/min_pet）

| 索引 | 名称 | 含义 |
|------|------|------|
| 0 | mean_acc | 窗口内纵向加速度均值 |
| 1 | min_acc | 窗口内最大制动（最小加速度） |
| 2 | jerk_peak | 窗口内 jerk 峰值绝对值 |
| 3 | response_mean_acc | t_peak 前后小窗口内平均加速度（风险时刻响应） |
| 4 | response_min_acc | t_peak 前后小窗口内最小加速度 |
| 5 | mean_thw | 窗口内平均跟车时间距（距离/ego 速度） |
| 6 | mean_speed_ratio | ego 与 partner 速度比均值 |
| 7 | relative_speed | ego 平均速度 − partner 平均速度 |

**min_ttc、min_pet** 只写入 **episode_meta.json**，不进入特征矩阵，用于标记交互强度与后续排序/筛选。

### 2.4 输出文件

- **episode_features.npz**  
  - `features`：shape (N, 8)，N 为 episode 总数。  
  - `feature_names`：长度为 8 的列名列表。  
  - `feature_version`：特征 schema 版本。  
  下游脚本会校验 `feature_names`，与当前代码不一致时需重新导出。
- **episode_meta.json**  
  长度为 N 的列表，第 i 条对应第 i 行 features，字段包括：scenario_index, ego_track_id, partner_track_id, t_peak, t_start, t_end, min_ttc, min_pet。

### 2.5 常用命令示例

```bash
python -m bc_baseline.scripts.extract_interaction_episodes \
    --waymo_dir /path/to/exp_filtered \
    --num_scenarios 348 \
    --start_index 0 \
    --waymo_dt 0.1 \
    --ttc_threshold 5.0 \
    --window_seconds 3.0 \
    --min_episode_frames 10 \
    --output_dir bc_baseline/outputs/interaction_episodes
```

---

## 三、步骤 2：特征分布可视化（可选）

**脚本**：`bc_baseline/scripts/visualize_episodes.py`

**作用**：读取 `episode_features.npz`，对 **8 个特征** 分别画直方图，得到一张 2×4 的图，用于检查特征分布、异常值、量纲等。

- 输入：默认 `bc_baseline/outputs/interaction_episodes/episode_features.npz`（需含 `feature_names`，否则会报错）。
- 输出：`bc_baseline/outputs/interaction_episodes/feature_distributions.png`。

**不依赖 Waymo 路径**，只依赖已导出的 npz。适合在“提取完 episode 之后、聚类之前”做一次分布检查。

---

## 四、步骤 3：单 Episode 轨迹可视化（可选）

**脚本**：`bc_baseline/scripts/visualize_episode_trajectory.py`

**作用**：按某种规则（如 min_ttc 最小、min_pet 最小、jerk_peak 最大，或随机）选出若干 episode，对**每一个**在二维平面画出 ego 与 partner 在 [t_start, t_end] 内的轨迹，并标出 **t_peak** 位置及该 episode 的 8 维特征数值。

- **依赖 Waymo 场景**：需要能通过 `BCExpertEnv` 加载对应 scenario_index 的 .pkl，以便读取轨迹位置。
- 输入：`--waymo_dir`、`--meta`（episode_meta.json）、`--features`（episode_features.npz）、`--n`（选几个）、`--sort_by`（min_ttc / min_pet / jerk_peak / random）、`--output_dir`。
- 输出：多张 PNG，如 `rank000_ep123.png`，表示按排序规则选出的第 0 个 episode（全局第 123 条）的轨迹图。

用途：人工查看“最危险”（min_ttc 小）或“最不平顺”（jerk 大）的片段，核对提取与特征是否合理。

---

## 五、步骤 4：驾驶风格聚类

**脚本**：`bc_baseline/scripts/cluster_driving_styles.py`

**依赖**：必须先有 `episode_features.npz` 和 `episode_meta.json`（即先完成步骤 1），且 npz 中需包含 `feature_names` 并与脚本内定义的 8 维顺序一致。

### 5.1 聚类用特征（5 维）

聚类时**不是**用全部 8 维，而是用 5 维核心特征（索引 3,4,5,2,6）：

- response_mean_acc、response_min_acc、mean_thw、jerk_peak、mean_speed_ratio  

排除：mean_acc、min_acc（由 response 系列取代）、relative_speed（与 mean_speed_ratio 高度相关）。

### 5.2 两种运行模式

**（1）Elbow 模式**：用于选聚类数 K。

- 对 K=2～10 分别做 K-Means（特征先 StandardScaler），计算 SSE、Silhouette、DBI。
- 输出：`elbow_analysis.png`（三联图）、`elbow_table.csv`（K 与三指标表格）。  
  根据 Silhouette 最大或 DBI 最小等确定一个合适的 K。

**（2）Cluster 模式**：按指定 K 做一次聚类并写标签与报告。

- 输出：
  - **style_labels.json**：结构为 `{scenario_index: {track_id: cluster_label}}`。同一辆车若有多个 episode，先得到多个标签，再按**众数**确定该车的风格标签；平票时用 min_ttc 最小的那条 episode 的标签代表该车。
  - **episode_labels.json**：在每条 episode 的 meta 上附加 `cluster_label` 和 `semantic_label`（如 conservative / normal / aggressive）。
  - **cluster_centers.json**：各簇中心在 5 维上的物理值及语义标签。
  - **cluster_report.txt**：每簇样本数、中心、语义标签，以及“同一车多 episode 标签一致率”等简要分析。

语义标签规则（由聚类中心相对大小自动分配）：response_mean_acc 最小的簇标为 conservative，最大的标为 aggressive，中间按 mean_thw 再细分为 normal / normal_1 / normal_2 等。

### 5.3 常用命令示例

```bash
# 选 K
python -m bc_baseline.scripts.cluster_driving_styles \
    --features_path bc_baseline/outputs/interaction_episodes/episode_features.npz \
    --meta_path bc_baseline/outputs/interaction_episodes/episode_meta.json \
    --mode elbow \
    --output_dir bc_baseline/outputs/driving_style

# 正式聚类（例如 K=4）
python -m bc_baseline.scripts.cluster_driving_styles \
    --features_path bc_baseline/outputs/interaction_episodes/episode_features.npz \
    --meta_path bc_baseline/outputs/interaction_episodes/episode_meta.json \
    --mode cluster --k 4 \
    --output_dir bc_baseline/outputs/driving_style
```

---

## 六、聚类之后可以做什么（与“可视化”的关系）

- **cluster_report.txt** 与 **cluster_centers.json**：直接阅读，了解各风格簇的物理含义与样本占比。
- **episode_labels.json**：每条 episode 带有 `cluster_label` 和 `semantic_label`，可用来：
  - 按风格筛选 episode 做进一步分析或训练；
  - 若需要“按聚类结果可视化”，可自行写脚本：读 episode_labels.json，按 cluster_label 筛选或着色，再调用与 `visualize_episode_trajectory` 类似的逻辑画轨迹（当前仓库未内置“按 cluster 着色的轨迹图”脚本）。
- **style_labels.json**：以“车”为单位的风格标签，适合做 per-vehicle 分析或与仿真中车辆 ID 对应。

当前流程中，“聚类后的可视化”主要体现在**报告与中心解读**；若要把“聚类标签”和“轨迹图”结合，需要基于 episode_labels.json 做一次按簇筛选再调用轨迹绘图。

---

## 七、流程顺序小结（可直接记入科研日志）

1. **提取 Episode**  
   运行 `extract_interaction_episodes.py`，从 Waymo .pkl 中找出所有 TTC 低于阈值的交互窗口，为每个窗口计算 8 维特征并写入 `episode_features.npz` 与 `episode_meta.json`。

2. **特征分布可视化（可选）**  
   运行 `visualize_episodes.py`，对 8 维特征画直方图，检查分布与量纲。

3. **单 Episode 轨迹可视化（可选）**  
   运行 `visualize_episode_trajectory.py`，按 min_ttc / min_pet / jerk_peak 或随机选取若干 episode，在二维平面画出 ego 与 partner 轨迹并标出 t_peak 与特征值，用于定性检查。

4. **聚类**  
   先运行 `cluster_driving_styles.py --mode elbow` 选 K，再 `--mode cluster --k K` 得到 style_labels、episode_labels、cluster_centers、cluster_report。

5. **后续**  
   根据 cluster_report 与 cluster_centers 理解各风格；用 episode_labels / style_labels 做按风格筛选、分析或接入 IRL 等下游流程。

以上即“提取 episode → 可视化特征/轨迹 → 聚类 → 利用聚类结果”的完整工作流，可按此顺序记录在科研日志中。
