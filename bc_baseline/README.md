# BC Baseline

基于 **MetaDrive** 仿真器和 **Waymo Open Motion Dataset**（经 ScenarioNet 转换）的自动驾驶行为克隆（Behavior Cloning）训练系统。

本模块从 Waymo 轨迹中提取 Ego-centric 观测与逆动力学动作，训练 MLP 策略网络，并在闭环仿真中评估驾驶性能。

## 目录结构

```text
bc_baseline/
├── Algorithm/                 # 策略网络
│   └── bc_net.py              # BCActor：51 维观测 → 2 维动作
├── Env/                       # 环境封装
│   ├── expert_env.py          # BCExpertEnv：离线专家数据提取（轨迹 → obs/action）
│   ├── eval_env.py            # BCEvalEnv：闭环评估（Ego 用策略，背景车 log-replay）
│   └── utils.py               # extract_ego_observation：51 维 Ego-centric 观测
├── datasets/
│   └── bc_dataset.py          # BCDataset：从 .npz 加载 (obs, action)
├── scripts/                       # 工具脚本
│   ├── generate_bc_data.py        # 从 Waymo .pkl 生成 bc_training_data.npz
│   ├── cluster_driving_styles.py  # 驾驶风格聚类（elbow / cluster）
│   ├── extract_interaction_episodes.py  # 提取强交互 episodes 及 8 维特征
│   ├── visualize_episode_trajectory.py  # 可视化单个 interaction episode 的轨迹
│   ├── visualize_episodes.py     # 统计 episode 特征分布直方图
│   ├── eval_bc.py                 # 闭环评估：Success / Collision / Out-of-Road
│   └── visualize_bc.py            # 单场景可视化回放
├── outputs/                       # 默认输出目录（相对项目根）
│   ├── data/                      # bc_training_data.npz
│   ├── checkpoints/               # best_bc.pth, last_bc.pth
│   ├── logs/                      # TensorBoard
│   ├── driving_style/             # 驾驶风格聚类相关输出
│   │   ├── elbow_curve.png
│   │   ├── elbow_table.csv
│   │   ├── elbow_table.png
│   │   └── style_labels.json
│   └── interaction_episodes/      # 交互 episode 相关输出
│       ├── episode_features.npz
│       ├── episode_meta.json
│       ├── feature_distributions.png
│       └── episode_vis/           # 单 episode 轨迹可视化 PNG
├── train.py                   # BC 训练入口
└── README.md
```

## 路径约定（相对项目根）

- **数据**：Waymo 场景 `data/exp_filtered`（或自定义）；BC 训练数据 `bc_baseline/outputs/data/`
- **模型**：Checkpoint 保存 `bc_baseline/outputs/checkpoints/`
- **日志**：TensorBoard 写入 `bc_baseline/outputs/logs/tensorboard/`

---

## 输入输出维度

### 观测空间（obs）：51 维

| 部分 | 维度 | 说明 |
|------|------|------|
| **Ego 特征** | 11 | lateral_offset, heading_error, vel_longitudinal, vel_lateral, yaw_rate(=0), waypoints_ego(3×2) |
| **Neighbor 特征** | 40 | 10 个槽位 × 4 维：(rel_x, rel_y, vel_x, vel_y)，自车坐标系 |

- **lateral_offset**：自车相对车道中心线横向偏移（米）
- **heading_error**：自车航向与车道切向偏差角（弧度）
- **vel_longitudinal / vel_lateral**：自车坐标系下纵向/横向速度（m/s）
- **yaw_rate**：本基线中固定为 0.0
- **waypoints_ego**：10m、20m、30m 处车道中心线在自车坐标系下的 (rel_x, rel_y)，共 6 维
- **Neighbor**：30m 半径内最近 10 辆车，每车 4 维（位置 + 速度，自车坐标系）

### 动作空间（action）：2 维

| 索引 | 含义 | 范围 |
|------|------|------|
| 0 | steering（前轮转角，归一化） | [-1, 1] |
| 1 | acceleration（纵向加速度，归一化） | [-1, 1] |

逆动力学模块将轨迹反推为物理量后，按 `max_steering=0.7 rad`、`max_acc=8.0 m/s²` 归一化到 [-1, 1]。

### 模型 I/O

| 模块 | 输入 | 输出 |
|------|------|------|
| **BCActor** | obs: (B, 51) | action: (B, 2)，Tanh 输出 [-1, 1] |
| **BCDataset** | .npz 文件 | obs (N, 51), actions (N, 2) |

---

## 全流程命令

> **说明**：以下命令均在项目根目录 `MAGAIL4AutoDrive/` 下执行。

### 0. 数据准备（Waymo → ScenarioNet）

与 `legacy_magail` 相同，需先将 Waymo TFRecord 转为 ScenarioNet .pkl：

```bash
# 1) 下载 Waymo Motion（示例）
gsutil -m cp -r "gs://waymo_open_dataset_motion_v_1_2_0/uncompressed/scenario/training_20s" ./waymo/

# 2) ScenarioNet 转换
python -m scenarionet.convert_waymo -d data/exp_converted --raw_data_path ./waymo/training_20s --num_workers 64

# 3) 按需筛选场景（红绿灯、天桥等）
# 输出到 data/exp_filtered
```

### 1. 生成 BC 训练数据

从 ScenarioNet .pkl 场景中提取 (obs, action) 对，输出 .npz：

```bash
python -m bc_baseline.scripts.generate_bc_data \
    --waymo_dir /path/to/exp_filtered \
    --output_dir bc_baseline/outputs/data \
    --num_scenarios 348 \
    --start_index 0 \
    --waymo_dt 0.1
```

**输出**：`bc_baseline/outputs/data/bc_training_data.npz`

- `obs`：shape (N, 51)，dtype float32
- `actions`：shape (N, 2)，dtype float32

### 2. 模型训练

```bash
python -m bc_baseline.train \
    --data_path bc_baseline/outputs/data/bc_training_data.npz \
    --batch_size 256 \
    --lr 3e-4 \
    --epochs 50 \
    --val_ratio 0.2 \
    --num_workers 4 \
    --device cuda \
    --log_dir bc_baseline/outputs/logs/tensorboard \
    --ckpt_dir bc_baseline/outputs/checkpoints
```

**输出**：`best_bc.pth`、`last_bc.pth`、TensorBoard 日志

### 3. 闭环评估

```bash
python -m bc_baseline.scripts.eval_bc \
    --ckpt_path bc_baseline/outputs/checkpoints/best_bc.pth \
    --waymo_dir /path/to/exp_filtered \
    --num_scenarios 100 \
    --start_index 0 \
    --device cuda
```

**输出**：终端打印 Success Rate、Collision Rate、Out of Road Rate

### 4. 可视化回放

```bash
python -m bc_baseline.scripts.visualize_bc \
    --ckpt_path bc_baseline/outputs/checkpoints/best_bc.pth \
    --waymo_dir /path/to/exp_filtered \
    --scenario_index 0 \
    --top_down \
    --sleep 0.05
```

### 5. 驾驶风格聚类（可选）

用于对轨迹进行风格聚类，辅助后续分析或按风格训练：

```bash
# 肘部法确定最优 K
python -m bc_baseline.scripts.cluster_driving_styles \
    --waymo_dir /path/to/exp_filtered \
    --num_scenarios 348 \
    --mode elbow \
    --output_dir bc_baseline/outputs/driving_style    # 可选，默认同此路径

# 正式聚类（需指定 K）
python -m bc_baseline.scripts.cluster_driving_styles \
    --waymo_dir /path/to/exp_filtered \
    --num_scenarios 348 \
    --mode cluster --k 4 \
    --output_dir bc_baseline/outputs/driving_style    # 可选，默认同此路径
```

**输出目录**：`bc_baseline/outputs/driving_style/`

- mode elbow：
  - `elbow_curve.png`
  - `elbow_table.csv`
  - `elbow_table.png`
- mode cluster：
  - `style_labels.json`（{scenario_index: {track_id: cluster_label}}）

### 6. 提取强交互 episodes（可选）

```bash
python -m bc_baseline.scripts.extract_interaction_episodes \
    --waymo_dir /path/to/exp_filtered \
    --num_scenarios 348 \
    --start_index 0 \
    --waymo_dt 0.1 \
    --ttc_threshold 5.0 \
    --window_seconds 3.0 \
    --min_episode_frames 10 \
    --output_dir bc_baseline/outputs/interaction_episodes   # 可选，默认同此路径
```

**输出目录**：`bc_baseline/outputs/interaction_episodes/`

- `episode_features.npz`：shape (N, 8)，列顺序：
  - mean_speed, std_speed, max_speed,
  - mean_acc, min_acc, jerk_peak,
  - min_ttc, min_pet
- `episode_meta.json`：长度为 N 的列表，每个元素：
  - scenario_index, ego_track_id, partner_track_id,
  - t_peak, t_start, t_end,
  - min_ttc, min_pet

### 7. episode 相关可视化（可选）

1）**特征分布直方图**

```bash
python -m bc_baseline.scripts.visualize_episodes
```

**输出**：`bc_baseline/outputs/interaction_episodes/feature_distributions.png`

2）**单个 interaction episode 轨迹回放**

```bash
python -m bc_baseline.scripts.visualize_episode_trajectory \
    --waymo_dir /path/to/exp_filtered \
    --n 20 \
    --sort_by min_ttc \
    --output_dir bc_baseline/outputs/interaction_episodes/episode_vis
```

**输出目录**：`bc_baseline/outputs/interaction_episodes/episode_vis/`

- `rankXXX_epYYY.png`：按排序规则选出的若干 episode 轨迹图

---

## 数据流汇总

```text
Waymo TFRecord
    → ScenarioNet convert
    → .pkl 场景
    → generate_bc_data.py (BCExpertEnv + extract_ego_observation + InverseDynamics)
    → bc_training_data.npz (obs: N×51, actions: N×2)
    → train.py (BCDataset + BCActor)
    → best_bc.pth
    → eval_bc.py / visualize_bc.py (BCEvalEnv)
```

---

## 依赖

```bash
pip install scikit-learn matplotlib tqdm
# MetaDrive、ScenarioNet 需单独安装（conda 等）
```

详见 `bc_baseline/requirements.txt`。
