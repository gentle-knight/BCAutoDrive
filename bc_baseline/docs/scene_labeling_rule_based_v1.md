# Scene Labeling（规则型 v1）脚本与原理说明

本文档对应代码目录 `bc_baseline/scene_labeling/`，用于说明这里所有脚本的作用、关键计算原理，并重点解释**道路类型 `road_type` 的划分依据**。

---

## 0) 你会用到什么（新手速览）

这套“规则型场景标签器”会把每个场景（`scene_id`）打上三个标签：

- `control_type`：是 `none / stop_controlled / signalized` 里的哪一种（来自地图里的动态车道状态或 stop sign 数量）
- `road_type`：是 `freeway / urban_road / merge_ramp / intersection` 里的哪一种（来自轨迹几何统计：并行性、路口冲突 proxy、merge proxy）
- `interaction_type`：是 `following / lane_change / merge / crossing_conflict / turning_conflict / mixed` 里的哪一种（来自轨迹交互候选）

你最终会得到三个落盘文件（默认输出路径见第 3 节）：

- `scene_labels.json`：每个 `scene_id` 的三个标签、置信度、以及可追溯的 `rule_trace`
- `scene_features.csv`：把用于规则的特征“摊平”为表格，便于你用 Excel/脚本核对阈值
- `review_samples.json`：从不同标签里抽样一批 `scene_id`，用于人工复核

---

## 0.1 快速开始（最常用的两条命令）

1) 先生成标签与特征（规则执行入口）

```bash
python -m bc_baseline.scene_labeling.label_scenes \
  --input legacy_magail/data/exp_filtered/ \
  --output_dir bc_baseline/scene_labeling/outputs \
  --samples_per_class 20 \
  --seed 42
```

2) 再把标签分组导出成 ScenarioNet 子集（可选）

```bash
python -m bc_baseline.scene_labeling.export_labeled_subsets \
  --dataset_dir legacy_magail/data/exp_filtered/ \
  --scene_labels_json bc_baseline/scene_labeling/outputs/scene_labels.json \
  --output_root bc_baseline/scene_labeling/outputs/labeled_subsets \
  --exist_ok
```

---

## 1) 目录与脚本清单

`bc_baseline/scene_labeling/` 中的脚本/模块分工如下：

- `label_scenes.py`：入口脚本。遍历输入场景 `.pkl`，抽取地图与轨迹特征，应用规则得到 `control_type / road_type / interaction_type`，并落盘 `scene_labels.json / scene_features.csv / review_samples.json`。
- `export_labeled_subsets.py`：把 `scene_labels.json` 中的标签分组，把对应场景从原数据集中导出成多个 ScenarioNet 子集（通过写 `dataset_summary.pkl` + `dataset_mapping.pkl`）。
- `map_parser.py`：解析 `map_features` 与 `dynamic_map_states`，输出地图/交通控制相关的特征（如车道数量、stop sign、是否存在动态车道状态等）。
- `track_parser.py`：解析 `tracks`，计算交通/拓扑/交互候选相关的特征（如车速统计、lane-change 候选、merge 候选、crossing 候选、转向轨迹计数、以及 pairwise 的 TTC/THW 统计等）。
- `interaction_utils.py`：供 `track_parser.py` 调用的几何/统计工具函数（lane-change/merge 的打分近似、圆周统计、TTC/THW 的计算等）。
- `scene_rules.py`：核心规则引擎。把 `SceneFeatures` 映射为三个标签（`control_type / road_type / interaction_type`），并生成可解释的 `rule_trace`。
- `schemas.py`：类型定义（`RoadType / InteractionType / ControlType`、`SceneFeatures`、`SceneLabel`、`RuleTraceItem` 等）。

---

## 2) 总体数据流（从输入到输出）

入口脚本 `label_scenes.py` 的处理链路可以概括为：

1. 读输入（`--input` 可以是单个 `scenario_xxx.pkl` 或整个场景目录）
2. 遍历所有 `.pkl` 场景文件（按文件存在性/目录结构做多策略发现）
3. 对每个场景：
   1. `map_parser.extract_map_features()` → 地图与动态控制特征
   2. `track_parser.extract_track_features()` → 车辆/轨迹与交互相关特征
   3. `scene_rules.label_scene()` → 根据阈值规则得到 `control_type / road_type / interaction_type`，并记录 `rule_trace`
4. 写输出：
   - `scene_labels.json`：每个 `scene_id` 的标签 + `rule_trace` + `features`
   - `scene_features.csv`：`FEATURE_COLUMNS` 指定字段的平铺表，便于阈值排查/统计分析
   - `review_samples.json`：按三类标签抽样一组 scene_id，用于人工复核

---

## 3) `label_scenes.py`：入口脚本做了什么

### 3.1 scene_id 如何获得

`_scene_id_from_scenario()` 优先级：

1. `scenario["id"]`
2. `scenario["metadata"]["scenario_id"]`（部分转换器使用这个字段）
3. 退化：使用 pkl 文件名（去掉扩展名）

### 3.2 输入 `.pkl` 的发现逻辑

`_discover_pkl_files(input_path)` 支持三种情况：

1. 目录下直接递归找 `.pkl`（优先，忽略 `dataset_mapping.pkl / dataset_summary.pkl`）
2. 若找到 `dataset_mapping.pkl`，则用它映射场景文件名 → 场景目录，拼出真实 scenario pkl 路径
3. 兜底：递归扫描所有 `.pkl`（仍跳过 mapping/summary）

### 3.3 输出里有哪些关键结构

`scene_labels.json` 对每个 scene_id 保存：

- `control_type` / `road_type` / `interaction_type`：每个标签是 `{label, confidence, rule_trace}` 的 JSON 可序列化形式
- `rule_trace`：规则引擎按步骤聚合后的全局 trace
- `features`：本场景用到的所有 `SceneFeatures`（来自 map + track）

同时 `scene_features.csv` 用 `FEATURE_COLUMNS` 列出一组固定列，方便你直接用表做阈值/特征的可视化或 sanity check。

### 3.4 review 抽样逻辑

脚本先把 scene_id 分组到：

- `by_control_type`
- `by_road_type`
- `by_interaction_type`

然后 `random.Random(seed)` 从每组里抽取 `samples_per_class` 个（不足则全取），写入 `review_samples.json`。

### 3.5 你如何“读懂输出文件”（建议新手先做）

1) 打开 `scene_labels.json`，挑一个你关心的 `scene_id`：先找到 `road_type.label` 看最终类别，再在对应的 `rule_trace` 里确认触发顺序与触发条件（代码是按 `control_type -> road_type -> interaction_type` 依次执行的）。
2) 用 `scene_features.csv` 核对该 `scene_id` 的关键特征值：`num_crossing_candidates / num_merge_candidates / parallel_lane_ratio / p90_vehicle_speed`。
3) 如果最终类别和你的直觉不一致：先看 `rule_trace` 是否被 `intersection` 的优先级截断（因为 intersection 是最高优先级），再回到 `track_parser.py` 检查 crossing/merge 的 proxy 计数逻辑（第 8 节详细讲）。

---

## 4) `map_parser.py`：道路控制与地图特征如何计算

`extract_map_features(scenario)` 从两个字段读取：

- `scenario["map_features"]`：地图静态结构（车道、路边界、路口人行横道、stop sign、减速带等）
- `scenario["dynamic_map_states"]`：动态控制（转换后通常对应交通灯等）

### 4.1 关键特征与计算方式

- `num_lanes`：遍历 `map_features`，当 `feat["type"]` 包含 `"LANE_"` 时计数 `num_lanes += 1`。
- `num_lane_boundaries`：对每个 lane feature，如果存在 `left_boundaries/right_boundaries` 列表，分别累加长度。
- `num_road_boundaries`：统计 `feat["type"].startswith("ROAD_EDGE_")` 的条目数。
- `num_crosswalks / num_stop_signs / num_speed_bumps`：分别统计 type 等于 `CROSSWALK / STOP_SIGN / SPEED_BUMP` 的条目数。
- `has_dynamic_lane_state`：
  - 遍历 `dynamic_map_states` 里的动态对象；
  - 若其中存在 `lane` 且 `state["object_state"]` 序列里出现了非 `None` 状态，则判定 `has_dynamic_lane_state=True`。

该 `has_dynamic_lane_state` 会直接影响 `scene_rules.py` 里的 `control_type`。

---

## 5) `track_parser.py`：轨迹特征与交互候选如何计算

`extract_track_features(scenario, config)` 主要分三块：

1. 从 `tracks` 中筛出车辆轨迹并判断“moving vs static”
2. 计算单车轨迹的 lane-change / merge / turning 等计数与速度统计
3. 对“车对”进行 pairwise 计算，得到：
   - crossing候选计数（路口冲突的 proxy）
   - `mean_min_ttc / mean_min_thw`（风险强度统计）

### 5.1 moving / static 的判定

轨迹是否静止由 `_is_vehicle_static()` 给出：

- 有效帧不足 2：直接认为静止
- 有效帧位移 `disp = ||pos[t_end] - pos[t_start]|| < static_displacement_threshold_m`
- 同时最大速度 `max_speed < static_speed_threshold_mps`

满足以上两者 → `is_static=True`，则该轨迹不参与后续的统计（包括 parallel_lane_ratio 与 pairwise 交互候选）。

### 5.2 lane-change 候选（`num_lane_changes`）

对每个 moving vehicle：

- 调用 `interaction_utils.lane_change_score_from_positions(...)`
- 若返回 `is_lane_change=True`，则 `lane_change_count += 1`

lane-change 的近似依据（核心思想）：

- 把世界坐标的位移旋转到“初始航向”为参考系；
- 看侧向（横向）位移是否达到阈值，并且有足够持续时间的侧向偏离（用 median 附近的离散帧计数估计转向过程）。

阈值入口在 `LabelingConfig`：

- `lane_change_lat_delta_threshold_m`
- `lane_change_min_transition_frames`

### 5.3 merge 候选（`num_merge_candidates`）

对每个 moving vehicle：

- 用初始航向参考系得到侧向序列 `y_lat`
- 调用 `interaction_utils.lane_merge_score_from_lateral(...)`：
  - `abs(lat_total) = abs(y[-1] - y[0])` 必须落在 `[merge_lat_min, merge_lat_max]`
  - `y` 的侧向变化需要近似单调（通过 `np.sign(dy)` 与整体趋势的匹配比例估计）
  - 航向变化 `d_heading` 必须不超过 `merge_heading_change_max_rad`

通过则 `merge_candidate_count += 1`。

### 5.4 turning 轨迹（`num_turning_tracks`）

对每个 moving vehicle：

- 计算起止航向差 `d_heading`（使用圆周角差）
- 若 `d_heading >= turning_heading_change_min_rad`，则 `turning_tracks_count += 1`

### 5.5 速度与航向统计（给 road_type 用的关键特征之一）

- `mean_vehicle_speed`：所有 moving vehicle 的速度均值
- `p90_vehicle_speed`：所有 moving vehicle 的速度 90 分位数（Road_type freeway 判断会用）
- `heading_dispersion`：所有 moving vehicle 的航向样本做圆周标准差（`circular_std`）

### 5.6 parallel_lane_ratio（给 road_type 用的关键特征之一）

`parallel_lane_ratio` 是一个“航向一致性比例”：

- 收集所有 moving vehicles 的航向样本
- 先用圆周均值得到“整体平均航向”`mean_angle`
- 再统计偏差 `|wrap(angles - mean_angle)| <= parallel_heading_window_rad` 的样本占比

该比例越高，表示更多车辆与整体航向接近，从而更像“高速并行道路”的交通形态。

---

## 6) pairwise 交互候选：crossing_candidates 与 TTC/THW 统计

`track_parser` 的交互候选计算在 moving vehicles 上做 pairwise（N^2 复杂度，但有 `pairwise_max_vehicle_count` 的上限保护）。

对每一对车辆 (i, j)：

1. 找重叠的有效时间帧索引 `valid_overlap`
2. 在重叠帧上计算两车距离，取最小距离对应的帧 `t_min`
3. 若最小距离 `dist_min > candidate_interaction_distance_m`，则认为它们交互不强 → 跳过
4. crossing 候选（路口冲突 proxy）：
   - 在 `t_min` 处计算航向差 `hd_deg`
   - 若 `crossing_heading_diff_min_deg <= hd_deg <= crossing_heading_diff_max_deg`，则 `crossing_cands += 1`
5. TTC/THW：
   - 计算 ego=i 相对于 other=j 的 TTC 序列，并取 `min_ttc`
   - 计算 THW 序列（要求 other 在 ego 前方且 ego_speed 足够），并取 `min_thw`
6. 把每个候选 pair 的 `min_ttc/min_thw` 加入列表，最后对候选 pair 求平均得到：
   - `mean_min_ttc`
   - `mean_min_thw`

> 说明：`road_type` 和 `interaction_type` 里会用到 `crossing_candidates`（由 pairwise crossing_cands 统计）以及 `mean_min_ttc`（转向冲突判断的一部分）。

---

## 7) `scene_rules.py`：标签规则如何落到 control/road/interaction

`scene_rules.label_scene()` 按顺序执行：

1. `control_type`
2. `road_type`
3. `interaction_type`

并为每个步骤生成可解释的 `rule_trace` 条目（`step / rule / condition / result / details / confidence`）。

阈值都集中在 `LabelingConfig`，文档后面单独列出 road_type 相关参数。

---

## 8) 重点：`road_type` 划分依据（intersection / merge_ramp / freeway / urban_road）

`road_type` 是一个“规则打分 + 明确优先级”的分类器。它只看 `SceneFeatures` 里几个字段，并且按下面顺序做判定（这是最重要的部分：**先命中谁就直接定类**，不会再看后面的条件）：

1. 命中 `intersection`（路口）
   - 触发条件：`num_crossing_candidates >= crossing_candidates_high`（默认 `>= 5`）
2. 否则命中 `merge_ramp`（并入/匝道）
   - 触发条件：`num_merge_candidates >= merge_candidates_high`（默认 `>= 3`）
3. 否则命中 `freeway`（高速并行道路）
   - 触发条件需要同时满足：
     - `parallel_lane_ratio >= parallel_lane_ratio_threshold`（默认 `>= 0.65`）
     - `p90_vehicle_speed >= p90_vehicle_speed_threshold`（默认 `>= 18.0`，单位 m/s）
     - `num_crossing_candidates <= crossing_candidates_few`（默认 `<= 1`，即路口冲突 proxy 不要多）
4. 否则落到 `urban_road`（城市一般道路）

### 8.1 每个字段（上面的特征）在代码里怎么来的

下面把 `road_type` 需要的 4 个特征，逐个对应到具体代码/计算逻辑。

#### 8.1.1 `num_crossing_candidates`（路口冲突 proxy）

来源：`track_parser.extract_track_features()` 中 pairwise 车辆对统计变量 `crossing_cands`。

对每一对“moving vehicles” (i, j)，大致做：

1. 取二者在时间上重叠的有效帧集合 `valid_overlap`
2. 在重叠帧里计算两车距离，找最接近的时刻：`dist_min` 是最小距离，`t_min` 是达到最小距离的帧索引。
3. 如果最小距离太远（默认 `candidate_interaction_distance_m = 8.0`），认为它们交互不强，跳过
4. 在 `t_min` 处计算航向差（用的是两车航向的圆周角差），将航向差转换为角度 `hd_deg`
5. 若 `hd_deg` 落在 `[crossing_heading_diff_min_deg, crossing_heading_diff_max_deg]`（默认 `[60, 120]`），则认为这个车对“像路口冲突”，执行 `crossing_cands += 1`

最后把所有车对的 `crossing_cands` 累加并作为 `scene_features["num_crossing_candidates"]`。

> 直觉理解：它不是直接看“路口几何”，而是用“最近接近时刻的相对航向是否接近交叉（大概 90 度）”来做 proxy。

#### 8.1.2 `num_merge_candidates`（并入/变道 merge proxy）

来源：`track_parser.extract_track_features()` 对每辆 moving vehicle 计算 `merge_candidate_count`。

对每个 moving vehicle：

1. 在初始航向参考系里把世界位移旋转成横向（侧向）序列 `y_lat`
2. 调用 `interaction_utils.lane_merge_score_from_lateral(y_lat_seq=..., heading_seq=..., valid_idx=...)`
3. 只要满足 lane_merge_score 的条件，就认为该车在该场景里“有 merge 候选”，累加 `merge_candidate_count += 1`

lane_merge_score 的关键判断条件（默认阈值）：

- `abs(lat_total)` 落在 `[merge_lat_min, merge_lat_max]`，默认 `[1.0, 4.0]`
- 侧向变化的趋势近似单调：用 `np.sign(dy)` 方向一致比例（`monotonic_ratio_threshold` 默认 `0.7`）衡量
- 起止航向变化不超过 `merge_heading_change_max_rad`，默认 `15 deg`（用弧度阈值存储）

最终得到 `scene_features["num_merge_candidates"]`。

> 直觉理解：它不是检测真实的匝道几何，而是检测“侧向移动范围 + 侧向趋势 + 航向变化是否符合并入机理”的 proxy。

#### 8.1.3 `parallel_lane_ratio`（并行/同向的一致性比例）

来源：`track_parser` 从所有 moving vehicle 收集航向样本后计算。

计算步骤：

1. 收集所有 moving vehicles 的航向样本
2. 计算圆周均值航向 `mean_angle`（用圆周统计，避免角度跨越导致的均值错误）
3. 统计满足 `|wrap(angles - mean_angle)| <= parallel_heading_window_rad`（默认 `10 deg`）的样本比例

比例越高说明整体交通流更“同向并行”，因此更像 freeway。

#### 8.1.4 `p90_vehicle_speed`（高速的速度证据）

来源：`track_parser` 对所有 moving vehicles 的速度（在有效帧上取范数）求 90 分位数。

最后把它作为 `scene_features["p90_vehicle_speed"]`。

---

### 8.2 confidence（置信度）如何产生（理解即可，不影响类别）

`road_type` 的类别由上面的阈值“命中”决定；置信度只是为了表示“命中的强弱”。

- `intersection/merge_ramp/freeway`：置信度来自命中强度 `hit_strength` 的非线性映射（`_confidence_from_hit_strength`，本质是 `1 - exp(-hit_strength)` 并裁剪到 `[0,1]`）
- `urban_road`：不满足前三类条件时，固定给较低置信度（默认 `0.55`）

> 你可以把它当作可解释的“软指标”：`road_type` 类别看硬规则，confidence 只是额外信息。

---

## 9) `interaction_type` 概览（用于完整性，不展开到每个细节）

`interaction_type` 使用 `track_parser` 输出的计数特征做组合命中：

- `following`：lane_change 与 merge 候选尽量少，crossing 候选也少（且至少存在一些 candidate interactions）
- `lane_change`：lane_changes 足够多，且 crossing 候选不多
- `merge`：merge_candidates 足够多，且 crossing 候选不多
- `crossing_conflict`：crossing_candidates 足够多
- `turning_conflict`：turning_tracks 足够多，且 `mean_min_ttc` 不大（更危险）
- `mixed`：如果满足的命中规则数达到 `mixed_min_hit_count (2)`，则标记为 mixed；否则按强命中顺序选单一规则，否则默认 `following`

---

## 10) `export_labeled_subsets.py`：如何从标签导出 ScenarioNet 子集

入口 `export_labeled_subsets.py` 逻辑：

1. 读取 `scene_labels.json`，得到每个 scene_id 的三类标签
2. 读取原数据集目录中的：
   - `dataset_summary.pkl`
   - `dataset_mapping.pkl`
3. 通过 `--scene_id_regex`（默认 `^sd_waymo_v1\\.2_(.*)\\.pkl$`）从 dataset_summary 的 scenario 文件名提取 scene_id
4. 对每个类别（`control_type / road_type / interaction_type`）和每个 label_value：
   - 生成一个导出计划 `ExportPlan`
   - 创建子集根目录：`<output_root>/by_<category>/<label_value>/`
   - 写入该子集的 `dataset_summary.pkl / dataset_mapping.pkl`

### review 模式

若开启 `--use_review_samples`：

- 只允许 `review_samples.json` 中出现的 scene_id 参与导出
- 每个类别会按 `by_control_type / by_road_type / by_interaction_type` 分别过滤

---

## 11) road_type 相关参数一览（可调参入口）

`LabelingConfig` 中 road_type 直接使用的阈值：

- `crossing_candidates_high = 5`（intersection）
- `merge_candidates_high = 3`（merge_ramp）
- `parallel_lane_ratio_threshold = 0.65`（freeway 条件）
- `p90_vehicle_speed_threshold = 18.0`（freeway 条件，单位 m/s）
- `crossing_candidates_few = 1`（freeway 条件：crossing 不要太多）

如需调整道路划分粗细程度，通常优先调这几个阈值；同时也建议你同步检查 `track_parser` 中 crossing/merge 的生成逻辑是否符合你的直觉标注口径。

