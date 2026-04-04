## Scene Labeling（规则型场景标签器 v1）

本目录包含一套“规则型、可解释”的场景标签流程，面向 ScenarioNet（Waymo Motion 转换后）数据。

**规则与阈值以代码为准**：`scene_rules.py` 中的 `LabelingConfig` + `label_scene()`。  
**完整说明文档**（脚本职责、`road_type`/`interaction_type` 判定、`rule_trace` 含义、默认阈值）：[`bc_baseline/docs/scene_labeling_rule_based_v1.md`](../docs/scene_labeling_rule_based_v1.md)。

**`road_type` 要点（摘要）**：融合地图特征（如 `num_crosswalks`、`num_stop_signs`）、`control_type`（如 `signalized`）与轨迹 proxy；优先级为 `intersection → freeway → merge_ramp → urban_road`；`freeway` 不再用 `p90_vehicle_speed` 做硬阈值（拥堵高速仍可判为 freeway，速度仅影响置信度）。

核心输出：
- `scene_labels.json`：每个 `scene_id` 的 `control_type / road_type / interaction_type` 标签 + `rule_trace`
- `scene_features.csv`：每个 `scene_id` 的特征证据（用于调阈值/排查）
- `review_samples.json`：按三类标签抽样的复核清单

此外还提供两个工具：
- `scripts/scene_id_to_scenario_index.py`：把 `scene_id`（哈希字符串）转成 `scenarionet.sim` 需要的整数 `--scenario_index`
- `export_labeled_subsets.py`：基于 `scene_labels.json` 把场景按标签分组成多个 ScenarioNet 子集数据库（每个 `<label>/` 是独立数据集根目录）

---

## 1. 生成规则标签

```bash
python -m bc_baseline.scene_labeling.label_scenes \
  --input legacy_magail/data/exp_filtered/ \
  --output_dir bc_baseline/scene_labeling/outputs \
  --samples_per_class 20 \
  --seed 42
```

输出：
- `scene_labels.json`
- `scene_features.csv`
- `review_samples.json`

---

## 2. 用 scene_id 复核到 scenarionet.sim 可视化

`scenarionet.sim --scenario_index` 需要整数索引，不接受 `scene_id` 字符串。

先转：

```bash
python -m bc_baseline.scripts.scene_id_to_scenario_index \
  -d legacy_magail/data/exp_filtered \
  --id a7545087f82dafeb
```

再渲染：

```bash
python -m scenarionet.sim -d legacy_magail/data/exp_filtered \
  --render 2D \
  --scenario_index <整数>
```

---

## 3. 导出按标签分组的 ScenarioNet 子集数据库

### 3.1 默认：导出全量（按三类标签三套分组）

```bash
python -m bc_baseline.scene_labeling.export_labeled_subsets \
  -d legacy_magail/data/exp_filtered/ \
  --scene_labels_json bc_baseline/scene_labeling/outputs/scene_labels.json \
  --output_root bc_baseline/scene_labeling/outputs/labeled_subsets \
  --exist_ok
```

会在 `output_root` 下生成（示例）：
- `by_control_type/stop_controlled/dataset_summary.pkl`
- `by_road_type/merge_ramp/dataset_summary.pkl`
- `by_interaction_type/mixed/dataset_summary.pkl`

每个 `<label>/` 都是独立的 ScenarioNet 数据集根目录：
- 目录内只有 `dataset_summary.pkl / dataset_mapping.pkl`
- scenario 的实际 `.pkl` 由 `dataset_mapping.pkl` 指向原始 `exp_converted/...`，不复制大文件

一个场景会同时出现在三类分组中（这是预期行为）。

### 3.2 开关：仅导出 review 样本子集

如果你只想导出 `review_samples.json` 里抽样出来的那些 `scene_id`：

```bash
python -m bc_baseline.scene_labeling.export_labeled_subsets \
  -d legacy_magail/data/exp_filtered/ \
  --scene_labels_json bc_baseline/scene_labeling/outputs/scene_labels.json \
  --output_root bc_baseline/scene_labeling/outputs/labeled_subsets_review \
  --use_review_samples
```

### 3.3 报告文件（统计每个 label 有多少场景）

脚本默认在 `output_root/export_report.json` 输出统计报告，包含：
- `control_type / road_type / interaction_type` 每个 label 的场景数
- 导出模式（all / review）
- 可选：`missing_info`

如需自定义报告路径：

```bash
python -m bc_baseline.scene_labeling.export_labeled_subsets \
  -d legacy_magail/data/exp_filtered/ \
  --scene_labels_json bc_baseline/scene_labeling/outputs/scene_labels.json \
  --output_root bc_baseline/scene_labeling/outputs/labeled_subsets \
  --report_path /tmp/export_report.json
```

