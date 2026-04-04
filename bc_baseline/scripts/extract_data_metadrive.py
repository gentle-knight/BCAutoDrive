import os
import glob
import argparse
import functools
import pickle
from typing import Any

import numpy as np
from tqdm.contrib.concurrent import process_map


MAX_NUM_OBJECTS = 64
MAX_POLYLINES = 256
MAX_TRAFFIC_LIGHTS = 16
CURRENT_INDEX = 10
NUM_POINTS_POLYLINE = 30
FUTURE_STEPS = 81

# MetaDrive converted directories contain helper index files; skip them.
SKIP_FILENAMES = {"dataset_mapping.pkl", "dataset_summary.pkl"}

# Keep agent type ids aligned with VBD/vbd/data/dataset.py::_process.
AGENT_TYPE_TO_ID = {
    "VEHICLE": 1,
    "PEDESTRIAN": 2,
    "CYCLIST": 3,
}

MAP_TYPE_TO_ID = {
    "UNKNOWN": 0,
    "LANE_FREEWAY": 1,
    "LANE_SURFACE_STREET": 2,
    "LANE_BIKE_LANE": 3,
    "ROAD_LINE_BROKEN_SINGLE_WHITE": 4,
    "ROAD_LINE_SOLID_SINGLE_WHITE": 5,
    "ROAD_LINE_SOLID_DOUBLE_WHITE": 6,
    "ROAD_LINE_BROKEN_SINGLE_YELLOW": 7,
    "ROAD_LINE_BROKEN_DOUBLE_YELLOW": 8,
    "ROAD_LINE_SOLID_SINGLE_YELLOW": 9,
    "ROAD_LINE_SOLID_DOUBLE_YELLOW": 10,
    "ROAD_LINE_PASSING_DOUBLE_YELLOW": 11,
    "ROAD_EDGE_BOUNDARY": 12,
    "ROAD_EDGE_MEDIAN": 13,
    "STOP_SIGN": 14,
    "CROSSWALK": 15,
    "SPEED_BUMP": 16,
    "DRIVEWAY": 17,
}

TRAFFIC_LIGHT_STATE_TO_ID = {
    "LANE_STATE_UNKNOWN": 0,
    "LANE_STATE_ARROW_STOP": 1,
    "LANE_STATE_ARROW_CAUTION": 2,
    "LANE_STATE_ARROW_GO": 3,
    "LANE_STATE_STOP": 4,
    "LANE_STATE_CAUTION": 5,
    "LANE_STATE_GO": 6,
    "LANE_STATE_FLASHING_STOP": 7,
    "LANE_STATE_FLASHING_CAUTION": 8,
}


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def calculate_relations(
    agents: np.ndarray,
    polylines: np.ndarray,
    traffic_lights: np.ndarray,
) -> np.ndarray:
    # Reuse the same relation construction logic as vbd/data/data_utils.py
    # so model input semantics remain unchanged across Waymax/MetaDrive sources.
    n_agents = agents.shape[0]
    n_polylines = polylines.shape[0]
    n_traffic_lights = traffic_lights.shape[0]
    n = n_agents + n_polylines + n_traffic_lights

    all_elements = np.concatenate(
        [
            agents[:, -1, :3],
            polylines[:, 0, :3],
            np.concatenate([traffic_lights[:, :2], np.zeros((n_traffic_lights, 1), dtype=np.float32)], axis=1),
        ],
        axis=0,
    )

    pos_diff = all_elements[:, :2][:, None, :] - all_elements[:, :2][None, :, :]
    cos_theta = np.cos(all_elements[:, 2])[:, None]
    sin_theta = np.sin(all_elements[:, 2])[:, None]
    local_pos_x = pos_diff[..., 0] * cos_theta + pos_diff[..., 1] * sin_theta
    local_pos_y = -pos_diff[..., 0] * sin_theta + pos_diff[..., 1] * cos_theta
    theta_diff = wrap_to_pi(all_elements[:, 2][:, None] - all_elements[:, 2][None, :])

    start_idx = n_agents + n_polylines
    theta_diff = np.where(
        (np.arange(n) >= start_idx)[:, None] | (np.arange(n) >= start_idx)[None, :],
        0,
        theta_diff,
    )

    diag_mask = np.eye(n, dtype=bool)
    epsilon = 0.01
    local_pos_x = np.where(diag_mask, epsilon, local_pos_x)
    local_pos_y = np.where(diag_mask, epsilon, local_pos_y)
    theta_diff = np.where(diag_mask, epsilon, theta_diff)

    zero_mask = np.logical_or(all_elements[:, 0][:, None] == 0, all_elements[:, 0][None, :] == 0)
    relations = np.stack([local_pos_x, local_pos_y, theta_diff], axis=-1)
    relations = np.where(zero_mask[..., None], 0.0, relations)

    return relations.astype(np.float32)


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_track_id(track_id: Any) -> str:
    return str(track_id)


def _pad_or_trim_1d(
    array: Any,
    length: int,
    dtype: np.dtype,
    fill_value: Any,
) -> np.ndarray:
    # Convert variable-length arrays into fixed-length vectors.
    arr = np.asarray(array, dtype=dtype).reshape(-1)
    if arr.shape[0] >= length:
        return arr[:length]
    out = np.full((length,), fill_value, dtype=dtype)
    out[: arr.shape[0]] = arr
    return out


def _pad_or_trim_2d(
    array: Any,
    length: int,
    width: int,
    dtype: np.dtype,
    fill_value: Any,
) -> np.ndarray:
    # Convert variable-length matrices into fixed [length, width] tensors.
    arr = np.asarray(array, dtype=dtype)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        arr = np.zeros((0, width), dtype=dtype)

    if arr.shape[1] < width:
        arr = np.pad(arr, ((0, 0), (0, width - arr.shape[1])), constant_values=fill_value)
    elif arr.shape[1] > width:
        arr = arr[:, :width]

    if arr.shape[0] >= length:
        return arr[:length]
    out = np.full((length, width), fill_value, dtype=dtype)
    out[: arr.shape[0]] = arr
    return out


def _agent_type_to_id(agent_type: Any) -> int:
    if isinstance(agent_type, (int, np.integer)):
        return int(agent_type)

    text = str(agent_type).upper()
    if "VEHICLE" in text:
        return AGENT_TYPE_TO_ID["VEHICLE"]
    if "PEDESTRIAN" in text:
        return AGENT_TYPE_TO_ID["PEDESTRIAN"]
    if "CYCLIST" in text:
        return AGENT_TYPE_TO_ID["CYCLIST"]
    return 0


def _extract_map_points(map_feature: dict[str, Any]) -> np.ndarray:
    # Normalize heterogeneous map geometry into a polyline-like array.
    # Priority: polyline > polygon > single position point.
    if "polyline" in map_feature:
        points = np.asarray(map_feature["polyline"], dtype=np.float32)
        if points.ndim == 2 and points.shape[0] > 0 and points.shape[1] >= 2:
            return points

    if "polygon" in map_feature:
        points = np.asarray(map_feature["polygon"], dtype=np.float32)
        if points.ndim == 2 and points.shape[0] > 0 and points.shape[1] >= 2:
            if points.shape[0] > 1 and not np.allclose(points[0, :2], points[-1, :2]):
                points = np.vstack([points, points[0]])
            return points

    if "position" in map_feature:
        point = np.asarray(map_feature["position"], dtype=np.float32).reshape(-1)
        if point.shape[0] >= 2:
            if point.shape[0] < 3:
                point = np.pad(point, (0, 3 - point.shape[0]), constant_values=0.0)
            return point[None, :3]

    return np.zeros((0, 3), dtype=np.float32)


def _compute_headings(points_xy: np.ndarray) -> np.ndarray:
    # Heading for each point is computed from local segment direction.
    n = points_xy.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.float32)
    if n == 1:
        return np.zeros((1,), dtype=np.float32)

    deltas = np.diff(points_xy, axis=0)
    deltas = np.vstack([deltas, deltas[-1]])
    headings = np.arctan2(deltas[:, 1], deltas[:, 0]).astype(np.float32)

    zero_delta = np.linalg.norm(deltas, axis=1) < 1e-6
    if np.any(zero_delta):
        non_zero_idx = np.where(~zero_delta)[0]
        if non_zero_idx.size == 0:
            headings[:] = 0.0
        else:
            headings[zero_delta] = headings[non_zero_idx[0]]

    return headings


def _extract_tl_state(tl_info: dict[str, Any], current_index: int) -> tuple[int, bool]:
    # MetaDrive stores traffic-light states as a timeline string list
    # under state["object_state"].
    state = tl_info.get("state")
    if isinstance(state, dict):
        state_seq = state.get("object_state", [])
    elif isinstance(state, (list, tuple, np.ndarray)):
        state_seq = state
    else:
        state_seq = []

    if len(state_seq) == 0:
        return 0, False

    index = min(max(current_index, 0), len(state_seq) - 1)
    state_name = state_seq[index]
    if state_name is None:
        return 0, False
    state_id = TRAFFIC_LIGHT_STATE_TO_ID.get(str(state_name), 0)
    return state_id, True


def data_process_agent(
    scenario: dict[str, Any],
    max_num_objects: int = 64,
    current_index: int = 10,
    future_steps: int = 81,
    selected_agents: Any = None,
    remove_history: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # This function mirrors data_utils.data_process_agent output schema:
    # history [N, 11, 8], future [N, FUTURE_STEPS, 5], interested/type/id.
    tracks_raw = scenario.get("tracks", {})
    tracks = {_normalize_track_id(track_id): track for track_id, track in tracks_raw.items()}
    metadata = scenario.get("metadata", {})
    object_summary = metadata.get("object_summary", {})
    objects_of_interest = {str(track_id) for track_id in metadata.get("objects_of_interest", [])}

    if len(tracks) == 0:
        agents_history = np.zeros((max_num_objects, current_index + 1, 8), dtype=np.float32)
        agents_future = np.zeros((max_num_objects, future_steps, 5), dtype=np.float32)
        agents_interested = np.zeros((max_num_objects,), dtype=np.int32)
        agents_type = np.zeros((max_num_objects,), dtype=np.int32)
        agents_id = np.full((0,), -1, dtype=np.int32)
        return agents_history, agents_future, agents_interested, agents_type, agents_id

    sdc_id = str(metadata.get("sdc_id", next(iter(tracks.keys()))))
    if sdc_id not in tracks:
        sdc_id = next(iter(tracks.keys()))

    def current_xy(track: dict[str, Any]) -> np.ndarray:
        state = track.get("state", {})
        position = np.asarray(state.get("position", []), dtype=np.float32)
        if position.ndim != 2 or position.shape[0] == 0 or position.shape[1] < 2:
            return np.zeros((2,), dtype=np.float32)
        idx = min(current_index, position.shape[0] - 1)
        return position[idx, :2]

    if selected_agents is None:
        # Match original extractor behavior: keep nearest agents to SDC.
        agent_ids_all = list(tracks.keys())
        sdc_position = current_xy(tracks[sdc_id])
        agents_positions = np.asarray([current_xy(tracks[track_id]) for track_id in agent_ids_all], dtype=np.float32)
        distances = np.linalg.norm(agents_positions - sdc_position[None, :], axis=-1)
        selected_idx = np.argsort(distances)[:max_num_objects]
        selected_agent_ids = [agent_ids_all[idx] for idx in selected_idx]
    else:
        selected_agent_ids = [str(track_id) for track_id in selected_agents if str(track_id) in tracks]

    selected_agent_ids = selected_agent_ids[:max_num_objects]

    agents_history = np.zeros((max_num_objects, current_index + 1, 8), dtype=np.float32)
    agents_future = np.zeros((max_num_objects, future_steps, 5), dtype=np.float32)
    agents_interested = np.zeros((max_num_objects,), dtype=np.int32)
    agents_type = np.zeros((max_num_objects,), dtype=np.int32)

    for i, agent_id in enumerate(selected_agent_ids):
        track = tracks[agent_id]
        state = track.get("state", {})

        position_raw = np.asarray(state.get("position", []), dtype=np.float32)
        if position_raw.ndim != 2 or position_raw.shape[0] == 0 or position_raw.shape[1] < 2:
            continue

        track_len = position_raw.shape[0]
        position = _pad_or_trim_2d(position_raw, track_len, 3, np.float32, 0.0)
        heading = _pad_or_trim_1d(state.get("heading", []), track_len, np.float32, 0.0)
        velocity = _pad_or_trim_2d(state.get("velocity", []), track_len, 2, np.float32, 0.0)
        length = _pad_or_trim_1d(state.get("length", []), track_len, np.float32, 0.0)
        width = _pad_or_trim_1d(state.get("width", []), track_len, np.float32, 0.0)
        height = _pad_or_trim_1d(state.get("height", []), track_len, np.float32, 0.0)
        valid = _pad_or_trim_1d(state.get("valid", []), track_len, np.bool_, True)

        if current_index >= track_len or not bool(valid[current_index]):
            agents_interested[i] = 0
            continue

        track_summary = object_summary.get(agent_id, {})
        agent_type_raw = track.get("type", track_summary.get("type", "UNKNOWN"))
        agents_type[i] = _agent_type_to_id(agent_type_raw)

        if agent_id == sdc_id or agent_id in objects_of_interest:
            agents_interested[i] = 10
        else:
            agents_interested[i] = 1

        full_history = np.column_stack(
            [
                position[:, 0],
                position[:, 1],
                heading,
                velocity[:, 0],
                velocity[:, 1],
                length,
                width,
                height,
            ]
        ).astype(np.float32)

        hist_len = min(current_index + 1, track_len)
        agents_history[i, :hist_len] = full_history[:hist_len]
        invalid_hist_idx = np.where(~valid[:hist_len])[0]
        if invalid_hist_idx.size > 0:
            agents_history[i, invalid_hist_idx, :] = 0.0

        full_future = np.column_stack(
            [
                position[:, 0],
                position[:, 1],
                heading,
                velocity[:, 0],
                velocity[:, 1],
            ]
        ).astype(np.float32)
        future_start = current_index
        future_end = min(current_index + future_steps, track_len)
        if future_end > future_start:
            # Keep fixed horizon for stable batching; remaining steps are zero.
            local_future = full_future[future_start:future_end]
            local_valid = valid[future_start:future_end]
            agents_future[i, : local_future.shape[0]] = local_future
            invalid_future_idx = np.where(~local_valid)[0]
            if invalid_future_idx.size > 0:
                agents_future[i, invalid_future_idx, :] = 0.0

    if remove_history:
        agents_history[:, :-1] = 0

    agents_id = np.asarray([_safe_int(track_id, -1) for track_id in selected_agent_ids], dtype=np.int32)

    return agents_history, agents_future, agents_interested, agents_type, agents_id


def data_process_traffic_light(
    scenario: dict[str, Any],
    current_index: int = 10,
    max_traffic_lights: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Build fixed-size [MAX_TRAFFIC_LIGHTS, 3] tensor: [x, y, state_id].
    dynamic_states = scenario.get("dynamic_map_states", {})
    lane_ids: list[int] = []
    state_ids: list[int] = []
    points: list[np.ndarray] = []

    sorted_items = sorted(dynamic_states.items(), key=lambda item: _safe_int(item[0], 0))

    for tl_key, tl_info in sorted_items:
        if not isinstance(tl_info, dict):
            continue

        lane = tl_info.get("lane", tl_key)
        if isinstance(lane, (list, tuple, np.ndarray)):
            lane = lane[0] if len(lane) > 0 else -1
        lane_id = _safe_int(lane, _safe_int(tl_key, -1))

        state_id, is_valid = _extract_tl_state(tl_info, current_index)
        stop_point = np.asarray(tl_info.get("stop_point", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(-1)
        point = np.zeros((3,), dtype=np.float32)
        if stop_point.shape[0] >= 2 and is_valid:
            point[:2] = stop_point[:2]
            point[2] = float(state_id)

        lane_ids.append(lane_id)
        state_ids.append(state_id if is_valid else 0)
        points.append(point)

    if len(points) == 0:
        # Keep deterministic shape even when no traffic-light data exists.
        traffic_light_points = np.zeros((max_traffic_lights, 3), dtype=np.float32)
        traffic_lane_ids = np.full((max_traffic_lights,), -1, dtype=np.int32)
        traffic_light_states = np.zeros((max_traffic_lights,), dtype=np.int32)
        return traffic_light_points, traffic_lane_ids, traffic_light_states

    traffic_light_points_raw = np.asarray(points, dtype=np.float32)
    traffic_lane_ids_raw = np.asarray(lane_ids, dtype=np.int32)
    traffic_light_states_raw = np.asarray(state_ids, dtype=np.int32)

    if traffic_light_points_raw.shape[0] >= max_traffic_lights:
        traffic_light_points = traffic_light_points_raw[:max_traffic_lights]
        traffic_lane_ids = traffic_lane_ids_raw[:max_traffic_lights]
        traffic_light_states = traffic_light_states_raw[:max_traffic_lights]
    else:
        pad_n = max_traffic_lights - traffic_light_points_raw.shape[0]
        traffic_light_points = np.pad(
            traffic_light_points_raw,
            ((0, pad_n), (0, 0)),
            mode="constant",
            constant_values=0.0,
        )
        traffic_lane_ids = np.pad(
            traffic_lane_ids_raw,
            (0, pad_n),
            mode="constant",
            constant_values=-1,
        )
        traffic_light_states = np.pad(
            traffic_light_states_raw,
            (0, pad_n),
            mode="constant",
            constant_values=0,
        )

    return traffic_light_points.astype(np.float32), traffic_lane_ids.astype(np.int32), traffic_light_states.astype(np.int32)


def data_process_map(
    scenario: dict[str, Any],
    agents_history: np.ndarray,
    agents_interested: np.ndarray,
    traffic_lane_ids: np.ndarray,
    traffic_light_states: np.ndarray,
    max_polylines: int = 256,
    num_points_polyline: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    # Build fixed-size lane polyline tensor [MAX_POLYLINES, NUM_POINTS, 5]:
    # [x, y, heading, tl_state, map_type].
    map_features = scenario.get("map_features", {})
    feature_records: list[dict[str, Any]] = []
    centroids: list[np.ndarray] = []

    for feature_id_raw, feature in map_features.items():
        if not isinstance(feature, dict):
            continue
        points = _extract_map_points(feature)
        if points.shape[0] == 0:
            continue

        type_name = str(feature.get("type", "UNKNOWN")).upper()
        type_id = MAP_TYPE_TO_ID.get(type_name, 0)
        feature_id = _safe_int(feature_id_raw, -1)

        feature_records.append(
            {
                "feature_id": feature_id,
                "type_id": type_id,
                "points": points,
            }
        )
        centroids.append(points[:, :2].mean(axis=0))

    if len(feature_records) == 0:
        polylines = np.zeros((max_polylines, num_points_polyline, 5), dtype=np.float32)
        polylines_valid = np.zeros((max_polylines,), dtype=np.int32)
        return polylines, polylines_valid

    centroids_array = np.asarray(centroids, dtype=np.float32)
    valid_agent_positions = agents_history[agents_interested > 0, -1, :2]
    if valid_agent_positions.shape[0] == 0:
        valid_agent_positions = np.zeros((1, 2), dtype=np.float32)

    distances = np.linalg.norm(
        centroids_array[:, None, :] - valid_agent_positions[None, :, :],
        axis=-1,
    )
    # Keep nearest map features around active agents.
    min_distances = np.min(distances, axis=1)
    selected_indices = np.argsort(min_distances)[:max_polylines]

    lane_state_lookup = {}
    for lane_id, light_state in zip(traffic_lane_ids.tolist(), traffic_light_states.tolist()):
        if lane_id >= 0:
            lane_state_lookup[lane_id] = light_state

    selected_polylines: list[np.ndarray] = []
    for idx in selected_indices:
        record = feature_records[idx]
        points = record["points"]
        xy = points[:, :2].astype(np.float32)
        heading = _compute_headings(xy)

        traffic_state = float(lane_state_lookup.get(record["feature_id"], 0))
        map_type = float(record["type_id"])

        polyline = np.concatenate(
            [
                xy,
                heading[:, None],
                np.full((xy.shape[0], 1), traffic_state, dtype=np.float32),
                np.full((xy.shape[0], 1), map_type, dtype=np.float32),
            ],
            axis=1,
        )
        sampled_index = np.linspace(0, polyline.shape[0] - 1, num_points_polyline, dtype=np.int32)
        # Uniform sampling to align with Waymax preprocessing shape.
        sampled_polyline = polyline[sampled_index]
        selected_polylines.append(sampled_polyline)

    if len(selected_polylines) == 0:
        polylines = np.zeros((max_polylines, num_points_polyline, 5), dtype=np.float32)
        polylines_valid = np.zeros((max_polylines,), dtype=np.int32)
        return polylines, polylines_valid

    polylines_raw = np.stack(selected_polylines, axis=0).astype(np.float32)
    polylines_valid_raw = np.ones((polylines_raw.shape[0],), dtype=np.int32)

    if polylines_raw.shape[0] >= max_polylines:
        polylines = polylines_raw[:max_polylines]
        polylines_valid = polylines_valid_raw[:max_polylines]
    else:
        pad_n = max_polylines - polylines_raw.shape[0]
        polylines = np.pad(
            polylines_raw,
            ((0, pad_n), (0, 0), (0, 0)),
            mode="constant",
            constant_values=0.0,
        )
        polylines_valid = np.pad(
            polylines_valid_raw,
            (0, pad_n),
            mode="constant",
            constant_values=0,
        )

    return polylines.astype(np.float32), polylines_valid.astype(np.int32)


def data_process_scenario(
    scenario: dict[str, Any],
    max_num_objects: int = 64,
    max_polylines: int = 256,
    max_traffic_lights: int = 16,
    current_index: int = 10,
    num_points_polyline: int = 30,
    future_steps: int = 81,
    selected_agents: Any = None,
    remove_history: bool = False,
) -> dict[str, np.ndarray]:
    # Unified scenario-level entrypoint. Output keys/shapes intentionally match
    # VBD/script/extract_data.py + vbd/data/data_utils.py conventions.
    agents_history, agents_future, agents_interested, agents_type, agents_id = data_process_agent(
        scenario,
        max_num_objects=max_num_objects,
        current_index=current_index,
        future_steps=future_steps,
        selected_agents=selected_agents,
        remove_history=remove_history,
    )

    traffic_light_points, traffic_lane_ids, traffic_light_states = data_process_traffic_light(
        scenario,
        current_index=current_index,
        max_traffic_lights=max_traffic_lights,
    )

    polylines, polylines_valid = data_process_map(
        scenario,
        agents_history=agents_history,
        agents_interested=agents_interested,
        traffic_lane_ids=traffic_lane_ids,
        traffic_light_states=traffic_light_states,
        max_polylines=max_polylines,
        num_points_polyline=num_points_polyline,
    )

    relations = calculate_relations(agents_history, polylines, traffic_light_points)

    return {
        "agents_history": np.float32(agents_history),
        "agents_interested": np.int32(agents_interested),
        "agents_type": np.int32(agents_type),
        "agents_future": np.float32(agents_future),
        "traffic_light_points": np.float32(traffic_light_points),
        "polylines": np.float32(polylines),
        "polylines_valid": np.int32(polylines_valid),
        "relations": np.float32(relations),
        "agents_id": np.int32(agents_id),
    }


def _get_scenario_id(file_path: str, scenario: dict[str, Any]) -> str:
    # Prefer explicit scenario ids from payload; fallback to filename suffix.
    for key in ("id",):
        if key in scenario and scenario[key] is not None:
            return str(scenario[key])

    metadata = scenario.get("metadata", {})
    for key in ("scenario_id", "id"):
        if key in metadata and metadata[key] is not None:
            return str(metadata[key])

    filename = os.path.basename(file_path)
    if filename.endswith(".pkl") and filename.startswith("sd_waymo_v1.2_"):
        return filename[len("sd_waymo_v1.2_") : -4]
    return os.path.splitext(filename)[0]


def process_single_file(
    file_path: str,
    save_dir: str,
    save_raw: bool = False,
    only_raw: bool = False,
    max_num_objects: int = MAX_NUM_OBJECTS,
    max_polylines: int = MAX_POLYLINES,
    max_traffic_lights: int = MAX_TRAFFIC_LIGHTS,
    current_index: int = CURRENT_INDEX,
    num_points_polyline: int = NUM_POINTS_POLYLINE,
    future_steps: int = FUTURE_STEPS,
) -> int:
    with open(file_path, "rb") as f:
        scenario = pickle.load(f)

    if not isinstance(scenario, dict) or "tracks" not in scenario:
        return 0

    scenario_id = _get_scenario_id(file_path, scenario)
    scenario_filename = os.path.join(save_dir, f"scenario_{scenario_id}.pkl")

    if os.path.exists(scenario_filename):
        # Enable incremental reruns without rewriting finished samples.
        return 0

    if only_raw:
        data_dict = {"scenario_raw": scenario}
    else:
        data_dict = data_process_scenario(
            scenario,
            max_num_objects=max_num_objects,
            max_polylines=max_polylines,
            max_traffic_lights=max_traffic_lights,
            current_index=current_index,
            num_points_polyline=num_points_polyline,
            future_steps=future_steps,
        )
        if save_raw:
            data_dict["scenario_raw"] = scenario

    data_dict["scenario_id"] = scenario_id

    with open(scenario_filename, "wb") as f:
        pickle.dump(data_dict, f)
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/rainbow/Pycharm_Project/AutoDrive/Senario_Gen/data/part_0000_0002",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/home/rainbow/Pycharm_Project/AutoDrive/Senario_Gen/data/VBD",
    )
    parser.add_argument("--save_raw", action="store_true")
    parser.add_argument("--only_raw", action="store_true")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--max_num_objects", type=int, default=MAX_NUM_OBJECTS)
    parser.add_argument("--max_polylines", type=int, default=MAX_POLYLINES)
    parser.add_argument("--max_traffic_lights", type=int, default=MAX_TRAFFIC_LIGHTS)
    parser.add_argument("--current_index", type=int, default=CURRENT_INDEX)
    parser.add_argument("--num_points_polyline", type=int, default=NUM_POINTS_POLYLINE)
    parser.add_argument("--future_steps", type=int, default=FUTURE_STEPS)
    parser.add_argument("--max_scenarios", type=int, default=-1)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    if args.only_raw:
        output_dir = os.path.join(args.save_dir, "extracted")
    else:
        output_dir = os.path.join(args.save_dir, "processed")
    os.makedirs(output_dir, exist_ok=True)

    all_files = glob.glob(os.path.join(args.data_dir, "**", "*.pkl"), recursive=True)
    data_files = [
        file_path
        for file_path in sorted(all_files)
        if os.path.basename(file_path) not in SKIP_FILENAMES
    ]

    if args.max_scenarios > 0:
        data_files = data_files[: args.max_scenarios]

    print(f"Processing data from {args.data_dir} and saving to {output_dir}")
    print(f"Found {len(data_files)} scenario files")

    process_fn = functools.partial(
        process_single_file,
        save_dir=output_dir,
        save_raw=args.save_raw,
        only_raw=args.only_raw,
        max_num_objects=args.max_num_objects,
        max_polylines=args.max_polylines,
        max_traffic_lights=args.max_traffic_lights,
        current_index=args.current_index,
        num_points_polyline=args.num_points_polyline,
        future_steps=args.future_steps,
    )

    results = process_map(process_fn, data_files, max_workers=args.num_workers, chunksize=1)
    created = int(np.sum(np.asarray(results, dtype=np.int32)))
    print(f"Completed. New files written: {created}")
