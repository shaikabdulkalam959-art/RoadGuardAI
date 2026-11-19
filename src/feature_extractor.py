"""
RoadGuard AI - Temporal Feature Extractor
==========================================
Extracts 12 numerical features per frame for temporal sequence generation.

Features:
1. Vehicle count
2. Pedestrian count
3. Motorcycle count
4. Average detection confidence
5. Maximum normalized bounding-box area
6. Average object centre X
7. Average object centre Y
8. Average movement speed
9. Maximum movement speed
10. Nearest vehicle-pedestrian separation
11. Relative movement speed
12. Active tracking ID count

Author: RoadGuard AI Research Team
"""

import numpy as np
from typing import List, Dict, Any

FEATURE_NAMES = [
    "vehicle_count",
    "pedestrian_count",
    "motorcycle_count",
    "avg_confidence",
    "max_norm_area",
    "avg_center_x",
    "avg_center_y",
    "avg_speed",
    "max_speed",
    "nearest_vehicle_ped_dist",
    "rel_speed",
    "active_track_count",
]

def extract_features(
    detections: List[Dict[str, Any]],
    track_ids: Optional[List[int]] = None,
    prev_positions: Optional[Dict[int, tuple]] = None,
) -> np.ndarray:
    feat = np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    # 1. Vehicle count (car, bus, truck, bicycle)
    feat[0] = sum(1 for d in detections if d["class_name"] in ["car", "bus", "truck", "bicycle"])

    # 2. Pedestrian count
    feat[1] = sum(1 for d in detections if d["class_name"] == "person")

    # 3. Motorcycle count
    feat[2] = sum(1 for d in detections if d["class_name"] == "motorcycle")

    # 4. Average detection confidence
    if detections:
        feat[3] = np.mean([d["confidence"] for d in detections])
    else:
        feat[3] = 0.0

    # 5. Maximum normalized bounding-box area
    if detections:
        max_area = max(d["area"] for d in detections)
        feat[4] = float(max_area / (1920 * 1080))
    else:
        feat[4] = 0.0

    # 6. Average object centre X
    if detections:
        feat[5] = np.mean([d["x_center"] for d in detections])
    else:
        feat[5] = 0.0

    # 7. Average object centre Y
    if detections:
        feat[6] = np.mean([d["y_center"] for d in detections])
    else:
        feat[6] = 0.0

    # 8. Average movement speed
    if detections and prev_positions:
        speeds = []
        for d in detections:
            prev_pos = prev_positions.get(d.get("track_id"))
            if prev_pos:
                dx = d["x_center"] - prev_pos[0]
                dy = d["y_center"] - prev_pos[1]
                speeds.append(np.sqrt(dx*dx + dy*dy))
        if speeds:
            feat[7] = np.mean(speeds)
    else:
        feat[7] = 0.0

    # 9. Maximum movement speed
    if detections and prev_positions:
        speeds = []
        for d in detections:
            prev_pos = prev_positions.get(d.get("track_id"))
            if prev_pos:
                dx = d["x_center"] - prev_pos[0]
                dy = d["y_center"] - prev_pos[1]
                speeds.append(np.sqrt(dx*dx + dy*dy))
        if speeds:
            feat[8] = max(speeds)
    else:
        feat[8] = 0.0

    # 10. Nearest vehicle-pedestrian separation
    feat[9] = _compute_nearest_vehicle_ped_distance(detections)

    # 11. Relative movement speed
    if detections and prev_positions:
        rel_speeds = []
        for d in detections:
            prev_pos = prev_positions.get(d.get("track_id"))
            if prev_pos:
                dx = d["x_center"] - prev_pos[0]
                dy = d["y_center"] - prev_pos[1]
                rel_speeds.append(np.sqrt(dx*dx + dy*dy))
        if rel_speeds:
            feat[10] = np.mean(rel_speeds)
    else:
        feat[10] = 0.0

    # 12. Active tracking ID count
    feat[11] = len(track_ids) if track_ids else 0

    return feat


def _compute_nearest_vehicle_ped_distance(detections: List[Dict]) -> float:
    peds = [d for d in detections if d["class_name"] == "person"]
    vehicles = [d for d in detections if d["class_name"] in ["car", "bus", "truck", "bicycle", "motorcycle"]]
    if not peds or not vehicles:
        return 0.0
    min_dist = float('inf')
    for ped in peds:
        for veh in vehicles:
            dx = ped["x_center"] - veh["x_center"]
            dy = ped["y_center"] - veh["y_center"]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = min(min_dist, dist)
    return float(min_dist / (1920 * 1080))


def features_to_frame_dict(features: np.ndarray) -> Dict[str, Any]:
    return {name: float(val) for name, val in zip(FEATURE_NAMES, features)}


def test_features():
    detections = [
        {"class_name": "person", "class_id": 0, "confidence": 0.9, "x_center": 300, "y_center": 200, "area": 10000},
        {"class_name": "car", "class_id": 1, "confidence": 0.8, "x_center": 500, "y_center": 300, "area": 20000},
        {"class_name": "motorcycle", "class_id": 2, "confidence": 0.7, "x_center": 700, "y_center": 150, "area": 8000},
    ]
    features = extract_features(detections)
    print(f"Features: {features}")
    print(f"Feature names: {FEATURE_NAMES}")
    assert features[0] == 1
    assert features[1] == 1
    assert features[2] == 1
    return True

if __name__ == "__main__":
    test_features()
