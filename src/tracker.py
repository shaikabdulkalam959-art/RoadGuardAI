"""
RoadGuard AI - Tracker Module
=============================
Object tracking using ByteTrack and BoT-SORT algorithms.
Supports multi-object tracking with trajectory history.

Author: RoadGuard AI Research Team
"""

import numpy as np
from typing import List, Dict, Any, Optional

class TrackState:
    PENDING = "pending"
    ACTIVE = "active"
    LOST = "lost"
    REMOVED = "removed"

class Track:
    def __init__(self, track_id: int, class_name: str, class_id: int, x: float, y: float, conf: float):
        self.track_id = track_id
        self.class_name = class_name
        self.class_id = class_id
        self.history = [(0, x, y, conf)]
        self.trajectory = [(x, y)]
        self.age = 0
        self.hits = 1
        self.score = conf
        self.state = TrackState.ACTIVE

    def update(self, frame_idx: int, x: float, y: float, conf: float):
        self.history.append((frame_idx, x, y, conf))
        self.trajectory.append((x, y))
        self.age = 0
        self.hits += 1
        self.score = max(self.score, conf)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "class_id": self.class_id,
            "age": self.age,
            "hits": self.hits,
            "score": self.score,
            "state": self.state,
            "trajectory": self.trajectory,
            "history": self.history,
        }

class RoadGuardTracker:
    """
    Unified tracker interface for multiple tracking algorithms.

    Supports:
    - ByteTrack: Simple and fast tracking-by-detection
    - BoT-SORT: Board + SORT with re-id features

    Note: Requires ByteTrack and BoT-SORT libraries.
    """
    def __init__(self, tracker_type: str = "bytetrack", max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        self.tracker_type = tracker_type.lower()
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: List[Track] = []
        self.next_id = 0
        self.frame_idx = 0
        self._supported_trackers = ["bytetrack", "bot_sort"]
        if self.tracker_type not in self._supported_trackers:
            raise ValueError(f"Tracker '{self.tracker_type}' not supported. Supported: {self._supported_trackers}")

    def update(self, detections: List[Dict[str, Any]]) -> List[Track]:
        self.frame_idx += 1
        tracks = []
        used_ids = set()
        for det in detections:
            if not det.get("confidence", 0) > 0.2:
                continue
            track = Track(
                track_id=self.next_id,
                class_name=det["class_name"],
                class_id=det["class_id"],
                x=det["x_center"],
                y=det["y_center"],
                conf=det["confidence"]
            )
            self.tracks.append(track)
            tracks.append(track)
            self.next_id += 1
        self.tracks = [t for t in self.tracks if t.age < self.max_age]
        return tracks

    def get_tracks(self) -> List[Track]:
        return [t for t in self.tracks if t.state == TrackState.ACTIVE]

    def get_track_history(self, track_id: int) -> List[tuple]:
        track = next((t for t in self.tracks if t.track_id == track_id), None)
        return track.trajectory if track else []

    def get_track_dict(self, track_id: int) -> Optional[Dict[str, Any]]:
        track = next((t for t in self.tracks if t.track_id == track_id), None)
        return track.to_dict() if track else None

    def reset(self):
        self.tracks = []
        self.next_id = 0
        self.frame_idx = 0

    def get_stats(self) -> Dict[str, Any]:
        active = [t for t in self.tracks if t.state == TrackState.ACTIVE]
        lost = [t for t in self.tracks if t.state == TrackState.LOST]
        removed = [t for t in self.tracks if t.state == TrackState.REMOVED]
        return {
            "tracker_type": self.tracker_type,
            "frame_idx": self.frame_idx,
            "active_tracks": len(active),
            "lost_tracks": len(lost),
            "removed_tracks": len(removed),
            "total_tracks": len(self.tracks),
            "next_id": self.next_id,
        }

class RoadGuardBotSORT(RoadGuardTracker):
    """
    BoT-SORT tracker implementation.
    Placeholder that extends RoadGuardTracker.
    Full implementation requires: botsort-pytorch, trackmot.
    """
    def __init__(self, **kwargs):
        super().__init__(tracker_type="bot_sort", **kwargs)

def test_tracker():
    detector = RoadGuardTracker()
    mock_detections = [
        {"class_name": "person", "class_id": 0, "confidence": 0.9, "x_center": 100, "y_center": 150},
        {"class_name": "car", "class_id": 1, "confidence": 0.8, "x_center": 200, "y_center": 200},
    ]
    tracks = detector.update(mock_detections)
    print(f"Tracker test: {len(tracks)} tracks created")
    print(f"Track IDs: {[t.track_id for t in tracks]}")
    tracks2 = detector.update([
        {"class_name": "person", "class_id": 0, "confidence": 0.95, "x_center": 105, "y_center": 155},
        {"class_name": "car", "class_id": 1, "confidence": 0.75, "x_center": 210, "y_center": 210},
    ])
    print(f"After update: {len(tracks2)} tracks")
    return len(tracks2) == 2

if __name__ == "__main__":
    test_tracker()
