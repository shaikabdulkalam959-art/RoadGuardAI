"""
RoadGuard AI - Object Detector Module
=====================================
Real-time vehicle and pedestrian detection using Ultralytics YOLO.
Supports CPU inference and can be swapped with ByteTrack/BoT-SORT later.

Author: RoadGuard AI Research Team
"""

import os
import time
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = "cpu"

# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
class RoadGuardDetector:
    """
    YOLO-based object detector for vehicles and pedestrians.

    Features:
    - Low-latency CPU inference (YOLOv8n)
    - Configurable confidence threshold per class
    - Class filtering for target classes
    - NMS post-processing
    - Lazy model loading for efficient resource usage

    Attributes
    ----------
    model : YOLO
        The loaded Ultralytics YOLO model instance.
    class_names : List[str]
        List of class names from the model.
    conf_thresh : float
        Detection confidence threshold (default 0.25).
    iou_thresh : float
        NMS IoU threshold (default 0.45).
    target_classes : set
        Classes the detector should report on.
    """

    # RoadGuard AI supported classes
    ROADGUARD_CLASSES = ["person", "car", "motorcycle", "bus", "truck", "bicycle"]

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        device: str = DEVICE,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        target_classes: Optional[List[str]] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.target_classes = set(target_classes or self.ROADGUARD_CLASSES)

        self.model: Optional[YOLO] = None
        self._model_loaded = False
        self._load_time = 0.0

    @property
    def classes(self) -> List[str]:
        if self.model is None or not self._model_loaded:
            return self.ROADGUARD_CLASSES
        return self.model.names

    def _load_model(self) -> None:
        """Load the YOLO model (lazy initialization)."""
        if self._model_loaded:
            return

        start = time.perf_counter()
        try:
            self.model = YOLO(self.model_name)
            self.model.to(self.device)
            self.model.conf = self.conf_thresh
            self.model.iou = self.iou_thresh
            self._model_loaded = True
            self._load_time = time.perf_counter() - start
        except Exception as e:
            raise RuntimeError(
                f"Failed to load YOLO model '{self.model_name}': {e}"
            ) from e

    def _filter_detections(self, result: Results) -> List[Dict[str, Any]]:
        """Filter and convert YOLO results to our internal format."""
        detections = []

        if result is None or result.boxes is None or len(result.boxes) == 0:
            return detections

        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = self.classes[cls_id] if cls_id < len(self.classes) else str(cls_id)

            # Only keep target classes
            if cls_name not in self.target_classes:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])

            detections.append({
                "class_name": cls_name,
                "class_id": cls_id,
                "confidence": conf,
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "x_center": float((x1 + x2) / 2),
                "y_center": float((y1 + y2) / 2),
                "width": float(x2 - x1),
                "height": float(y2 - y1),
                "area": float((x2 - x1) * (y2 - y1)),
            })

        return detections

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run detection on a single frame.

        Parameters
        ----------
        frame : np.ndarray
            Input BGR image.

        Returns
        -------
        List[Dict]
            List of detection dictionaries.
        """
        self._load_model()

        try:
            results = self.model(frame, verbose=False)
        except Exception as e:
            raise RuntimeError(f"YOLO inference failed: {e}") from e

        return self._filter_detections(results[0])

    def detect_batch(self, frames: List[np.ndarray], batch_size: int = 32) -> List[List[Dict]]:
        """Run detection on a batch of frames for improved throughput."""
        self._load_model()
        results = self.model(frames, verbose=False, batch=batch_size)
        return [self._filter_detections(r) for r in results]

    def get_load_time(self) -> float:
        """Return model load time in seconds."""
        return self._load_time


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
def test_detector():
    """Simple unit test for the detector."""
    detector = RoadGuardDetector()
    mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(mock_frame)

    print(f"Detector test: {len(detections)} detections on blank frame")
    print(f"  Classes available: {detector.classes}")
    return len(detections) == 0


if __name__ == "__main__":
    test_detector()
