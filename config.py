"""
RoadGuard AI - Project Configuration
====================================
Central configuration for the RoadGuard AI system.
Environment-aware defaults for CPU-only deployment.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Base paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR
SRC_DIR = BASE_DIR / "src"
TRAINING_DIR = BASE_DIR / "training"
PAGES_DIR = BASE_DIR / "pages"
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
OUTPUTS_DIR = BASE_DIR / "outputs"
VIDEOS_DIR = OUTPUTS_DIR / "videos"
REPORTS_DIR = OUTPUTS_DIR / "reports"
CHARTS_DIR = OUTPUTS_DIR / "charts"
ASSETS_DIR = BASE_DIR / "assets"
MODELS_DIR = BASE_DIR / "models"
TESTS_DIR = BASE_DIR / "tests"

# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------
ENVIRONMENT = os.getenv("ROADGUARD_ENV", "development")
CPU_ONLY = os.getenv("ROADGUARD_CPU_ONLY", "1") == "1"
USE_GPU = not CPU_ONLY and torch.cuda.is_available() if False else False  # CPU fallback enforced

# Force CPU mode on Windows laptops (no dedicated NVIDIA GPU in most cases)
DEVICE = "cpu"

# ---------------------------------------------------------------------------
# Model settings
# ---------------------------------------------------------------------------
YOLO_MODEL_NAME = os.getenv("ROADGUARD_YOLO_MODEL", "yolov8n.pt")
YOLO_IMAGE_SIZE = int(os.getenv("ROADGUARD_YOLO_IMG_SIZE", "640"))

# Temporal model configuration
TEMPORAL_SEQ_LENGTH = 16
TEMPORAL_HIDDEN_SIZE = 64
TEMPORAL_DROPOUT = 0.3
TEMPORAL_FC_SIZE = 32
TEMPORAL_INPUT_FEATURES = 12
TEMPORAL_TARGETS = ["collision_anticipate"]

# Risk classification thresholds (calibrated on validation data)
RISK_THRESHOLDS = {
    "normal": 0.33,
    "elevated": 0.66,
    "high": 0.85,
}

# Tracking configuration
TRACKER_ALGORITHMS = ["bytetrack", "bot_sort"]
DEFAULT_TRACKER = "bytetrack"

# ---------------------------------------------------------------------------
# Video / file handling
# ---------------------------------------------------------------------------
MAX_UPLOAD_SIZE_MB = 500
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov"}
ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "tiff"}

# Storage limits for analysis artifacts
MAX_STORAGE_MB = 2000

# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------
DATASETS = {
    "nexar": {
        "name": "Nexar Collision Prediction Dataset",
        "source": "https://www.kaggle.com/datasets/nropine/nexar-collision-prediction",
        "description": "Dashcam footages of near-miss and collision events",
        "required_fields": ["video_id", "target", "timestamp", "alert_timestamp"],
        "download_url": "https://www.kaggle.com/datasets/nropine/nexar-collision-prediction",
    },
    "tudat": {
        "name": "TU-DAT (Dortmund Accident Tracking)",
        "source": "https://www.tudat.tugraz.at/index.php?id=36398",
        "description": "Traffic accident and near-miss documentation",
        "required_fields": ["video_id", "target", "timestamp", "alert_timestamp"],
        "download_url": "https://www.tudat.tugraz.at",
    },
    "dada2000": {
        "name": "DADA-2000 (Daimler Accident Dataset)",
        "source": "https://www.vision.ee.ethz.ch/projects/2019/2019-Xia-DADA-2000/",
        "description": "Causal analysis of accidents in dashcam videos",
        "required_fields": ["video_id", "target", "timestamp", "alert_timestamp"],
        "download_url": "https://www.vision.ee.ethz.ch/projects/2019/2019-Xia-DADA-2000/",
    },
}

# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------
TRAINING_CONFIG = {
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 32,
    "epochs": 50,
    "early_stop_patience": 8,
    "optimizer": "adam",
    "loss": "binary_cross_entropy_with_logits",
}

# Reproducibility
SEED = 42
PYTHONHASHSEED = SEED

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR = OUTPUTS_DIR / "logs"

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_config():
    """Validate configuration at startup."""
    if not YOLO_MODEL_NAME.endswith((".pt", ".onnx", ".engine")):
        raise ValueError(f"Invalid YOLO model file: {YOLO_MODEL_NAME}")

    # Ensure required directories exist
    for d in [RAW_DIR, PROCESSED_DIR, ANNOTATIONS_DIR, VIDEOS_DIR, REPORTS_DIR, CHARTS_DIR, MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Seed for reproducibility
    python_seed(SEED)

    return True


def python_seed(seed: int):
    """Set Python's random seed for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    validate_config()
    print("RoadGuardAI config loaded")
    print(f"  Device: {DEVICE}")
    print(f"  Environment: {ENVIRONMENT}")

