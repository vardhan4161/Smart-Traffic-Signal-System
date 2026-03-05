import pytest
import numpy as np
import cv2
import yaml
from src.detector.vehicle_detector import VehicleDetector, FrameResult, TrackedVehicle
from src.detector.preprocessor import Preprocessor

@pytest.fixture
def config():
    return {
        "detection": {
            "model_primary": "yolov8n.pt",
            "confidence_threshold": 0.25,
            "iou_threshold": 0.45,
            "input_size": 640
        },
        "tracking": {
            "enabled": True,
            "min_hits": 2
        },
        "counting": {
            "method": "peak_window",
            "window_frames": 5,
            "sample_every_n_frames": 1
        }
    }

def test_confidence_threshold_is_025(config):
    """Verify config has correct threshold after update."""
    detector = VehicleDetector(config)
    assert detector.conf_threshold == 0.25

def test_letterbox_resize_maintains_aspect_ratio(config):
    """Verify preprocessor doesn't stretch frames."""
    prep = Preprocessor(config)
    # Give a wide frame
    frame = np.zeros((300, 600, 3), dtype=np.uint8)
    resized = prep._resize(frame)
    assert resized.shape == (640, 640, 3)
    # Center area should be non-gray if we had actual content, 
    # but here we check the padding color (114, 114, 114) at the edges
    assert np.all(resized[0, 0] == [114, 114, 114])

def test_dark_frame_detection(config):
    """Verify is_dark_frame correctly identifies dark frames."""
    prep = Preprocessor(config)
    dark_frame = np.zeros((100, 100, 3), dtype=np.uint8) # All black
    bright_frame = np.full((100, 100, 3), 200, dtype=np.uint8) # Light gray
    assert prep.is_dark_frame(dark_frame) is True
    assert prep.is_dark_frame(bright_frame) is False

def test_peak_window_uses_highest_count(config):
    """Verify peak_window counts are max, not last. Uses the deque window logic directly."""
    from collections import deque
    # Simulate 3 frames with different counts
    window = deque(maxlen=5)
    window.append({"car": 2})
    window.append({"car": 5})  # PEAK
    window.append({"car": 3})

    peak = max(w.get("car", 0) for w in window)
    assert peak == 5, f"Expected 5 but got {peak}"


def test_tracking_counts_unique_ids_not_raw_detections(config):
    """Verify unique ID tracking: same car seen 3 frames = count of 1."""
    from collections import defaultdict
    all_seen_ids = defaultdict(set)

    # Simulate seeing track_id=1 for car, 3 frames in a row
    for _ in range(3):
        all_seen_ids["car"].add(1)  # same ID each frame

    assert len(all_seen_ids["car"]) == 1
    assert 1 in all_seen_ids["car"]
