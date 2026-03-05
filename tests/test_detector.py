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

def test_peak_window_uses_highest_count(config, mocker):
    """Verify peak_window method returns max seen, not last seen."""
    # Mocking cap.read to simulate frames
    mocker.patch("cv2.VideoCapture.isOpened", return_value=True)
    mocker.patch("cv2.VideoCapture.get", return_value=3) # 3 frames
    
    # Mock cap.read to return 3 different frames then False
    side_effects = [
        (True, np.zeros((100,100,3), dtype=np.uint8)),
        (True, np.zeros((100,100,3), dtype=np.uint8)),
        (True, np.zeros((100,100,3), dtype=np.uint8)),
        (False, None)
    ]
    mocker.patch("cv2.VideoCapture.read", side_effect=side_effects)
    
    detector = VehicleDetector(config)
    
    # Mock detect_frame to return different counts
    # Frame 1: 2 cars
    # Frame 2: 5 cars (PEAK)
    # Frame 3: 3 cars
    res1 = FrameResult(1, [], {}, {"car": 2}, 2)
    res2 = FrameResult(2, [], {}, {"car": 5}, 5)
    res3 = FrameResult(3, [], {}, {"car": 3}, 3)
    
    mocker.patch.object(detector, "detect_frame", side_effect=[res1, res2, res3])
    
    result = detector.process_video("fake.mp4")
    assert result["counts"]["car"] == 5

def test_tracking_counts_unique_ids_not_raw_detections(config, mocker):
    """Verify that the same track_id appearing across frames counts only once if tracking is used."""
    detector = VehicleDetector(config)
    
    # Simulate seeing track_id #1 for 3 frames
    for i in range(1, 4):
        # Mocking model.track results
        mock_box = mocker.Mock()
        mock_box.cls = [2] # car
        mock_box.conf = [0.9]
        mock_box.xyxy = [[10, 10, 50, 50]]
        mock_box.id = [1] # track_id 1
        
        mock_res = mocker.Mock()
        mock_res.boxes = [mock_box]
        
        mocker.patch.object(detector.model, "track", return_value=[mock_res])
        
        res = detector.detect_frame(np.zeros((640,640,3), dtype=np.uint8))
        
    # Even after 3 frames, if it's the same track_id, cumulative count for "car" should be 1
    assert len(detector.all_seen_ids["car"]) == 1
    assert 1 in detector.all_seen_ids["car"]
