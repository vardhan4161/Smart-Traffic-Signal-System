import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.detector.vehicle_detector import VehicleDetector

@pytest.fixture
def config():
    return {
        "detection": {
            "model": "yolov8n.pt",
            "confidence_threshold": 0.45,
            "iou_threshold": 0.5,
            "target_classes": {"car": 2, "bus": 5}
        },
        "preprocessing": {"enable_clahe": False, "target_frame_size": [640, 640]}
    }

def test_count_by_type_returns_correct_counts(config):
    # We don't need to mock YOLO for this unit test as it's pure logic
    detector = VehicleDetector(config)
    detections = [
        {"class_name": "car", "confidence": 0.9, "bbox": [0,0,10,10]},
        {"class_name": "car", "confidence": 0.8, "bbox": [10,10,20,20]},
        {"class_name": "bus", "confidence": 0.7, "bbox": [20,20,30,30]}
    ]
    counts = detector.count_by_type(detections)
    assert counts == {"car": 2, "bus": 1}

@patch("src.detector.vehicle_detector.YOLO")
def test_detect_filters_non_vehicle_classes(mock_yolo, config):
    # Mock results from YOLO
    mock_result = MagicMock()
    mock_box_car = MagicMock()
    mock_box_car.cls = [2] # car
    mock_box_car.conf = [0.9]
    mock_box_car.xyxy = [[0, 0, 10, 10]]
    
    mock_box_person = MagicMock()
    mock_box_person.cls = [0] # person (not in target_classes)
    mock_box_person.conf = [0.9]
    mock_box_person.xyxy = [[10, 10, 20, 20]]
    
    mock_result.boxes = [mock_box_car, mock_box_person]
    mock_yolo.return_value.predict.return_value = [mock_result]
    
    detector = VehicleDetector(config)
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    detections = detector.detect(frame)
    
    assert len(detections) == 1
    assert detections[0]["class_name"] == "car"
