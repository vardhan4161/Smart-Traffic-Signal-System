import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from ultralytics import YOLO
from loguru import logger
from tqdm import tqdm
from src.detector.preprocessor import Preprocessor

class VehicleDetector:
    """YOLOv8-based vehicle detector with image preprocessing."""
    
    def __init__(self, config: dict):
        """Initialize YOLOv8 model and preprocessor."""
        try:
            model_path = config.get("detection", {}).get("model", "yolov8n.pt")
            self.model = YOLO(model_path)
            self.conf_threshold = config.get("detection", {}).get("confidence_threshold", 0.45)
            self.iou_threshold = config.get("detection", {}).get("iou_threshold", 0.5)
            self.target_classes = config.get("detection", {}).get("target_classes", {})
            self.preprocessor = Preprocessor(config)
            logger.info(f"VehicleDetector initialized with {model_path}")
        except Exception as e:
            logger.error(f"Failed to initialize VehicleDetector: {e}")
            raise

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run YOLO inference on a single frame."""
        try:
            # Enhance/preprocess frame
            processed_frame = self.preprocessor.enhance_night(frame)
            
            # Inference
            results = self.model.predict(
                source=processed_frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    # Check if class is in target classes
                    class_name = None
                    for name, target_id in self.target_classes.items():
                        if cls_id == target_id:
                            class_name = name
                            break
                    
                    if class_name:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        detections.append({
                            "class_name": class_name,
                            "confidence": float(box.conf[0]),
                            "bbox": [x1, y1, x2, y2],
                            "center": [(x1 + x2) / 2, (y1 + y2) / 2]
                        })
            
            return detections
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []

    def count_by_type(self, detections: List[Dict[str, Any]]) -> Dict[str, int]:
        """Convert list of detections to counts per vehicle type."""
        counts = {name: 0 for name in self.target_classes.keys()}
        for det in detections:
            counts[det["class_name"]] += 1
        return counts

    def detect_video(self, video_path: str, lane_id: str = "Unknown") -> List[Dict[str, Any]]:
        """Process video and return detections per frame."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Cannot open video: {video_path}")
                return []
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            all_detections = []
            
            logger.info(f"Processing video {video_path} for lane {lane_id}")
            with tqdm(total=total_frames, desc=f"Lane {lane_id}") as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    detections = self.detect(frame)
                    all_detections.append(detections)
                    pbar.update(1)
            
            cap.release()
            return all_detections
        except Exception as e:
            logger.error(f"Video processing error: {e}")
            return []
