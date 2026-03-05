import cv2
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO
from loguru import logger
import yaml
from src.detector.preprocessor import Preprocessor

@dataclass
class TrackedVehicle:
    """Represents a vehicle being tracked across frames."""
    track_id: int
    class_name: str
    confidence: float
    bbox: List[int]
    frames_seen: int = 1
    last_seen_frame: int = 0

@dataclass
class FrameResult:
    """Detection result for one frame."""
    frame_number: int
    detections: List[dict]
    tracked_vehicles: Dict[int, TrackedVehicle]
    unique_count_by_class: Dict[str, int]
    total_unique: int
    annotated_frame: Optional[np.ndarray] = None

class VehicleDetector:
    """
    YOLOv8-based vehicle detector with ByteTrack tracking.
    Counts UNIQUE vehicles using track IDs, not raw per-frame detections.
    """

    # COCO class IDs for vehicles
    VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    # Colors per class for bounding boxes (BGR)
    CLASS_COLORS = {
        "car":        (0, 255, 0),     # Green
        "motorcycle": (0, 255, 255),   # Cyan
        "bus":        (0, 0, 255),     # Red
        "truck":      (0, 165, 255),   # Orange
    }

    def __init__(self, config: dict):
        self.config = config
        det_cfg = config.get("detection", {})
        trk_cfg = config.get("tracking", {})
        cnt_cfg = config.get("counting", {})

        self.conf_threshold = det_cfg.get("confidence_threshold", 0.25)
        self.iou_threshold  = det_cfg.get("iou_threshold", 0.45)
        self.sample_every   = cnt_cfg.get("sample_every_n_frames", 3)
        self.window_frames  = cnt_cfg.get("window_frames", 30)
        self.min_hits       = trk_cfg.get("min_hits", 2)
        self.tracking_enabled = trk_cfg.get("enabled", True)

        self.preprocessor = Preprocessor(config)
        self.model = self._load_model(det_cfg)

        # Tracking state
        self.active_tracks: Dict[int, TrackedVehicle] = {}
        self.all_seen_ids: Dict[str, set] = defaultdict(set)
        self.frame_count = 0

        logger.info(f"VehicleDetector ready | conf={self.conf_threshold} | tracking={self.tracking_enabled}")

    def _load_model(self, det_cfg: dict) -> YOLO:
        """Load YOLOv8s with fallback to YOLOv8n."""
        primary = det_cfg.get("model_primary", "yolov8s.pt")
        fallback = det_cfg.get("model_fallback", "yolov8n.pt")
        try:
            model = YOLO(primary)
            logger.info(f"Loaded model: {primary}")
            return model
        except Exception as e:
            logger.warning(f"Could not load {primary}: {e}. Falling back to {fallback}")
            return YOLO(fallback)

    def reset_tracking(self):
        """Reset all tracking state. Call between different video analyses."""
        self.active_tracks.clear()
        self.all_seen_ids.clear()
        self.frame_count = 0

    def detect_frame(self, frame: np.ndarray, draw: bool = True) -> FrameResult:
        """
        Run detection + tracking on a single frame.
        Returns FrameResult with unique vehicle counts.
        """
        self.frame_count += 1

        # Preprocess
        if self.preprocessor.is_dark_frame(frame):
            proc_frame = self.preprocessor.night_enhance(frame)
        else:
            proc_frame = self.preprocessor.preprocess(frame)

        # Run YOLO with tracking
        if self.tracking_enabled:
            results = self.model.track(
                proc_frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=list(self.VEHICLE_CLASSES.keys()),
                tracker="bytetrack.yaml",
                persist=True,
                verbose=False
            )
        else:
            results = self.model.predict(
                proc_frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=list(self.VEHICLE_CLASSES.keys()),
                verbose=False
            )

        # Parse detections
        detections = []
        current_frame_tracks = {}

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                if cls_id not in self.VEHICLE_CLASSES:
                    continue

                class_name = self.VEHICLE_CLASSES[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # Get track ID if available
                track_id = None
                if self.tracking_enabled and box.id is not None:
                    track_id = int(box.id[0])
                    # Only count if track is stable (seen min_hits times)
                    if track_id not in self.active_tracks:
                        self.active_tracks[track_id] = TrackedVehicle(
                            track_id=track_id,
                            class_name=class_name,
                            confidence=conf,
                            bbox=[x1, y1, x2, y2],
                            last_seen_frame=self.frame_count
                        )
                    else:
                        tv = self.active_tracks[track_id]
                        tv.frames_seen += 1
                        tv.last_seen_frame = self.frame_count
                        tv.confidence = max(tv.confidence, conf)

                    # Register as confirmed if seen enough times
                    if self.active_tracks[track_id].frames_seen >= self.min_hits:
                        self.all_seen_ids[class_name].add(track_id)

                    current_frame_tracks[track_id] = self.active_tracks[track_id]

                detections.append({
                    "class_name": class_name,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                    "center": [cx, cy],
                    "track_id": track_id
                })

        # Build unique counts using cumulative track IDs
        unique_counts = {cls: len(ids) for cls, ids in self.all_seen_ids.items()}
        for cls in ["car", "motorcycle", "bus", "truck"]:
            unique_counts.setdefault(cls, 0)

        total_unique = sum(unique_counts.values())

        # Annotate frame if requested
        annotated = None
        if draw:
            annotated = self._annotate_frame(frame.copy(), detections, unique_counts)

        return FrameResult(
            frame_number=self.frame_count,
            detections=detections,
            tracked_vehicles=current_frame_tracks,
            unique_count_by_class=unique_counts,
            total_unique=total_unique,
            annotated_frame=annotated
        )

    def _annotate_frame(self, frame: np.ndarray,
                        detections: List[dict],
                        unique_counts: Dict[str, int]) -> np.ndarray:
        """Draw bounding boxes, labels, and HUD on frame."""
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cls = det["class_name"]
            conf = det["confidence"]
            tid = det.get("track_id")
            color = self.CLASS_COLORS.get(cls, (255, 255, 255))

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label: "Car #12  0.87"
            label = f"{cls.capitalize()}"
            if tid is not None:
                label += f" #{tid}"
            label += f"  {conf:.2f}"

            # Background for label
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

        # HUD — top left
        hud_lines = [
            f"FRAME: {self.frame_count}",
            f"DETECTED (this frame): {len(detections)}",
            f"UNIQUE TOTAL: {sum(unique_counts.values())}",
            f"  Cars: {unique_counts.get('car', 0)}",
            f"  Bikes: {unique_counts.get('motorcycle', 0)}",
            f"  Buses: {unique_counts.get('bus', 0)}",
            f"  Trucks: {unique_counts.get('truck', 0)}",
        ]
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 8), (220, 10 + len(hud_lines)*22), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        for i, line in enumerate(hud_lines):
            color = (0, 255, 0) if i == 0 else (200, 200, 200)
            cv2.putText(frame, line, (12, 26 + i*22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)
        return frame

    def process_video(self, video_path: str,
                      progress_callback=None) -> Dict:
        """
        Process an entire video file.
        Returns final unique vehicle counts using peak-window method.
        """
        self.reset_tracking()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_idx = 0

        # Sliding window for peak counting
        window = deque(maxlen=self.window_frames)
        peak_counts = defaultdict(int)

        logger.info(f"Processing {total_frames} frames from {video_path}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # Sample every N frames for speed
            if frame_idx % self.sample_every != 0:
                continue

            result = self.detect_frame(frame, draw=False)
            window.append(result.unique_count_by_class.copy())

            # Update peak counts from window
            for cls in ["car", "motorcycle", "bus", "truck"]:
                window_max = max((w.get(cls, 0) for w in window), default=0)
                if window_max > peak_counts[cls]:
                    peak_counts[cls] = window_max

            if progress_callback:
                progress_callback(frame_idx, total_frames)

        cap.release()

        # Final counts: use MAX of (peak_window, cumulative_unique_ids)
        final_counts = {}
        for cls in ["car", "motorcycle", "bus", "truck"]:
            cumulative = len(self.all_seen_ids.get(cls, set()))
            peak = peak_counts.get(cls, 0)
            # Use the higher of the two — catches both slow-moving and fast-moving traffic
            final_counts[cls] = max(cumulative, peak)

        total = sum(final_counts.values())
        logger.info(f"Final counts: {final_counts} | Total: {total}")

        return {
            "counts": final_counts,
            "total": total,
            "frames_processed": frame_idx // self.sample_every,
            "total_frames": total_frames,
            "fps": fps,
            "unique_track_ids": {cls: len(ids) for cls, ids in self.all_seen_ids.items()}
        }

    def detect_and_show(self, video_path: str):
        """
        Process video and display annotated output live in OpenCV window.
        Press Q to quit, SPACE to pause.
        """
        self.reset_tracking()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        delay = max(1, int(1000 / fps))
        paused = False

        logger.info("Press Q to quit, SPACE to pause/resume")

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                result = self.detect_frame(frame, draw=True)
                display = result.annotated_frame
                cv2.imshow("Smart Traffic Detection | Q=quit SPACE=pause", display)

            key = cv2.waitKey(delay) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused

        cap.release()
        cv2.destroyAllWindows()

        return {
            "counts": {cls: len(ids) for cls, ids in self.all_seen_ids.items()},
            "total": sum(len(ids) for ids in self.all_seen_ids.values())
        }

    # Helper for GUI compatibility
    def detect(self, frame: np.ndarray):
        """Alias for detect_frame for GUI compatibility."""
        res = self.detect_frame(frame, draw=False)
        return res.detections

    def annotate(self, frame: np.ndarray, detections):
        """Redundant but kept for GUI code compatibility."""
        # Convert all_seen_ids (dict of sets) to counts (dict of ints)
        unique_counts = {cls: len(ids) for cls, ids in self.all_seen_ids.items()}
        for cls in ["car", "motorcycle", "bus", "truck"]:
            unique_counts.setdefault(cls, 0)
        return self._annotate_frame(frame, detections, unique_counts)

    def count_vehicles(self, detections):
        """Redundant but kept for GUI code compatibility."""
        counts = {cls: 0 for cls in ["car", "motorcycle", "bus", "truck"]}
        for d in detections:
            counts[d["class_name"]] += 1
        return counts
