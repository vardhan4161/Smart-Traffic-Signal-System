import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from loguru import logger

class FrameAnnotator:
    """Draws bounding boxes and HUD overlays on video frames."""
    
    def __init__(self, config: dict):
        """Load visualization settings from config."""
        viz_config = config.get("visualization", {})
        self.font_scale = viz_config.get("font_scale", 0.6)
        self.thickness = viz_config.get("box_thickness", 2)
        self.colors = {
            "car": (0, 255, 0),        # Green
            "motorcycle": (255, 255, 0), # Cyan/Yellowish
            "bus": (0, 0, 255),        # Red
            "truck": (0, 165, 255),    # Orange
            "unknown": (255, 255, 255)  # White
        }

    def annotate_frame(self, frame: np.ndarray, detections: List[Dict[str, Any]], signal_info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Draw bounding boxes and optional HUD info on the frame."""
        try:
            # Draw Bounding Boxes
            for det in detections:
                x1, y1, x2, y2 = map(int, det["bbox"])
                cls_name = det["class_name"]
                conf = det["confidence"]
                color = self.colors.get(cls_name, self.colors["unknown"])
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.thickness)
                label = f"{cls_name} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, color, 1)

            # Draw HUD
            if signal_info:
                overlay = frame.copy()
                cv2.rectangle(overlay, (10, 10), (280, 150), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                
                y_offset = 35
                cv2.putText(frame, f"Lane: {signal_info.get('lane_name', 'N/A')}", (20, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                y_offset += 30
                cv2.putText(frame, f"Density: {signal_info.get('density', 0.0):.1f} ({signal_info.get('level', 'LOW')})", 
                            (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                
                y_offset += 30
                cv2.putText(frame, f"Green Time: {signal_info.get('green_time', 0.0)}s", 
                            (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                y_offset += 30
                phase = signal_info.get("phase", "RED")
                phase_color = (0, 255, 0) if phase == "GREEN" else (0, 255, 255) if phase == "YELLOW" else (0, 0, 255)
                cv2.putText(frame, f"Phase: {phase}", (20, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, phase_color, 2)

            return frame
        except Exception as e:
            logger.error(f"Error annotating frame: {e}")
            return frame

    def add_comparison_overlay(self, frame: np.ndarray, adaptive_time: float, fixed_time: float) -> np.ndarray:
        """Add a comparison bar at the bottom of the frame."""
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, h - 40), (w, h), (30, 30, 30), -1)
        
        text = f"Adaptive: {adaptive_time}s | Fixed: {fixed_time}s"
        improvement = ((fixed_time - adaptive_time) / fixed_time * 100) if fixed_time > 0 else 0
        text += f" ( {improvement:+.1f}% Efficiency )"
        
        color = (0, 255, 0) if adaptive_time < fixed_time else (0, 0, 255)
        cv2.putText(frame, text, (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        return frame

    def save_annotated_frame(self, frame: np.ndarray, filename: str):
        """Save frame to the outputs/annotated directory."""
        try:
            cv2.imwrite(filename, frame)
        except Exception as e:
            logger.error(f"Failed to save annotated frame {filename}: {e}")
