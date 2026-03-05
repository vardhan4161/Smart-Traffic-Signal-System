import cv2
import numpy as np
from loguru import logger

class Preprocessor:
    """Handles image enhancement and preprocessing for vehicle detection."""
    
    def __init__(self, config: dict):
        """Load CLAHE settings from config."""
        self.enable_clahe = config.get("preprocessing", {}).get("enable_clahe", True)
        self.clip_limit = config.get("preprocessing", {}).get("clahe_clip_limit", 2.0)
        self.tile_size = tuple(config.get("preprocessing", {}).get("clahe_tile_size", [8, 8]))
        self.target_size = tuple(config.get("preprocessing", {}).get("target_frame_size", [640, 640]))
        
        if self.enable_clahe:
            self.clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_size)

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE and resize frame."""
        try:
            if self.enable_clahe:
                # Convert to LAB color space
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                
                # Apply CLAHE to L channel
                l_enhanced = self.clahe.apply(l)
                
                # Merge channels and convert back to BGR
                lab_enhanced = cv2.merge((l_enhanced, a, b))
                frame = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
            
            # Resize to target frame size
            frame = cv2.resize(frame, self.target_size)
            return frame
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            return frame

    def enhance_night(self, frame: np.ndarray) -> np.ndarray:
        """Detect if frame is dark and apply enhancements."""
        try:
            # Calculate mean brightness
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            v_channel = hsv[:, :, 2]
            mean_brightness = np.mean(v_channel)
            
            if mean_brightness < 80:
                logger.info(f"Low brightness detected ({mean_brightness:.2f}). Applying night enhancement.")
                # Gamma correction (gamma=0.5)
                inv_gamma = 1.0 / 0.5
                table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                frame = cv2.LUT(frame, table)
                
            # Always apply CLAHE if enabled after gamma or if not dark
            return self.preprocess(frame)
        except Exception as e:
            logger.error(f"Night enhancement error: {e}")
            return frame
