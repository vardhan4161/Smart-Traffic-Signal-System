import cv2
import numpy as np

class Preprocessor:
    def __init__(self, config: dict):
        self.target_size = config.get("detection", {}).get("input_size", 640)

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Full preprocessing pipeline for best detection accuracy."""
        frame = self._resize(frame)
        frame = self._enhance_contrast(frame)
        return frame

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        """Resize to 640x640 maintaining aspect ratio with letterboxing.
        CRITICAL: Do NOT stretch — use letterbox padding with gray (114,114,114).
        Stretching distorts vehicle shapes and kills detection accuracy."""
        h, w = frame.shape[:2]
        target = self.target_size
        scale = min(target/h, target/w)
        new_w, new_h = int(w*scale), int(h*scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # Pad to 640x640
        top = (target - new_h) // 2
        bottom = target - new_h - top
        left = (target - new_w) // 2
        right = target - new_w - left
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return padded

    def _enhance_contrast(self, frame: np.ndarray) -> np.ndarray:
        """CLAHE on L channel only — improves detection in low contrast / overcast lighting."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    def is_dark_frame(self, frame: np.ndarray) -> bool:
        """Detect if frame needs night enhancement."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return gray.mean() < 60

    def night_enhance(self, frame: np.ndarray) -> np.ndarray:
        """Gamma correction for dark footage before CLAHE."""
        gamma = 0.4
        inv_gamma = 1.0 / gamma
        table = np.array([(i/255.0)**inv_gamma * 255
                         for i in range(256)]).astype(np.uint8)
        brightened = cv2.LUT(frame, table)
        return self._enhance_contrast(brightened)
