"""
Project: Tactix
File Created: 2026-02-05 17:42:12
Author: Xingnan Zhu
File Name: ai_estimator.py
Description:
    Implements the pitch keypoint estimation using a YOLO-Pose model.
    It detects 27 standard keypoints on the football pitch to establish
    the correspondence between the video frame and the 2D tactical board.
"""

from ultralytics import YOLO
import numpy as np
from typing import Tuple, Optional, List
from tactix.vision.calibration.base import BasePitchEstimator

class AIPitchEstimator(BasePitchEstimator):
    def __init__(self, model_path: str, device: str = 'mps'):
        print(f"🏟️ Loading AI Pitch Model: {model_path}...")
        self.model = YOLO(model_path)
        self.device = device

    def predict(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Returns: (keypoints_xy, confidences)
        xy shape: (27, 2)
        conf shape: (27, )
        """
        # Run inference (verbose=False suppresses logs)
        results = self.model(frame, device=self.device, verbose=False)[0]
        
        if results.keypoints is not None and len(results.keypoints.data) > 0:
            # Logic: Find the box with the largest area or highest confidence
            # Here we simply take the one with the highest box confidence
            best_idx = 0
            if results.boxes is not None:
                # Find index of max confidence
                best_idx = results.boxes.conf.argmax().item()

            # data shape: (1, 27, 3) -> [x, y, conf]
            kpts = results.keypoints.data[0].cpu().numpy()
            xy = kpts[:, :2]
            conf = kpts[:, 2]
            return xy, conf
        
        # If nothing detected, return None
        return None, None
    
class MockPitchEstimator(BasePitchEstimator):
    """
    A dummy estimator that returns fixed coordinates. Useful for testing without loading heavy models.
    """
    def __init__(self, mock_points: List[Tuple[int, int, int]]):
        print(f"⚠️ Warning: Using Mock Pitch Estimator (Fixed Coordinates)")
        self.mock_points = mock_points
        
        # Construct a dummy output array (27 points, all 0)
        # 27 because our V4 standard defines 27 points
        self.dummy_xy = np.zeros((27, 2), dtype=float)
        self.dummy_conf = np.zeros(27, dtype=float)
        
        # Fill in the 4 fixed points
        for x, y, idx in mock_points:
            if idx < 27:
                self.dummy_xy[idx] = [x, y]
                self.dummy_conf[idx] = 1.0 # Full confidence

    def predict(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Return the 4 points regardless of the input frame
        return self.dummy_xy, self.dummy_conf