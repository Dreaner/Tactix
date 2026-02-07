"""
Project: Tactix
File Created: 2026-02-05 17:46:15
Author: Xingnan Zhu
File Name: manual_estimator.py
Description:
    A manual pitch estimator that uses a set of initial keypoints provided by the user.
    It uses Optical Flow to track these keypoints across subsequent frames,
    providing a stable calibration when the automatic model fails.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
from tactix.vision.calibration.base import BasePitchEstimator

class ManualPitchEstimator(BasePitchEstimator):
    def __init__(self, initial_points: List[Tuple[int, int, int]]):
        """
        :param initial_points: List of (x, y, keypoint_index)
        """
        print(f"🔧 Initializing Manual Pitch Estimator with {len(initial_points)} points...")
        self.initial_points_data = initial_points
        
        # Internal state
        self.prev_gray = None
        self.current_points = None # shape: (N, 1, 2) for cv2.calcOpticalFlow
        self.point_indices = []    # Records keypoint_index for each point
        
        # Extract initial coordinates and indices
        pts = []
        for x, y, idx in initial_points:
            pts.append([float(x), float(y)])
            self.point_indices.append(idx)
            
        if pts:
            self.current_points = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)

        # Optical Flow parameters
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

    def predict(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Returns datasets conforming to PitchEstimator interface: (xy, conf)
        xy shape: (27, 2)
        conf shape: (27, )
        """
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. First frame: Initialization
        if self.prev_gray is None:
            self.prev_gray = frame_gray
            return self._format_output()

        # 2. Subsequent frames: Optical Flow tracking
        if self.current_points is not None and len(self.current_points) > 0:
            new_points, status, error = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, frame_gray, self.current_points, None, **self.lk_params
            )
            
            # Filter successfully tracked points
            good_new = []
            good_indices = []
            
            if status is not None:
                for i, (new, st) in enumerate(zip(new_points, status)):
                    if st == 1: # Tracking successful
                        good_new.append(new)
                        good_indices.append(self.point_indices[i])
            
            # Update state
            if len(good_new) > 0:
                self.current_points = np.array(good_new, dtype=np.float32).reshape(-1, 1, 2)
                self.point_indices = good_indices
            else:
                # If all lost, keep previous frame (or return empty)
                print("⚠️ Manual Estimator lost all points!")
                pass

        self.prev_gray = frame_gray
        return self._format_output()

    def _format_output(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fills currently tracked points into standard (27, 2) array
        """
        # 27 because our V4 standard defines 27 points
        output_xy = np.zeros((27, 2), dtype=float)
        output_conf = np.zeros(27, dtype=float)
        
        if self.current_points is not None:
            # flatten points: (N, 1, 2) -> (N, 2)
            flat_points = self.current_points.reshape(-1, 2)
            
            for i, (x, y) in enumerate(flat_points):
                idx = self.point_indices[i]
                if idx < 27:
                    output_xy[idx] = [x, y]
                    output_conf[idx] = 1.0 # Manually calibrated points have full confidence
                    
        return output_xy, output_conf
