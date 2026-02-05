"""
Project: Tactix
File Created: 2026-02-02 12:13:36
Author: Xingnan Zhu
File Name: camera.py
Description:
    Implements camera motion tracking using Optical Flow (Lucas-Kanade).
    It stabilizes the pitch detection by tracking keypoints across frames,
    reducing jitter and handling frames where the detection model might fail.
    Includes a smoothing mechanism (Moving Average) for robust coordinate output.
"""

from typing import Optional, List
import cv2
import numpy as np
from collections import deque

class CameraSmoother:
    """
    Smooths coordinate points using Moving Average.
    """
    def __init__(self, window_size=5):
        self.window_size = window_size
        # Queue to store historical coordinate data
        self.history = deque(maxlen=window_size)

    def update(self, current_points: np.ndarray) -> np.ndarray:
        """
        Adds new data and returns the smoothed result.
        :param current_points: Coordinates of the current frame (N, 2)
        :return: Smoothed coordinates (N, 2)
        """
        self.history.append(current_points)
        
        # Calculate average
        # stack converts list of arrays into (window_size, N, 2)
        # mean(axis=0) averages over the time dimension -> (N, 2)
        smoothed = np.mean(np.stack(self.history), axis=0)
        return smoothed


class CameraTracker:
    def __init__(self, initial_keypoints: np.ndarray = None, smoothing_window=5):
        """
        Initializes the camera tracker.
        :param initial_keypoints: Initial keypoint coordinates (N, 2), if None, waits for first update to initialize
        :param smoothing_window: Size of the smoothing window
        """
        self.current_keypoints: Optional[np.ndarray] = None
        if initial_keypoints is not None:
            self.current_keypoints = initial_keypoints.astype(np.float32).reshape(-1, 1, 2)
            
        self.prev_gray: Optional[np.ndarray] = None
        
        # Optical Flow parameters (LK Optical Flow)
        self.lk_params = dict(
            winSize=(20, 20),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Initialize smoother
        self.smoother = CameraSmoother(window_size=smoothing_window)

    def reset(self, keypoints: np.ndarray, frame: np.ndarray):
        """
        Force resets tracking points (usually called when YOLO detects high-confidence keypoints).
        """
        self.current_keypoints = keypoints.astype(np.float32).reshape(-1, 1, 2)
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Reset smoother history to avoid old data dragging down new position accuracy
        self.smoother = CameraSmoother(window_size=self.smoother.window_size)
        self.smoother.update(keypoints) # Immediately push current value

    def update(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Takes the current frame, calculates new positions using Optical Flow, and returns smoothed results.
        Returns None if not initialized.
        """
        if self.current_keypoints is None:
            return None

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # If running update for the first time and no prev_gray (but points given in init)
        if self.prev_gray is None:
            self.prev_gray = frame_gray
            # Directly return initial values (processed by smoother)
            raw_points = self.current_keypoints.reshape(-1, 2)
            return self.smoother.update(raw_points)

        # === Core: Optical Flow Tracking ===
        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, frame_gray, self.current_keypoints, None, **self.lk_params
        )

        # Check tracking status
        # status: 1=Found, 0=Not Found
        if status is not None and np.all(status == 1):
            self.current_keypoints = new_points
            
            # Get raw predicted points (N, 2)
            raw_points = self.current_keypoints.reshape(-1, 2)
            
            # Apply smoothing
            smoothed_points = self.smoother.update(raw_points)
            
            self.prev_gray = frame_gray
            return smoothed_points
        else:
            # If lost (status contains 0), two strategies:
            # 1. Return None, letting external system know to re-detect with YOLO
            # 2. Temporarily keep previous frame position
            # Here we choose to return None, forcing system recalibration
            print("⚠️ Camera Tracker lost keypoints!")
            return None
