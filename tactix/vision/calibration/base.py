"""
Project: Tactix
File Created: 2026-02-05 17:41:59
Author: Xingnan Zhu
File Name: base.py
Description:
    Defines the abstract base class for all pitch calibration estimators.
    Ensures a consistent interface for AI, Manual, and Hybrid approaches.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np

class BasePitchEstimator(ABC):
    """
    Abstract base class for pitch keypoint estimation.
    """
    
    @abstractmethod
    def predict(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Predict pitch keypoints for the given frame.
        
        Args:
            frame: The current video frame (BGR).
            
        Returns:
            Tuple containing:
            - keypoints_xy: np.ndarray of shape (27, 2) containing (x, y) coordinates.
            - confidences: np.ndarray of shape (27,) containing confidence scores (0.0 - 1.0).
            
            Returns (None, None) if estimation fails.
        """
        pass
