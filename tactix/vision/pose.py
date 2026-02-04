"""
Project: Tactix
File Created: 2026-02-02 16:19:19
Author: Xingnan Zhu
File Name: pose.py
Description: xxx...
"""

from ultralytics import YOLO
import numpy as np
from typing import Tuple, Optional

class PitchEstimator:
    def __init__(self, model_path: str, device: str = 'mps'):
        print(f"🏟️ Loading Pitch Model: {model_path}...")
        self.model = YOLO(model_path)
        self.device = device

    def predict(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        返回: (keypoints_xy, confidences)
        xy shape: (27, 2)
        conf shape: (27,)
        """
        # 运行推理 (verbose=False 不打印废话)
        results = self.model(frame, device=self.device, verbose=False)[0]
        
        if results.keypoints is not None:
            # data shape: (1, 27, 3) -> [x, y, conf]
            kpts = results.keypoints.data[0].cpu().numpy()
            xy = kpts[:, :2]
            conf = kpts[:, 2]
            return xy, conf
        
        return None, None