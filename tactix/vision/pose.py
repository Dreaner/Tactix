"""
Project: Tactix
File Created: 2026-02-02 16:19:19
Author: Xingnan Zhu
File Name: pose.py
Description: xxx...
"""

from ultralytics import YOLO
import numpy as np
from typing import Tuple, Optional, List


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
    
# === [新增] 假的 AI (测试用) ===
class MockPitchEstimator:
    def __init__(self, mock_points: List[Tuple[int, int, int]]):
        print(f"⚠️ Warning: Using Mock Pitch Estimator (Fixed Coordinates)")
        self.mock_points = mock_points
        
        # 构造一个假的输出数组 (27个点，全是0)
        # 27 是因为我们的 V4 标准定义了 27 个点
        self.dummy_xy = np.zeros((27, 2), dtype=float)
        self.dummy_conf = np.zeros(27, dtype=float)
        
        # 把那 4 个固定点填进去
        for x, y, idx in mock_points:
            if idx < 27:
                self.dummy_xy[idx] = [x, y]
                self.dummy_conf[idx] = 1.0 # 置信度拉满

    def predict(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # 无论给什么图片，我都返回那 4 个点
        return self.dummy_xy, self.dummy_conf