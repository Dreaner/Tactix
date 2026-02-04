"""
Project: Tactix
File Created: 2026-02-02 12:13:36
Author: Xingnan Zhu
File Name: camera.py
Description: xxx...
"""

"""
Camera Movement Tracker based on Optical Flow
"""
from typing import Optional

import cv2
import numpy as np


class CameraTracker:
    def __init__(self, initial_keypoints: np.ndarray):
        """
        初始化相机跟踪器
        :param initial_keypoints: 第0帧的4个关键点坐标 (N, 2)
        """
        # 转换为 float32 并 reshape 为 (N, 1, 2) 以适配 OpenCV API
        self.current_keypoints = initial_keypoints.astype(np.float32).reshape(-1, 1, 2)
        
        self.prev_gray: Optional[np.ndarray] = None
        
        # 光流法参数 (LK Optical Flow)
        self.lk_params = dict(
            winSize=(20, 20),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

    def update(self, frame: np.ndarray) -> np.ndarray:
        """
        传入当前帧，返回更新后的关键点坐标
        """
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #如果是第一帧，只初始化，不计算光流
        if self.prev_gray is None:
            self.prev_gray = frame_gray
            return self.current_keypoints.reshape(-1, 2)

        # 计算光流
        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, frame_gray, self.current_keypoints, None, **self.lk_params
        )

        # 只有当所有点都追踪成功时才更新
        if status is not None and np.all(status == 1):
            self.current_keypoints = new_points
        else:
            # 如果跟丢了，保持上一帧的位置 (或者可以在这里抛出警告)
            pass

        # 更新历史帧
        self.prev_gray = frame_gray
        
        # 返回扁平化的坐标 (N, 2) 方便外部使用
        return self.current_keypoints.reshape(-1, 2)
