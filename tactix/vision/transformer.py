"""
Project: Tactix
File Created: 2026-02-02 23:22:57
Author: Xingnan Zhu
File Name: transformer.py
Description: xxx...
"""


import cv2
import numpy as np
from typing import List, Optional, Tuple

# 引入之前的定义
from tactix.core.types import PitchConfig, Player
from tactix.core.keypoints import YOLO_INDEX_MAP 
from tactix.core.geometry import WORLD_POINTS # 确保 geometry.py 里定义了 WORLD_POINTS (米)

class ViewTransformer:
    def __init__(self):
        self.homography_matrix = None
        
        # 预计算目标板的像素尺寸 (用于把米映射到小地图像素)
        self.scale_x = PitchConfig.PIXEL_WIDTH / PitchConfig.LENGTH
        self.scale_y = PitchConfig.PIXEL_HEIGHT / PitchConfig.WIDTH

    def update(self, keypoints: np.ndarray, confs: np.ndarray, threshold: float = 0.5) -> bool:
        """
        核心: 使用 RANSAC 自动计算单应性矩阵
        """
        if keypoints is None: return False

        src_pts = [] # 视频点 (Pixel)
        dst_pts = [] # 战术板点 (Pixel)

        for i, (x, y) in enumerate(keypoints):
            if confs[i] < threshold: continue
            
            # 查表: 它是哪个点?
            name = YOLO_INDEX_MAP.get(i)
            if name and name in WORLD_POINTS:
                # 1. 视频坐标
                src_pts.append([x, y])
                
                # 2. 世界坐标(米) -> 战术板坐标(像素)
                world_x, world_y = WORLD_POINTS[name]
                target_x = int(world_x * self.scale_x)
                target_y = int(world_y * self.scale_y)
                dst_pts.append([target_x, target_y])

        if len(src_pts) < 4:
            return False # 点不够，无法计算

        src_arr = np.array(src_pts).reshape(-1, 1, 2)
        dst_arr = np.array(dst_pts).reshape(-1, 1, 2)

        # RANSAC 计算矩阵
        H, _ = cv2.findHomography(src_arr, dst_arr, cv2.RANSAC, 5.0)
        
        if H is not None:
            self.homography_matrix = H
            return True
        return False

    def transform_point(self, xy: Tuple[float, float]) -> Optional[Tuple[int, int]]:
        if self.homography_matrix is None: return None
        point_arr = np.array([[[xy[0], xy[1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point_arr, self.homography_matrix)[0][0]
        return int(transformed[0]), int(transformed[1])

    def transform_players(self, players: List[Player]):
        """批量转换球员坐标"""
        for p in players:
            # 转换脚底锚点
            result = self.transform_point(p.anchor)
            if result:
                # 这是一个 Hack: 这里的 result 是小地图像素坐标
                # 我们暂时把它存到 pitch_position 里
                # (你需要确保 Player 类的 pitch_position 支持 int tuple 或者 Point)
                from tactix.core.types import Point
                p.pitch_position = Point(x=result[0], y=result[1])