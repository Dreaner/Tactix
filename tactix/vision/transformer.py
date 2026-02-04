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
from tactix.core.types import PitchConfig, Player
from tactix.core.keypoints import YOLO_INDEX_MAP 
from tactix.core.geometry import WORLD_POINTS

class ViewTransformer:
    def __init__(self):
        self.homography_matrix = None
        self.scale_x = PitchConfig.PIXEL_WIDTH / PitchConfig.LENGTH
        self.scale_y = PitchConfig.PIXEL_HEIGHT / PitchConfig.WIDTH

    def update(self, keypoints: np.ndarray, confs: np.ndarray, threshold: float = 0.5) -> bool:
        """
        尝试更新矩阵。
        返回: bool (当前是否有可用的矩阵，无论是新的还是旧的)
        """
        if keypoints is None: 
            # 如果没点，看看有没有老本可以吃
            return self.homography_matrix is not None

        src_pts = [] 
        dst_pts = [] 

        for i, (x, y) in enumerate(keypoints):
            if confs[i] < threshold: continue
            
            name = YOLO_INDEX_MAP.get(i)
            if name and name in WORLD_POINTS:
                src_pts.append([x, y])
                world_x, world_y = WORLD_POINTS[name]
                target_x = int(world_x * self.scale_x)
                target_y = int(world_y * self.scale_y)
                dst_pts.append([target_x, target_y])

        # 🔥 核心修改：如果点不够，不报错，不清空，直接沿用上一帧的矩阵
        if len(src_pts) < 4:
            return self.homography_matrix is not None

        src_arr = np.array(src_pts).reshape(-1, 1, 2)
        dst_arr = np.array(dst_pts).reshape(-1, 1, 2)

        # RANSAC 计算

        h, mask = cv2.findHomography(src_arr, dst_arr, cv2.RANSAC, 5.0)
        
        if h is not None:
             # 二次校验
             inliers = np.sum(mask)
             if inliers >= 4:
                 self.homography_matrix = h # 更新为新的
                 return True
        
        # 如果新算的不好，也继续用旧的
        return self.homography_matrix is not None

    def transform_point(self, xy: Tuple[float, float]) -> Optional[Tuple[int, int]]:
        # 只要有矩阵（哪怕是旧的），我就给你算！
        if self.homography_matrix is None: return None
        
        point_arr = np.array([[[xy[0], xy[1]]]], dtype=np.float32)
        try:
            transformed = cv2.perspectiveTransform(point_arr, self.homography_matrix)[0][0]
            
            # 🔥 额外保护：检查坐标是否飞出地球了
            # 如果算出来坐标是负数或者巨大无比，说明矩阵有问题，返回 None 避免画崩
            tx, ty = int(transformed[0]), int(transformed[1])
            if -500 < tx < 3000 and -500 < ty < 2000: # 宽容的边界
                return tx, ty
        except Exception:
            pass
            
        return None

    def transform_players(self, players: List[Player]):
        for p in players:
            # 使用脚底坐标 (bottom_center) 转换更准，如果没有就用中心点
            # 假设 Player.rect 是 [x1, y1, x2, y2]
            # anchor_x = (x1 + x2) / 2
            # anchor_y = y2 (脚底)
            result = self.transform_point(p.anchor)
            
            if result:
                # 这种赋值方式取决于你的 types.py 里的 Point 定义
                # 如果 p.pitch_position 是 Point 类型：
                from tactix.core.types import Point
                p.pitch_position = Point(x=result[0], y=result[1])
            else:
                p.pitch_position = None