"""
Project: Tactix
File Created: 2026-02-02 23:22:57
Author: Xingnan Zhu
File Name: transformer.py
Description: xxx...
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from tactix.core.types import PitchConfig

class ViewTransformer:
    def __init__(self, source_points: np.ndarray, target_points: np.ndarray = None):
        """
        初始化透视变换器
        :param source_points: 视频中点击的4个点 [TL, TR, BR, BL]
        :param target_points: 战术板上对应的4个点坐标 (自定义模式)
        """
        if source_points is None or len(source_points) != 4:
            raise ValueError("必须提供 4 个源点坐标")

        source_points = source_points.astype(np.float32)

        # 1. 确定目标点
        if target_points is None:
            # 默认模式：全场映射 (0,0) -> (w, h)
            w = PitchConfig.PIXEL_WIDTH
            h = PitchConfig.PIXEL_HEIGHT
            self.target_vertices = np.array([
                [0, 0],       # 左上
                [w, 0],       # 右上
                [w, h],       # 右下
                [0, h]        # 左下
            ], dtype=np.float32)
        else:
            # 🔥 高级模式：使用你传入的自定义点 (比如中线、禁区角等)
            if len(target_points) != 4:
                raise ValueError("目标点必须也是 4 个")
            self.target_vertices = target_points.astype(np.float32)
        
        # 2. 计算变换矩阵
        self.matrix = cv2.getPerspectiveTransform(source_points, self.target_vertices)
        print(f"✅ 透视变换矩阵初始化完成 (目标点模式: {'自定义' if target_points is not None else '默认全场'})")

    def transform_point(self, xy: Tuple[float, float]) -> Optional[Tuple[int, int]]:
        """
        把视频坐标 (x, y) -> 战术板坐标 (x, y)
        """
        if self.matrix is None:
            return None
            
        # OpenCV 需要 [[[x, y]]] 形状的数组
        point_array = np.array([[[xy[0], xy[1]]]], dtype=np.float32)
        
        # 执行变换
        transformed_point = cv2.perspectiveTransform(point_array, self.matrix)[0][0]
        
        return int(transformed_point[0]), int(transformed_point[1])

    def transform_players(self, players: List):
        """
        批量给球员添加 pitch_position 属性
        """
        from tactix.core.types import Point # 避免循环引用
        
        for p in players:
            # 使用 anchor (脚下点) 进行映射最准确
            p_map_pos = self.transform_point(p.anchor)
            
            if p_map_pos:
                # 存入 pitch_position
                p.pitch_position = Point(x=p_map_pos[0], y=p_map_pos[1])