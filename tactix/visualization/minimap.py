"""
Project: Tactix
File Created: 2026-02-03 09:54:59
Author: Xingnan Zhu
File Name: minimap.py
Description: 
    Minimap Renderer
"""


import cv2
import numpy as np
from typing import List, Optional
from tactix.core.types import FrameData, TeamID

class MinimapRenderer:
    def __init__(self, bg_image_path: str, width: int = 1559, height: int = 1010):
        # 加载背景图
        self.bg_image = cv2.imread(bg_image_path)
        if self.bg_image is None:
            print(f"⚠️ Warning: Minimap background not found at {bg_image_path}. Using black.")
            self.bg_image = np.zeros((height, width, 3), dtype=np.uint8)
            
        # 预定义的颜色表 (BGR)
        self.colors = {
            TeamID.A: (0, 0, 255),       # 红
            TeamID.B: (255, 0, 0),       # 蓝
            TeamID.GOALKEEPER: (0, 255, 255), # 黄
            TeamID.REFEREE: (0, 0, 0),   # 黑
            TeamID.UNKNOWN: (200, 200, 200) # 灰
        }

    def draw(self, frame_data: FrameData) -> np.ndarray:
        """
        绘制当前帧的小地图
        """
        # 复制背景，避免污染原图
        minimap = self.bg_image.copy()

        # 1. 画球员
        for p in frame_data.players:
            if p.pitch_position:
                mx, my = int(p.pitch_position.x), int(p.pitch_position.y)
                color = self.colors.get(p.team, self.colors[TeamID.UNKNOWN])
                
                # 实心点 + 描边
                cv2.circle(minimap, (mx, my), 15, color, -1)
                cv2.circle(minimap, (mx, my), 15, (255, 255, 255), 2)

        # 2. 画球
        if frame_data.ball and frame_data.ball.pitch_position:
            bx, by = int(frame_data.ball.pitch_position.x), int(frame_data.ball.pitch_position.y)
            cv2.circle(minimap, (bx, by), 12, (0, 0, 0), -1)
            cv2.circle(minimap, (bx, by), 8, (255, 255, 255), -1)

        return minimap