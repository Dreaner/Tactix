"""
Project: Tactix
File Created: 2026-02-02 16:16:28
Author: Xingnan Zhu
File Name: geometry.py
Description:
    Defines the physical geometry of the football pitch. It maps standard
    pitch landmarks to their real-world coordinates (in meters), serving as
    the ground truth for perspective transformation and tactical analysis.
"""

from tactix.core.types import PitchConfig

# 真实球场尺寸 (米)
L = PitchConfig.LENGTH # 105
W = PitchConfig.WIDTH  # 68

# 定义关键点在"真实世界"的坐标 (单位: 米)
# 坐标系: 原点 (0,0) 在左上角角旗
# X轴: 向右, Y轴: 向下
WORLD_POINTS = {
    "TL_CORNER": [0, 0],
    "BL_CORNER": [0, W],
    "TR_CORNER": [L, 0],
    "BR_CORNER": [L, W],
    
    "MID_TOP":    [L/2, 0],
    "MID_BOTTOM": [L/2, W],
    "CENTER_SPOT":[L/2, W/2],
    "CIRCLE_TOP": [L/2, W/2 - 9.15],
    "CIRCLE_BOTTOM": [L/2, W/2 + 9.15],

    # 左禁区
    "L_PA_TOP_CORNER":   [16.5, 13.84],
    "L_PA_BOTTOM_CORNER":[16.5, W-13.84],
    "L_PA_TOP_LINE":     [0, 13.84],
    "L_PA_BOTTOM_LINE":  [0, W-13.84],
    "L_PENALTY_SPOT":    [11.0, W/2],
    
    # 左小禁区
    "L_GA_TOP_CORNER":   [5.5, 24.84],
    "L_GA_BOTTOM_CORNER":[5.5, W-24.84],

    # 右禁区 (X = 105 - distance)
    "R_PA_TOP_CORNER":   [L-16.5, 13.84],
    "R_PA_BOTTOM_CORNER":[L-16.5, W-13.84],
    "R_PA_TOP_LINE":     [L, 13.84],
    "R_PA_BOTTOM_LINE":  [L, W-13.84],
    "R_PENALTY_SPOT":    [L-11.0, W/2],
    
    # 右小禁区
    "R_GA_TOP_CORNER":   [L-5.5, 24.84],
    "R_GA_BOTTOM_CORNER":[L-5.5, W-24.84],
}