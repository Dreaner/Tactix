"""
Project: Tactix
File Created: 2026-02-02 11:56:08
Author: Xingnan Zhu
File Name: config.py
Description: xxx...
"""


import torch

class Config:
    # 自动检测是否可以使用 Apple Silicon 加速
    DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # 视觉参数
    TEAM_A_COLOR = (0, 0, 255) # 红色 (BGR)
    TEAM_B_COLOR = (255, 0, 0) # 蓝色 (BGR)
    BALL_COLOR = (0, 255, 0)   # 绿色
    
    # 战术参数
    MAX_PASS_DISTANCE = 300    # 像素距离，超过这个距离不画线