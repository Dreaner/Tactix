"""
Project: Tactix
File Created: 2026-02-02 23:35:48
Author: Xingnan Zhu
File Name: keypoints.py
Description: xxx...
"""


# tactix/core/keypoints.py
import numpy as np
from tactix.core.types import PitchConfig

# ==========================================
# ⚙️ 基础配置
# ==========================================
# 直接读取 types.py 里的配置，避免硬编码
W = PitchConfig.PIXEL_WIDTH   # 1559
H = PitchConfig.PIXEL_HEIGHT  # 1010

LENGTH_M = PitchConfig.LENGTH # 105.0
WIDTH_M = PitchConfig.WIDTH   # 68.0

# 比例尺 (像素/米)
X_PER_M = W / LENGTH_M
Y_PER_M = H / WIDTH_M

def to_px(x_m, y_m):
    """把物理坐标(米)转换为战术板像素坐标"""
    return [int(x_m * X_PER_M), int(y_m * Y_PER_M)]

# ==========================================
# 📍 全场关键点坐标库 (Full Pitch Dictionary)
# ==========================================
# 原点 (0,0) = 左上角角旗杆
# X轴 -> 向右 (0 到 105)
# Y轴 -> 向下 (0 到 68)

KEY_POINTS = {
    # ----------------------------------------
    # 1. 四个角旗 (Corners)
    # ----------------------------------------
    "TL_CORNER": to_px(0, 0),             # 左上 (Top-Left)
    "BL_CORNER": to_px(0, WIDTH_M),       # 左下 (Bottom-Left)
    "TR_CORNER": to_px(LENGTH_M, 0),      # 右上 (Top-Right)
    "BR_CORNER": to_px(LENGTH_M, WIDTH_M),# 右下 (Bottom-Right)

    # ----------------------------------------
    # 2. 中线区域 (Midfield)
    # ----------------------------------------
    "MID_TOP":    to_px(LENGTH_M/2, 0),       # 中线上沿
    "MID_BOTTOM": to_px(LENGTH_M/2, WIDTH_M), # 中线下沿
    "CENTER_SPOT":to_px(LENGTH_M/2, WIDTH_M/2), # 中圈开球点
    
    # 中圈与中线交点 (半径 9.15m)
    "CIRCLE_TOP":    to_px(LENGTH_M/2, WIDTH_M/2 - 9.15),
    "CIRCLE_BOTTOM": to_px(LENGTH_M/2, WIDTH_M/2 + 9.15),

    # ----------------------------------------
    # 3. 左半场禁区 (Left Penalty Area) - X = 0 ~ 16.5
    # ----------------------------------------
    # 大禁区 (宽度 40.32m, 距边线 13.84m)
    "L_PA_TOP_LINE":    to_px(0, 13.84),        # 上沿与底线交点
    "L_PA_TOP_CORNER":  to_px(16.5, 13.84),     # 🔥左大禁区-右上角 (常用!)
    "L_PA_BOTTOM_CORNER":to_px(16.5, 68-13.84), # 🔥左大禁区-右下角 (常用!)
    "L_PA_BOTTOM_LINE": to_px(0, 68-13.84),     # 下沿与底线交点
    "L_PENALTY_SPOT":   to_px(11.0, 34.0),      # 左点球点

    # 小禁区 (宽度 18.32m, 距边线 24.84m, 深 5.5m)
    "L_GA_TOP_CORNER":    to_px(5.5, 24.84),    # 左小禁区-右上角
    "L_GA_BOTTOM_CORNER": to_px(5.5, 68-24.84), # 左小禁区-右下角

    # ----------------------------------------
    # 4. 右半场禁区 (Right Penalty Area) - X = 88.5 ~ 105
    # ----------------------------------------
    # 计算逻辑：X = 105 - 16.5 = 88.5
    "R_PA_TOP_LINE":    to_px(LENGTH_M, 13.84),      # 上沿与右底线交点
    "R_PA_TOP_CORNER":  to_px(105-16.5, 13.84),      # 🔥右大禁区-左上角 (常用!)
    "R_PA_BOTTOM_CORNER":to_px(105-16.5, 68-13.84),  # 🔥右大禁区-左下角 (常用!)
    "R_PA_BOTTOM_LINE": to_px(LENGTH_M, 68-13.84),   # 下沿与右底线交点
    "R_PENALTY_SPOT":   to_px(105-11.0, 34.0),       # 右点球点

    # 小禁区 - X = 105 - 5.5 = 99.5
    "R_GA_TOP_CORNER":    to_px(105-5.5, 24.84),     # 右小禁区-左上角
    "R_GA_BOTTOM_CORNER": to_px(105-5.5, 68-24.84),  # 右小禁区-左下角
}

def get_target_points(keys):
    """
    根据名字获取坐标数组
    """
    points = []
    for k in keys:
        if k not in KEY_POINTS:
            # 容错：允许反向查找 (比如把 PA_TOP_LEFT 映射到 L_PA_TOP_CORNER)
            # 这里简单起见，如果找不到就报错
            available = list(KEY_POINTS.keys())
            raise ValueError(f"❌ 未知点位 '{k}'。请从以下列表选择:\n{available}")
        points.append(KEY_POINTS[k])
    return np.array(points)


# ==========================================
# 5. YOLO V4 模型输出映射 (Model Definition)
# ==========================================
# 这个顺序必须与训练时的 football-pitch.yaml 严格一致
YOLO_INDEX_MAP = {
    0: "CENTER_SPOT",
    1: "CIRCLE_TOP",         # Circle_Intersect_Top
    2: "CIRCLE_BOTTOM",      # Circle_Intersect_Bot
    3: "MID_TOP",            # Mid_Line_Top
    4: "MID_BOTTOM",         # Mid_Line_Bottom
    
    5: "TL_CORNER",          # L_Corner_TL
    6: "BL_CORNER",          # L_Corner_BL
    
    # 左禁区关键点
    7: "L_PA_TOP_CORNER",    # L_Penalty_TL
    8: "L_PA_BOTTOM_CORNER", # L_Penalty_BL
    9: "L_PA_TOP_LINE",      # L_Penalty_Line_Top
    10: "L_PA_BOTTOM_LINE",  # L_Penalty_Line_Bot
    11: "L_GA_TOP_CORNER",   # L_SixYard_TL
    12: "L_GA_BOTTOM_CORNER",# L_SixYard_BL
    13: "L_GA_TOP_LINE",     # L_SixYard_Line_Top (需在 KEY_POINTS 里补全或忽略)
    14: "L_GA_BOTTOM_LINE",  # L_SixYard_Line_Bot (需在 KEY_POINTS 里补全或忽略)
    15: "L_PENALTY_SPOT",
    
    # 右半场 (对称)
    16: "TR_CORNER",
    17: "BR_CORNER",
    18: "R_PA_TOP_CORNER",
    19: "R_PA_BOTTOM_CORNER",
    20: "R_PA_TOP_LINE",
    21: "R_PA_BOTTOM_LINE",
    22: "R_GA_TOP_CORNER",
    23: "R_GA_BOTTOM_CORNER",
    24: "R_GA_TOP_LINE",
    25: "R_GA_BOTTOM_LINE",
    26: "R_PENALTY_SPOT"
}

