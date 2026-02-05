"""
Project: Tactix
File Created: 2026-02-02 23:35:48
Author: Xingnan Zhu
File Name: keypoints.py
Description: Defines standard football pitch dimensions and key coordinate points.
    It maps physical locations to pixel coordinates and provides a mapping for
    YOLO model outputs to specific pitch landmarks.
"""


# tactix/core/keypoints.py
import numpy as np
from tactix.core.types import PitchConfig

# ==========================================
# ⚙️ Basic Configuration
# ==========================================
# Read configuration from types.py to avoid hardcoding
W = PitchConfig.PIXEL_WIDTH   # 1559
H = PitchConfig.PIXEL_HEIGHT  # 1010

LENGTH_M = PitchConfig.LENGTH # 105.0
WIDTH_M = PitchConfig.WIDTH   # 68.0

# Scale (Pixels/Meter)
X_PER_M = W / LENGTH_M
Y_PER_M = H / WIDTH_M

def to_px(x_m, y_m):
    """Convert physical coordinates (meters) to tactical board pixel coordinates"""
    return [int(x_m * X_PER_M), int(y_m * Y_PER_M)]

# ==========================================
# 📍 Full Pitch Keypoint Dictionary
# ==========================================
# Origin (0,0) = Top-Left Corner Flag
# X-axis -> Right (0 to 105)
# Y-axis -> Down (0 to 68)

KEY_POINTS = {
    # ----------------------------------------
    # 1. Four Corners
    # ----------------------------------------
    "TL_CORNER": to_px(0, 0),             # Top-Left
    "BL_CORNER": to_px(0, WIDTH_M),       # Bottom-Left
    "TR_CORNER": to_px(LENGTH_M, 0),      # Top-Right
    "BR_CORNER": to_px(LENGTH_M, WIDTH_M),# Bottom-Right

    # ----------------------------------------
    # 2. Midfield Area
    # ----------------------------------------
    "MID_TOP":    to_px(LENGTH_M/2, 0),       # Midline Top
    "MID_BOTTOM": to_px(LENGTH_M/2, WIDTH_M), # Midline Bottom
    "CENTER_SPOT":to_px(LENGTH_M/2, WIDTH_M/2), # Center Spot
    
    # Center Circle Intersections (Radius 9.15m)
    "CIRCLE_TOP":    to_px(LENGTH_M/2, WIDTH_M/2 - 9.15),
    "CIRCLE_BOTTOM": to_px(LENGTH_M/2, WIDTH_M/2 + 9.15),

    # ----------------------------------------
    # 3. Left Penalty Area - X = 0 ~ 16.5
    # ----------------------------------------
    # Penalty Area (Width 40.32m, Distance from sideline 13.84m)
    "L_PA_TOP_LINE":    to_px(0, 13.84),        # Top line intersection with goal line
    "L_PA_TOP_CORNER":  to_px(16.5, 13.84),     # Left PA Top-Right Corner (Common!)
    "L_PA_BOTTOM_CORNER":to_px(16.5, 68-13.84), # Left PA Bottom-Right Corner (Common!)
    "L_PA_BOTTOM_LINE": to_px(0, 68-13.84),     # Bottom line intersection with goal line
    "L_PENALTY_SPOT":   to_px(11.0, 34.0),      # Left Penalty Spot

    # Goal Area (Width 18.32m, Distance from sideline 24.84m, Depth 5.5m)
    "L_GA_TOP_CORNER":    to_px(5.5, 24.84),    # Left GA Top-Right Corner
    "L_GA_BOTTOM_CORNER": to_px(5.5, 68-24.84), # Left GA Bottom-Right Corner

    # ----------------------------------------
    # 4. Right Penalty Area - X = 88.5 ~ 105
    # ----------------------------------------
    # Calculation Logic: X = 105 - 16.5 = 88.5
    "R_PA_TOP_LINE":    to_px(LENGTH_M, 13.84),      # Top line intersection with right goal line
    "R_PA_TOP_CORNER":  to_px(105-16.5, 13.84),      # Right PA Top-Left Corner (Common!)
    "R_PA_BOTTOM_CORNER":to_px(105-16.5, 68-13.84),  # Right PA Bottom-Left Corner (Common!)
    "R_PA_BOTTOM_LINE": to_px(LENGTH_M, 68-13.84),   # Bottom line intersection with right goal line
    "R_PENALTY_SPOT":   to_px(105-11.0, 34.0),       # Right Penalty Spot

    # Goal Area - X = 105 - 5.5 = 99.5
    "R_GA_TOP_CORNER":    to_px(105-5.5, 24.84),     # Right GA Top-Left Corner
    "R_GA_BOTTOM_CORNER": to_px(105-5.5, 68-24.84),  # Right GA Bottom-Left Corner
}

def get_target_points(keys):
    """
    Get coordinate array by key names
    """
    points = []
    for k in keys:
        if k not in KEY_POINTS:
            # Fault tolerance: Allow reverse lookup (e.g., mapping PA_TOP_LEFT to L_PA_TOP_CORNER)
            # For simplicity, raise error if not found
            available = list(KEY_POINTS.keys())
            raise ValueError(f"❌ Unknown point '{k}'. Please choose from:\n{available}")
        points.append(KEY_POINTS[k])
    return np.array(points)


# ==========================================
# 5. YOLO V4 Model Output Mapping (Model Definition)
# ==========================================
# This order must strictly match football-pitch.yaml used during training
YOLO_INDEX_MAP = {
    0: "CENTER_SPOT",
    1: "CIRCLE_TOP",         # Circle_Intersect_Top
    2: "CIRCLE_BOTTOM",      # Circle_Intersect_Bot
    3: "MID_TOP",            # Mid_Line_Top
    4: "MID_BOTTOM",         # Mid_Line_Bottom
    
    5: "TL_CORNER",          # L_Corner_TL
    6: "BL_CORNER",          # L_Corner_BL
    
    # Left Penalty Area Keypoints
    7: "L_PA_TOP_CORNER",    # L_Penalty_TL
    8: "L_PA_BOTTOM_CORNER", # L_Penalty_BL
    9: "L_PA_TOP_LINE",      # L_Penalty_Line_Top
    10: "L_PA_BOTTOM_LINE",  # L_Penalty_Line_Bot
    11: "L_GA_TOP_CORNER",   # L_SixYard_TL
    12: "L_GA_BOTTOM_CORNER",# L_SixYard_BL
    13: "L_GA_TOP_LINE",     # L_SixYard_Line_Top (Need to complete in KEY_POINTS or ignore)
    14: "L_GA_BOTTOM_LINE",  # L_SixYard_Line_Bot (Need to complete in KEY_POINTS or ignore)
    15: "L_PENALTY_SPOT",
    
    # Right Half (Symmetric)
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
