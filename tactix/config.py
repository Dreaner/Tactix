"""
Project: Tactix
File Created: 2026-02-02 11:56:08
Author: Xingnan Zhu
File Name: config.py
Description: Central configuration file for paths, colors, and parameters.
"""


from dataclasses import dataclass, field
from typing import List, Tuple
from enum import Enum

class CalibrationMode(Enum):
    AI_ONLY = "ai"             # Pure AI (YOLO)
    MANUAL_FIXED = "manual"    # Pure Manual (Fixed points tracking)
    PANORAMA = "panorama"      # Panorama (Manual Init + Global Motion)
    # HYBRID = "hybrid"        # Future work

@dataclass
class Config:
    # === Path Settings ===
    PITCH_MODEL_PATH: str = "assets/weights/football_pitch.pt"
    PLAYER_MODEL_PATH: str = "assets/weights/football_v1.pt"
    
    INPUT_VIDEO: str = "assets/samples/InterGoalClip.mp4"
    OUTPUT_VIDEO: str = "assets/output/Final_V4_Result.mp4"
    PITCH_TEMPLATE: str = "assets/pitch_bg.png"
    
    # Export Settings
    EXPORT_DATA: bool = True
    OUTPUT_JSON: str = "assets/output/tracking_data.json"

    # === Model Parameters ===
    DEVICE: str = "mps"
    CONF_PITCH: float = 0.3
    CONF_PLAYER: float = 0.3

    # === Tactical Parameters ===
    MAX_PASS_DIST: int = 400
    BALL_OWNER_DIST: int = 60

    # === Calibration Settings ===
    INTERACTIVE_MODE: bool = True
    
    # Current Calibration Mode
    CALIBRATION_MODE: CalibrationMode = CalibrationMode.PANORAMA
    
    # Legacy flag (kept for compatibility, but logic moved to CALIBRATION_MODE)
    USE_MOCK_PITCH: bool = True 

    MANUAL_KEYPOINTS: List[Tuple[int, int, int]] = field(default_factory=lambda: [
        (137, 89, 9),    # L_PA_TOP_LINE
        (1126, 87, 3),   # MID_TOP
        (1045, 398, 2),  # CIRCLE_BOTTOM
        (138, 222, 15)   # L_PENALTY_SPOT
    ])