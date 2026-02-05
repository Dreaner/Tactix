"""
Project: Tactix
File Created: 2026-02-02 11:56:08
Author: Xingnan Zhu
File Name: config.py
Description: Central configuration file for paths, colors, and parameters.
"""


from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class Config:
    # === Path Settings ===
    # Pitch Model (Model A)
    PITCH_MODEL_PATH: str = "assets/weights/football_pitch.pt"
    # Player Model (Model B)
    PLAYER_MODEL_PATH: str = "assets/weights/football_v1.pt"
    
    INPUT_VIDEO: str = "assets/samples/InterGoalClip.mp4"
    OUTPUT_VIDEO: str = "assets/output/Final_V4_Result.mp4"
    PITCH_TEMPLATE: str = "assets/pitch_bg.png"

    # === Model Parameters ===
    DEVICE: str = "mps"  # 'mps' for Mac, 'cuda' for Windows/Linux, 'cpu' for CPU only
    
    # Confidence Thresholds
    CONF_PITCH: float = 0.3  # Stricter for pitch keypoints
    CONF_PLAYER: float = 0.3 # Slightly lenient for players

    # === Tactical Parameters ===
    MAX_PASS_DIST: int = 400
    BALL_OWNER_DIST: int = 60


    # === [New] Calibration Mode Switches ===
    
    # INTERACTIVE_MODE: If True, launches the UI to let user click points at startup
    INTERACTIVE_MODE: bool = True

    # USE_MOCK_PITCH: If True, uses MANUAL_KEYPOINTS (either hardcoded or from interactive mode)
    # instead of the AI model.
    USE_MOCK_PITCH: bool = True

    # Manual calibration data (Frame 0)
    # Format: [x, y, keypoint_id]
    # ID comes from keypoints.py YOLO_INDEX_MAP
    # Default values (can be overwritten by Interactive Mode)
    MANUAL_KEYPOINTS: List[Tuple[int, int, int]] = field(default_factory=lambda: [
        (137, 89, 9),    # L_PA_TOP_LINE
        (1126, 87, 3),   # MID_TOP
        (1045, 398, 2),  # CIRCLE_BOTTOM
        (138, 222, 15)   # L_PENALTY_SPOT
    ])