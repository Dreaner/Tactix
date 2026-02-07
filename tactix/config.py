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
import supervision as sv

class CalibrationMode(Enum):
    AI_ONLY = "ai"             # Pure AI (YOLO)
    MANUAL_FIXED = "manual"    # Pure Manual (Fixed points tracking)
    PANORAMA = "panorama"      # Panorama (Manual Init + Global Motion)
    # HYBRID = "hybrid"        # Future work

@dataclass
class Colors:
    """
    Centralized color palette for the application.
    Colors are defined as (R, G, B) tuples.
    """
    # Entities
    TEAM_A: Tuple[int, int, int] = (230, 57, 70)    # Red #E63946
    TEAM_B: Tuple[int, int, int] = (69, 123, 157)   # Blue #457B9D
    REFEREE: Tuple[int, int, int] = (255, 214, 10)  # Yellow #FFD60A
    GOALKEEPER: Tuple[int, int, int] = (29, 53, 87) # Dark Blue/Black #1D3557
    UNKNOWN: Tuple[int, int, int] = (128, 128, 128) # Grey
    BALL: Tuple[int, int, int] = (255, 165, 0)      # Orange #FFA500

    # UI / Debug
    KEYPOINT: Tuple[int, int, int] = (0, 255, 255)  # Cyan (BGR: 255, 255, 0) -> RGB: 0, 255, 255
    TEXT: Tuple[int, int, int] = (255, 255, 255)    # White
    
    # Pressure Colors (Low -> High)
    PRESSURE_LOW: Tuple[int, int, int] = (0, 255, 0)    # Green
    PRESSURE_MED: Tuple[int, int, int] = (255, 255, 0)  # Yellow
    PRESSURE_HIGH: Tuple[int, int, int] = (255, 0, 0)   # Red
    
    @staticmethod
    def to_bgr(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Convert RGB to BGR for OpenCV"""
        return (rgb[2], rgb[1], rgb[0])

    @staticmethod
    def to_sv(rgb: Tuple[int, int, int]) -> sv.Color:
        """Convert RGB to Supervision Color"""
        return sv.Color(r=rgb[0], g=rgb[1], b=rgb[2])

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
    PRESSURE_RADIUS: float = 8.0 # Meters
    SHADOW_LENGTH: float = 20.0 # Meters
    SHADOW_ANGLE: float = 20.0 # Degrees

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

    # === Visualization Settings (Default State) ===
    SHOW_VORONOI: bool = False
    SHOW_HEATMAP: bool = False
    SHOW_COMPACTNESS: bool = False
    SHOW_PASS_NETWORK: bool = False
    SHOW_VELOCITY: bool = False
    SHOW_PRESSURE: bool = False
    SHOW_COVER_SHADOW: bool = True # New flag
    SHOW_DEBUG_KEYPOINTS: bool = False