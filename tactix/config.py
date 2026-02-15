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
    PLAYER_MODEL_PATH: str = "assets/weights/ball_player_yolo26x.pt"
    
    INPUT_VIDEO: str = "assets/samples/test2.mp4"
    OUTPUT_VIDEO: str = "assets/output/test2_Result.mp4"
    PITCH_TEMPLATE: str = "assets/pitch_bg.png"
    
    # FIFA EPTS Standard Transfer Format Export
    EXPORT_STF: bool = True
    OUTPUT_STF_DIR: str = "assets/output/stf"
    STF_MATCH_ID: str = "TACTIX-001"
    STF_HOME_TEAM_NAME: str = "Home"
    STF_AWAY_TEAM_NAME: str = "Away"
    STF_VENUE: str = "Unknown Stadium"
    STF_MATCH_DATE: str = ""  # Auto-filled from run timestamp if empty

    # === Model Parameters ===
    DEVICE: str = "mps"
    CONF_PITCH: float = 0.3
    CONF_PLAYER: float = 0.3
    
    # Jersey Number Detection (OCR-based player identification)
    ENABLE_JERSEY_OCR: bool = True   # Auto-disabled if easyocr not installed
    JERSEY_MIN_CROP_H: int = 40      # Skip OCR if bbox height too small
    JERSEY_MIN_CROP_W: int = 30      # Skip OCR if bbox width too small
    JERSEY_OCR_FRAME_SKIP: int = 10  # Run OCR every Nth frame for performance

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

# === Visualization Settings (Default State) ===
    GEOMETRY_ENABLED: bool = True # Master switch for pitch calibration/minimap
    
    SHOW_MINIMAP: bool = False
    SHOW_VORONOI: bool = False
    SHOW_HEATMAP: bool = False
    SHOW_COMPACTNESS: bool = False
    SHOW_PASS_NETWORK: bool = False
    SHOW_VELOCITY: bool = False
    SHOW_PRESSURE: bool = False
    SHOW_COVER_SHADOW: bool = False
    SHOW_TEAM_CENTROID: bool = False
    SHOW_TEAM_WIDTH_LENGTH: bool = False
    SHOW_DEBUG_KEYPOINTS: bool = False

    # === Color Pre-Scan Settings ===
    # Pre-scan the video to collect jersey colors before the main pipeline.
    # This produces a much more stable K-Means classifier.
    ENABLE_COLOR_PRESCAN: bool = True
    PRESCAN_NUM_FRAMES: int = 30      # How many frames to sample across the video
    PRESCAN_MIN_PLAYERS: int = 4      # Min players per frame to include in training data

    # === Tracking & Interpolation Settings ===
    # Max consecutive frames the ball can be missing before interpolation stops.
    BALL_INTERP_MAX_GAP: int = 10

    # === Cache Settings ===
    ENABLE_CACHE: bool = False
    CACHE_DIR: str = "assets/cache"

    # === Zone Definitions (meters) ===
    # Zone 14: area directly in front of the penalty box
    ZONE_14: tuple = (77, 88, 25, 43)       # (x_min, x_max, y_min, y_max)
    ATTACKING_THIRD_X: float = 70.0          # x > this value = attacking third
    DEFENSIVE_THIRD_X: float = 35.0          # x < this value = defensive third
    WIDE_ZONE_Y: float = 15.0                # y < this OR y > (68 - this) = wide channel

    # === Event Detection Thresholds ===
    SHOT_VELOCITY_THRESHOLD: float = 10.0   # m/s — minimum ball speed to detect a shot
    DUEL_DISTANCE: float = 2.0              # m  — max distance to classify as a duel
    POSSESSION_CONFIRM_FRAMES: int = 3      # frames to confirm a possession change
    WALL_MIN_PLAYERS: int = 3               # min players to constitute a free-kick wall
    WALL_LINEAR_TOLERANCE: float = 1.5      # m — SVD minor-axis tolerance for wall linearity

    # === Phase Analysis Toggles ===
    SHOW_SHOT_MAP: bool = False
    SHOW_PASS_SONAR: bool = False
    SHOW_ZONE_14: bool = False
    SHOW_BUILDUP: bool = False
    SHOW_TRANSITION: bool = False
    SHOW_DUEL_HEATMAP: bool = False
    SHOW_SET_PIECES: bool = False
    SHOW_FORMATION: bool = False

    # === Formation Detection Parameters ===
    FORMATION_WINDOW_FRAMES: int = 90    # ~3s at 30fps — sliding window for mode vote
    FORMATION_MIN_PLAYERS: int = 8       # min outfield players needed per team

    # === PDF Report Export ===
    EXPORT_PDF: bool = True
    OUTPUT_PDF: str = "assets/output/match_report.pdf"

    # === Transition Parameters ===
    TRANSITION_ATTACK_MAX_FRAMES: int = 450   # ≈ 15 s at 30 fps
    TRANSITION_DEFEND_MAX_FRAMES: int = 900   # ≈ 30 s at 30 fps

    # === Set Piece Parameters ===
    SET_PIECE_WINDOW_FRAMES: int = 300        # ≈ 10 s at 30 fps