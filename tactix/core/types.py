"""
Project: Tactix
File Created: 2026-02-02 16:16:34
Author: Xingnan Zhu
File Name: types.py
Description:
    Defines the core datasets structures and types used throughout the Tactix system.
    Includes definitions for TeamID, Point, Player, Ball, FrameData, and PitchConfig.
    Acts as the central contract for datasets exchange between modules.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any

import numpy as np


# ==========================================
# 1. Basic Definitions and Enums
# ==========================================

class TeamID(Enum):
    A = 0           # Home Team
    B = 1           # Away Team
    REFEREE = 2     # Referee
    GOALKEEPER = 3  # Goalkeeper
    UNKNOWN = -1    # Unknown

@dataclass
class Point:
    """Represents a coordinate point, useful for type hinting"""
    x: float
    y: float
    
    def to_tuple(self) -> Tuple[int, int]:
        return int(self.x), int(self.y)
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y])

# ==========================================
# 2. Geometry and Skeleton (Vision Layer Output)
# ==========================================

@dataclass
class Keypoints:
    """Keypoints extracted from Pose model"""
    nose: Optional[np.ndarray] = None
    left_foot: Optional[np.ndarray] = None
    right_foot: Optional[np.ndarray] = None
    # Can add shoulders, knees, etc. if needed for pose analysis
    
    @property
    def bottom_center(self) -> Optional[np.ndarray]:
        if self.left_foot is not None and self.right_foot is not None:
            return (self.left_foot + self.right_foot) / 2
        return self.left_foot or self.right_foot

# ==========================================
# 3. Core Entities (Player & Ball)
# ==========================================

@dataclass
class Player:
    """
    Player Entity: Contains visual info + physical info + tactical state
    """
    # --- Basic Visual Info (Filled by Detector/Tracker) ---
    id: int
    rect: Tuple[float, float, float, float]  # [x1, y1, x2, y2]
    class_id: int = 0
    confidence: float = 0.0
    team: TeamID = TeamID.UNKNOWN
    keypoints: Optional[Keypoints] = None

    # --- Physical Info (Filled by Semantics Layer) ---
    # Real pitch coordinates (Unit: meters). Origin usually at center circle or top-left corner.
    # Used for: Minimap drawing, real distance calculation, Voronoi diagram
    pitch_position: Optional[Point] = None 
    
    # Velocity vector (Unit: m/s or px/frame)
    # Used for: Speed bar display, movement prediction
    velocity: Optional[Point] = None 
    
    # Instantaneous speed magnitude (Scalar, for displaying "30 km/h")
    speed: float = 0.0 
    
    # Body orientation vector (Unit vector)
    # Used for: Pass probability calculation (hard to pass backwards)
    orientation: Optional[Point] = None
    
    # --- Tactical Info (Filled by Tactics Layer) ---
    # Pressure Index (0.0 - 1.0): How much pressure this player is under
    pressure: float = 0.0

    # --- Historical Info (For drawing trails/trajectories) ---
    # Records pixel coordinates of past N frames [(x,y), (x,y)...]
    trajectory: List[Point] = field(default_factory=list)

    @property
    def anchor(self) -> tuple[tuple[Any, ...], ...] | tuple[int, int]:
        """Drawing anchor point (Prefer feet)"""
        if self.keypoints and self.keypoints.bottom_center is not None:
            return tuple(self.keypoints.bottom_center.astype(int))
        x1, y1, x2, y2 = self.rect
        return int((x1 + x2) / 2), int(y2)

@dataclass
class Ball:
    rect: Tuple[float, float, float, float]
    score: float = 0.0
    pitch_position: Optional[Point] = None # Real position of the ball
    velocity: Optional[Point] = None       # Ball velocity (for shot/pass power)
    owner_id: Optional[int] = None         # Ball carrier ID
    
    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.rect
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

# ==========================================
# 4. Tactical Analysis Results (Analysis Layer Output)
# ==========================================

@dataclass
class PassEvent:
    """Defines a passing event (for passing network)"""
    sender_id: int
    receiver_id: int
    start_frame: int
    end_frame: int
    is_successful: bool = True

@dataclass
class FrameAnalysis:
    """
    Stores advanced tactical metadata for the current frame
    """
    # --- Offside Line ---
    # X coordinate of the last defender in pitch coordinate system (assuming X axis is goal line direction)
    offside_line_x: Optional[float] = None 
    # Offside line coordinates projected back to screen [(x1,y1), (x2,y2)]
    offside_line_pixels: Optional[Tuple[Point, Point]] = None
    
    # --- Space Control (Voronoi/Pitch Control) ---
    # Simple: Which team controls more area (0.0 - 1.0)
    possession_value: float = 0.5 
    
    # --- Passing Network ---
    # Potential receivers for the current ball carrier: {teammate_id: probability/score}
    pass_candidates: Dict[int, float] = field(default_factory=dict)

# ==========================================
# 5. Frame Bus (The Bus)
# ==========================================

@dataclass
class FrameData:
    frame_index: int
    image_shape: Tuple[int, int]
    
    # Entities
    players: List[Player] = field(default_factory=list)
    ball: Optional[Ball] = None
    
    # Camera Data (For AR stability)
    # Homography matrix (3x3 matrix), handles Pixel <-> Meter conversion
    homography: Optional[np.ndarray] = None 
    
    # Tactical Data
    analysis: Optional[FrameAnalysis] = None
    
    def get_team_players(self, team: TeamID) -> List[Player]:
        return [p for p in self.players if p.team == team]

    def get_player_by_id(self, pid: int) -> Optional[Player]:
        for p in self.players:
            if p.id == pid: return p
        return None
    
# ==========================================
# 6. Pitch Standards (FIFA / UEFA Standard)
# ==========================================
class PitchConfig:
    # Logical dimensions in real world (Unit: meters)
    # Used for calculating real speed and distance
    LENGTH = 105.0
    WIDTH = 68.0
    
    # Note: After generating the image, check its details in assets/pitch_bg.png
    # Right click image on Mac -> Get Info -> Dimensions
    PIXEL_WIDTH = 1559
    PIXEL_HEIGHT = 1010
    
    # Auto-calculate scale (Pixels/Meter)
    X_SCALE = PIXEL_WIDTH / LENGTH
    Y_SCALE = PIXEL_HEIGHT / WIDTH