"""
Project: Tactix
File Created: 2026-02-11
Author: Xingnan Zhu
File Name: events.py
Description:
    Event data classes for the Tactix tactical analysis system.

    Events are discrete, timestamped occurrences detected from the
    frame-by-frame tracking data. They form the foundation of all
    Phase analysis (Attacking, Transition, Defense, Set Pieces).

    Events are produced by EventDetector (tactix/tactics/event_detector.py)
    and consumed by phase-specific analyzers.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from tactix.core.types import Point, TeamID


# ==========================================
# Possession
# ==========================================

@dataclass
class PossessionChangeEvent:
    """
    Ball ownership transferred from one team to the other.
    Confirmed only after POSSESSION_CONFIRM_FRAMES consecutive frames
    to suppress noise from split-second proximity changes.
    """
    frame_index: int
    losing_team: TeamID
    gaining_team: TeamID
    location: Optional[Point] = None   # Ball position at moment of change
    losing_player_id: int = -1
    gaining_player_id: int = -1


# ==========================================
# Ball Actions
# ==========================================

@dataclass
class PassEvent:
    """
    Ball successfully transferred from one teammate to another.
    Detected when ball.owner_id changes within the same team.
    """
    frame_index: int
    sender_id: int
    receiver_id: int
    team: TeamID
    origin: Optional[Point] = None       # Sender's pitch position
    destination: Optional[Point] = None  # Receiver's pitch position
    distance_m: float = 0.0              # Pass distance in meters
    angle_deg: float = 0.0               # Direction (0° = rightward along X axis)


@dataclass
class ShotEvent:
    """
    A player attempts a shot toward the opponent's goal.
    Detected by high ball velocity combined with direction toward goal.
    """
    frame_index: int
    shooter_id: int
    team: TeamID
    location: Optional[Point] = None     # Shooter's pitch position
    distance_to_goal_m: float = 0.0
    angle_to_goal_deg: float = 0.0
    ball_speed_ms: float = 0.0           # Ball speed at moment of shot (m/s)
    on_target: bool = False              # True if headed toward goal frame
    outcome: str = "unknown"             # "goal" | "saved" | "blocked" | "off_target" | "unknown"


@dataclass
class DuelEvent:
    """
    Two opposing players contesting for the ball within DUEL_DISTANCE.
    Updated every frame; consecutive frames produce repeated events
    (deduplicate downstream if needed).
    """
    frame_index: int
    player_a_id: int
    player_b_id: int
    team_a: TeamID
    team_b: TeamID
    location: Optional[Point] = None
    distance_m: float = 0.0
    winner_team: Optional[TeamID] = None   # None = contested


# ==========================================
# Set Pieces
# ==========================================

@dataclass
class CornerEvent:
    """
    Ball crossed the goal line and last touched by the defending team.
    Triggers corner-kick analysis for the next ~10 seconds.
    """
    frame_index: int
    attacking_team: TeamID
    defending_team: TeamID
    side: str = "unknown"   # "left" | "right" (which side of the pitch)
    taker_id: int = -1
    outcome: str = "unknown"   # "shot" | "goal" | "clearance" | "possession" | "unknown"


@dataclass
class FreeKickEvent:
    """
    A free kick is about to be taken. Detected when a defensive wall
    (≥ WALL_MIN_PLAYERS players in a linear formation) is identified
    near the ball while play is paused.
    """
    frame_index: int
    attacking_team: TeamID
    location: Optional[Point] = None
    distance_to_goal_m: float = 0.0
    angle_to_goal_deg: float = 0.0
    is_direct: bool = True
    wall_size: int = 0
    estimated_xg: float = 0.0
    outcome: str = "unknown"   # "goal" | "saved" | "blocked" | "miss" | "unknown"


# ==========================================
# Frame-level event container
# ==========================================

@dataclass
class FrameEvents:
    """
    All events detected in a single frame.
    Produced by EventDetector and passed to phase analyzers.
    """
    frame_index: int
    possession_change: Optional[PossessionChangeEvent] = None
    passes: List[PassEvent] = field(default_factory=list)
    shots: List[ShotEvent] = field(default_factory=list)
    duels: List[DuelEvent] = field(default_factory=list)
    corner: Optional[CornerEvent] = None
    free_kick: Optional[FreeKickEvent] = None

    def has_any(self) -> bool:
        return (
            self.possession_change is not None
            or bool(self.passes)
            or bool(self.shots)
            or bool(self.duels)
            or self.corner is not None
            or self.free_kick is not None
        )
