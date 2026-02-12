"""
Core data contracts used throughout Tactix.

    from tactix.core import FrameData, TeamID, Player, Ball, Point
    from tactix.core import FrameEvents, ShotEvent, PassEvent, DuelEvent
"""
from tactix.core.types import (
    Ball,
    FrameData,
    PitchConfig,
    Player,
    Point,
    TacticalOverlays,
    TeamID,
)
from tactix.core.events import (
    CornerEvent,
    DuelEvent,
    FrameEvents,
    FreeKickEvent,
    PassEvent,
    PossessionChangeEvent,
    ShotEvent,
)

__all__ = [
    # types
    "FrameData", "Player", "Ball", "Point", "TeamID",
    "TacticalOverlays", "PitchConfig",
    # events
    "FrameEvents", "ShotEvent", "PassEvent", "DuelEvent",
    "CornerEvent", "FreeKickEvent", "PossessionChangeEvent",
]
