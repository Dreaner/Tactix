"""
Tactix Analytics — all analysis modules in one namespace.

    from tactix.analytics import (
        EventDetector,
        HeatmapAccumulator, PassNetwork, PressureIndex,
        ShotMap, ZoneAnalyzer, PassSonar, BuildupTracker,
        TransitionTracker,
        DuelHeatmap,
        CornerAnalyzer, FreeKickAnalyzer,
    )
"""
# Base
from tactix.analytics.base import HeatmapAccumulator, PassNetwork, PressureIndex
# Events (M0)
from tactix.analytics.events import EventDetector
# Attacking (M1)
from tactix.analytics.attacking import (
    BuildupSequence, BuildupTracker,
    PassSonar,
    ShotMap, ShotRecord,
    ZoneAnalyzer,
)
# Transition (M2)
from tactix.analytics.transition import TransitionPhase, TransitionSequence, TransitionTracker
# Defense (M3)
from tactix.analytics.defense import DuelHeatmap
# Set pieces (M4)
from tactix.analytics.set_pieces import (
    CornerAnalyzer, CornerSequence,
    FreeKickAnalyzer, FreeKickSequence,
)

__all__ = [
    # Base
    "HeatmapAccumulator", "PassNetwork", "PressureIndex",
    # M0
    "EventDetector",
    # M1
    "ShotMap", "ShotRecord", "ZoneAnalyzer", "PassSonar",
    "BuildupTracker", "BuildupSequence",
    # M2
    "TransitionTracker", "TransitionSequence", "TransitionPhase",
    # M3
    "DuelHeatmap",
    # M4
    "CornerAnalyzer", "CornerSequence",
    "FreeKickAnalyzer", "FreeKickSequence",
]
