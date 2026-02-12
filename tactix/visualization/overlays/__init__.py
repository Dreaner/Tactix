"""
Tactix overlay renderers — all rendering classes in one namespace.

    from tactix.visualization.overlays import (
        # Base
        HeatmapOverlay, VoronoiOverlay, CompactnessOverlay,
        CoverShadowOverlay, CentroidOverlay, WidthLengthOverlay,
        # M1 Attacking
        ShotMapOverlay, Zone14Overlay, PassSonarOverlay, BuildupOverlay,
        # M2 Transition
        TransitionOverlay,
        # M3 Defense
        DuelHeatmapOverlay,
        # M4 Set pieces
        SetPiecesOverlay,
    )
"""
# Base
from tactix.visualization.overlays.base import (
    HeatmapOverlay,
    VoronoiOverlay,
    CompactnessOverlay,
    CoverShadowOverlay,
    CentroidOverlay,
    WidthLengthOverlay,
)
# M1 — Attacking phase
from tactix.visualization.overlays.attacking import (
    ShotMapOverlay,
    Zone14Overlay,
    PassSonarOverlay,
    BuildupOverlay,
)
# M2 — Transition phase
from tactix.visualization.overlays.transition import TransitionOverlay
# M3 — Defense phase
from tactix.visualization.overlays.defense import DuelHeatmapOverlay
# M4 — Set pieces
from tactix.visualization.overlays.set_pieces import SetPiecesOverlay

__all__ = [
    # Base
    "HeatmapOverlay", "VoronoiOverlay", "CompactnessOverlay",
    "CoverShadowOverlay", "CentroidOverlay", "WidthLengthOverlay",
    # M1
    "ShotMapOverlay", "Zone14Overlay", "PassSonarOverlay", "BuildupOverlay",
    # M2
    "TransitionOverlay",
    # M3
    "DuelHeatmapOverlay",
    # M4
    "SetPiecesOverlay",
]
