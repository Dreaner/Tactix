"""
Project: Tactix
File Created: 2026-02-11
Author: Xingnan Zhu
File Name: shot_map.py
Description:
    Pure-data accumulator for shot events across the match.
    No rendering logic — see tactix.visualization.overlays.attacking.shot_map
    for the corresponding RGBA overlay renderer.

    Usage:
        shot_map = ShotMap(cfg)
        shot_map.record(shot_event)
        stats = shot_map.summary()
"""

from dataclasses import dataclass
from typing import List

from tactix.core.events import ShotEvent
from tactix.core.types import TeamID


@dataclass
class ShotRecord:
    """Single shot observation stored by ShotMap."""
    x: float                     # pitch x (meters)
    y: float                     # pitch y (meters)
    team: TeamID
    outcome: str                 # "goal" | "saved" | "blocked" | "off_target" | "unknown"
    on_target: bool
    frame_index: int


class ShotMap:
    """
    Accumulates all shots throughout the match.

    Produces:
        - .shots  — full list of ShotRecord
        - .summary() — per-team totals (total, on_target, goals)
    """

    def __init__(self) -> None:
        self.shots: List[ShotRecord] = []

    # ------------------------------------------------------------------
    # Accumulation
    # ------------------------------------------------------------------

    def record(self, shot: ShotEvent) -> None:
        """Register a new shot. Called whenever a ShotEvent fires."""
        if shot.location is None:
            return
        self.shots.append(ShotRecord(
            x=shot.location.x,
            y=shot.location.y,
            team=shot.team,
            outcome=shot.outcome,
            on_target=shot.on_target,
            frame_index=shot.frame_index,
        ))

    def update_outcome(self, index: int, outcome: str) -> None:
        """Retroactively update the outcome of the shot at position `index`."""
        if 0 <= index < len(self.shots):
            self.shots[index].outcome = outcome
            self.shots[index].on_target = outcome in ("goal", "saved")

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Returns basic shot statistics per team."""
        result = {}
        for team in (TeamID.A, TeamID.B):
            shots = [s for s in self.shots if s.team == team]
            result[team] = {
                "total": len(shots),
                "on_target": sum(1 for s in shots if s.on_target),
                "goals": sum(1 for s in shots if s.outcome == "goal"),
            }
        return result
