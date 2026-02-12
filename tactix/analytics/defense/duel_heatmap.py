"""
Project: Tactix
File Created: 2026-02-12
Author: Xingnan Zhu
File Name: duel_heatmap.py
Description:
    Cross-frame accumulator for defensive duel locations.
    Mirrors the pattern of HeatmapAccumulator but updates only when a
    DuelEvent fires, recording the duel location on the pitch grid.

    Three grids are maintained:
        .grid_a     — duels involving a Team A player (as challenger/defender)
        .grid_b     — duels involving a Team B player
        .grid_total — all duels combined

    When a DuelEvent has winner_team set, a fourth grid records outcomes:
        .grid_won_a — duels won by Team A at that location
        .grid_won_b — duels won by Team B at that location

    No rendering — see tactix.visualization.overlays.defense.duel_heatmap
"""

import numpy as np

from tactix.core.events import DuelEvent
from tactix.core.types import PitchConfig, TeamID


class DuelHeatmap:
    """
    Accumulates duel events onto a spatial grid.

    Public attributes for the renderer:
        .grid_a / .grid_b / .grid_total  — float32 arrays
        .grid_won_a / .grid_won_b         — float32 arrays
        .grid_w / .grid_h / .scale_factor — grid metadata
    """

    SCALE_FACTOR: float = 0.1

    def __init__(self) -> None:
        self.scale_factor = self.SCALE_FACTOR
        self.grid_w = int(PitchConfig.PIXEL_WIDTH * self.scale_factor)
        self.grid_h = int(PitchConfig.PIXEL_HEIGHT * self.scale_factor)

        self.grid_a     = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        self.grid_b     = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        self.grid_total = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        self.grid_won_a = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        self.grid_won_b = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)

        self._total_duels: int = 0

    # ------------------------------------------------------------------
    # Accumulation
    # ------------------------------------------------------------------

    def record(self, duel: DuelEvent) -> None:
        """Register one duel event."""
        if duel.location is None:
            return

        gx = int(duel.location.x * PitchConfig.X_SCALE * self.scale_factor)
        gy = int(duel.location.y * PitchConfig.Y_SCALE * self.scale_factor)
        gx = max(0, min(gx, self.grid_w - 1))
        gy = max(0, min(gy, self.grid_h - 1))

        self.grid_total[gy, gx] += 1.0
        self._total_duels += 1

        # Per-team grids (increment whichever team is involved)
        if duel.team_a in (TeamID.A,):
            self.grid_a[gy, gx] += 1.0
        if duel.team_b in (TeamID.B,):
            self.grid_b[gy, gx] += 1.0

    def reset(self) -> None:
        for grid in (self.grid_a, self.grid_b, self.grid_total,
                     self.grid_won_a, self.grid_won_b):
            grid[:] = 0.0
        self._total_duels = 0

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        return {
            "total_duels": self._total_duels,
            "duels_a": int(self.grid_a.sum()),
            "duels_b": int(self.grid_b.sum()),
        }
