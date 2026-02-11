"""
Project: Tactix
File Created: 2026-02-11
Author: Xingnan Zhu
File Name: zone_analyzer.py
Description:
    Pure-data accumulator for Zone 14 pass statistics.
    No rendering logic — see tactix.visualization.overlays.attacking.zone_14
    for the corresponding RGBA overlay renderer.

    Zone 14 = the central area directly in front of the penalty box:
        x: 77–88m, y: 25–43m (configurable via ZONE_14 in config)
"""

from typing import Dict, Tuple

import numpy as np

from tactix.config import Config
from tactix.core.events import PassEvent
from tactix.core.types import TeamID


class ZoneAnalyzer:
    """
    Records passes originating from Zone 14 and builds a per-team
    frequency grid (GRID × GRID cells) for the zone's interior.

    Public attributes for the renderer:
        .grid   — Dict[TeamID, np.ndarray]  frequency counts per cell
        .counts — Dict[TeamID, int]          total Zone 14 passes per team
        .zone   — Tuple[float,...]           (x_min, x_max, y_min, y_max) meters
    """

    GRID: int = 10   # resolution of internal frequency grid

    def __init__(self, cfg: Config) -> None:
        self.zone: Tuple[float, float, float, float] = cfg.ZONE_14
        self.grid: Dict[TeamID, np.ndarray] = {
            TeamID.A: np.zeros((self.GRID, self.GRID), dtype=np.float32),
            TeamID.B: np.zeros((self.GRID, self.GRID), dtype=np.float32),
        }
        self.counts: Dict[TeamID, int] = {TeamID.A: 0, TeamID.B: 0}

    # ------------------------------------------------------------------
    # Accumulation
    # ------------------------------------------------------------------

    def record(self, pass_event: PassEvent) -> None:
        """Register a pass if its origin falls inside Zone 14."""
        if pass_event.origin is None:
            return

        x, y = pass_event.origin.x, pass_event.origin.y
        x_min, x_max, y_min, y_max = self.zone
        if not (x_min <= x <= x_max and y_min <= y <= y_max):
            return

        team = pass_event.team
        if team not in self.grid:
            return

        gx = min(int((x - x_min) / (x_max - x_min) * self.GRID), self.GRID - 1)
        gy = min(int((y - y_min) / (y_max - y_min) * self.GRID), self.GRID - 1)
        self.grid[team][gy, gx] += 1
        self.counts[team] += 1

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, int]:
        return {"team_a": self.counts[TeamID.A], "team_b": self.counts[TeamID.B]}
