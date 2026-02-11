"""
Project: Tactix
File Created: 2026-02-11
Author: Xingnan Zhu
File Name: pass_sonar.py
Description:
    Pure-data accumulator for per-player directional pass statistics.
    No rendering logic — see tactix.visualization.overlays.attacking.pass_sonar
    for the corresponding RGBA overlay renderer.

    Maintains 8-sector (45° each) pass counts per tracker_id, plus the
    player's last known pitch position for the renderer to use.
"""

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from tactix.core.events import PassEvent
from tactix.core.types import FrameData, TeamID

_NUM_SECTORS = 8
_SECTOR_DEG = 360.0 / _NUM_SECTORS


def _angle_to_sector(angle_deg: float) -> int:
    """Map a bearing in degrees to one of 8 sectors (0 = right, CCW)."""
    angle_deg = angle_deg % 360.0
    return int((angle_deg + _SECTOR_DEG / 2) / _SECTOR_DEG) % _NUM_SECTORS


class PassSonar:
    """
    Accumulates per-player pass-direction counts and caches pitch positions.

    Public attributes for the renderer:
        .counts    — Dict[int, List[int]]            tracker_id → [count_s0…s7]
        .positions — Dict[int, Tuple[float,float,TeamID]]  tracker_id → (x_m, y_m, team)
    """

    NUM_SECTORS: int = _NUM_SECTORS

    def __init__(self) -> None:
        self.counts: Dict[int, List[int]] = defaultdict(lambda: [0] * _NUM_SECTORS)
        # Stores pitch coords in meters — NO canvas dependency
        self.positions: Dict[int, Tuple[float, float, TeamID]] = {}

    # ------------------------------------------------------------------
    # Accumulation
    # ------------------------------------------------------------------

    def record(self, pass_event: PassEvent) -> None:
        """Register the direction of one pass for the sender."""
        if pass_event.origin is None or pass_event.destination is None:
            return
        sector = _angle_to_sector(pass_event.angle_deg)
        self.counts[pass_event.sender_id][sector] += 1

    def update_positions(self, frame_data: FrameData) -> None:
        """Cache current pitch positions (meters) of all tracked players."""
        for p in frame_data.players:
            if p.pitch_position is None or p.id == -1:
                continue
            self.positions[p.id] = (p.pitch_position.x, p.pitch_position.y, p.team)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def top_directions(self, player_id: int) -> List[Tuple[int, int]]:
        """Returns (sector, count) pairs sorted descending by count."""
        counts = self.counts.get(player_id, [0] * _NUM_SECTORS)
        return sorted(enumerate(counts), key=lambda x: x[1], reverse=True)
