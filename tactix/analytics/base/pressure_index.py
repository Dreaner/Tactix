"""
Project: Tactix
File Created: 2026-02-07
Author: Xingnan Zhu
File Name: pressure_index.py
Description:
    Calculates the Pressure Index for each player.
    Pressure is the weighted sum of opponents within pressure_radius meters,
    normalized to [0, 1]. Result is written to player.pressure in-place.

    No rendering — pressure is visualized by MinimapRenderer when
    show_pressure=True.
"""

import numpy as np

from tactix.core.types import FrameData, TeamID


class PressureIndex:
    def __init__(self, pressure_radius: float = 8.0) -> None:
        self.pressure_radius = pressure_radius

    def calculate(self, frame_data: FrameData) -> None:
        """Mutates player.pressure for every player in frame_data."""
        players = frame_data.players
        for p in players:
            if p.pitch_position is None or p.team in (TeamID.UNKNOWN, TeamID.REFEREE):
                p.pressure = 0.0
                continue

            opponents = [
                op for op in players
                if op.team != p.team
                and op.team in (TeamID.A, TeamID.B)
                and op.pitch_position
            ]

            score = 0.0
            for op in opponents:
                dist = np.sqrt(
                    (p.pitch_position.x - op.pitch_position.x) ** 2
                    + (p.pitch_position.y - op.pitch_position.y) ** 2
                )
                if dist < self.pressure_radius:
                    score += 1.0 / (dist + 0.5)

            p.pressure = min(score / 3.0, 1.0)
