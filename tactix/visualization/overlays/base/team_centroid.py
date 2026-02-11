"""
Project: Tactix
File Created: 2026-02-11
Author: Xingnan Zhu
File Name: team_centroid.py  (overlay renderer)
Description:
    Team centroid (geometric mean position) overlay renderer.
    Draws an X marker + circle at each team's average pitch position.
"""

import cv2
import numpy as np

from tactix.config import Colors
from tactix.core.types import FrameData, PitchConfig, TeamID


class CentroidOverlay:
    """Stateless renderer — call render() to produce an RGBA overlay."""

    @staticmethod
    def render(frame_data: FrameData) -> np.ndarray:
        w, h = PitchConfig.PIXEL_WIDTH, PitchConfig.PIXEL_HEIGHT
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        CentroidOverlay._draw(frame_data, TeamID.A, overlay, Colors.to_bgr(Colors.TEAM_A))
        CentroidOverlay._draw(frame_data, TeamID.B, overlay, Colors.to_bgr(Colors.TEAM_B))
        return overlay

    @staticmethod
    def _draw(frame_data: FrameData, team: TeamID, overlay: np.ndarray, color: tuple) -> None:
        positions = [
            (p.pitch_position.x, p.pitch_position.y)
            for p in frame_data.players
            if p.team == team and p.pitch_position
        ]
        if not positions:
            return

        avg_x = sum(x for x, _ in positions) / len(positions)
        avg_y = sum(y for _, y in positions) / len(positions)
        px = int(avg_x * PitchConfig.X_SCALE)
        py = int(avg_y * PitchConfig.Y_SCALE)

        m = 20
        cv2.line(overlay, (px - m, py - m), (px + m, py + m), (*color, 255), 3, cv2.LINE_AA)
        cv2.line(overlay, (px + m, py - m), (px - m, py + m), (*color, 255), 3, cv2.LINE_AA)
        cv2.circle(overlay, (px, py), m + 5, (*color, 255), 2, cv2.LINE_AA)
