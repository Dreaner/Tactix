"""
Project: Tactix
File Created: 2026-02-11
Author: Xingnan Zhu
File Name: team_width_length.py  (overlay renderer)
Description:
    Team width & length (bounding box) overlay renderer.
    Draws axis-aligned rectangles + dimension labels for each team.
"""

import cv2
import numpy as np

from tactix.config import Colors
from tactix.core.types import FrameData, PitchConfig, TeamID


class WidthLengthOverlay:
    """Stateless renderer — call render() to produce an RGBA overlay."""

    @staticmethod
    def render(frame_data: FrameData) -> np.ndarray:
        w, h = PitchConfig.PIXEL_WIDTH, PitchConfig.PIXEL_HEIGHT
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        WidthLengthOverlay._draw(frame_data, TeamID.A, overlay, Colors.to_bgr(Colors.TEAM_A))
        WidthLengthOverlay._draw(frame_data, TeamID.B, overlay, Colors.to_bgr(Colors.TEAM_B))
        return overlay

    @staticmethod
    def _draw(frame_data: FrameData, team: TeamID, overlay: np.ndarray, color: tuple) -> None:
        xs = [p.pitch_position.x for p in frame_data.players if p.team == team and p.pitch_position]
        ys = [p.pitch_position.y for p in frame_data.players if p.team == team and p.pitch_position]
        if len(xs) < 2:
            return

        px1 = int(min(xs) * PitchConfig.X_SCALE)
        py1 = int(min(ys) * PitchConfig.Y_SCALE)
        px2 = int(max(xs) * PitchConfig.X_SCALE)
        py2 = int(max(ys) * PitchConfig.Y_SCALE)

        cv2.rectangle(overlay, (px1, py1), (px2, py2), (*color, 150), 1, cv2.LINE_AA)

        length_m = max(xs) - min(xs)
        width_m = max(ys) - min(ys)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(overlay, f"{length_m:.1f}m", ((px1 + px2) // 2 - 20, py2 + 15), font, 0.5, (*color, 255), 1)
        cv2.putText(overlay, f"{width_m:.1f}m", (px2 + 5, (py1 + py2) // 2), font, 0.5, (*color, 255), 1)
