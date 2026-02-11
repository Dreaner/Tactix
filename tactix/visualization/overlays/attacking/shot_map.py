"""
Project: Tactix
File Created: 2026-02-11
Author: Xingnan Zhu
File Name: shot_map.py  (overlay renderer)
Description:
    RGBA overlay renderer for ShotMap analytics data.

    Draws each recorded shot on the minimap canvas:
      - Filled circle + white border = goal
      - Hollow circle + dot          = on-target (saved)
      - X mark                       = off-target / blocked / unknown
"""

import cv2
import numpy as np

from tactix.analytics.attacking.shot_map import ShotMap
from tactix.config import Colors
from tactix.core.types import TeamID

_PITCH_W = 105.0
_PITCH_H = 68.0


class ShotMapOverlay:
    """Stateless renderer — call render() to produce an RGBA overlay."""

    @staticmethod
    def render(
        data: ShotMap,
        canvas_w: int,
        canvas_h: int,
        pitch_w: float = _PITCH_W,
        pitch_h: float = _PITCH_H,
    ) -> np.ndarray:
        """
        Returns an RGBA ndarray (H×W×4) with all shots drawn.
        Caller composites this onto the minimap canvas.
        """
        overlay = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)

        for shot in data.shots:
            px = int(shot.x / pitch_w * canvas_w)
            py = int(shot.y / pitch_h * canvas_h)

            team_rgb = Colors.TEAM_A if shot.team == TeamID.A else Colors.TEAM_B
            bgr = Colors.to_bgr(team_rgb)

            if shot.outcome == "goal":
                cv2.circle(overlay, (px, py), 7, (*bgr, 220), -1)
                cv2.circle(overlay, (px, py), 7, (255, 255, 255, 255), 2)
            elif shot.on_target:
                cv2.circle(overlay, (px, py), 6, (*bgr, 200), 2)
                cv2.circle(overlay, (px, py), 3, (*bgr, 200), -1)
            else:
                r = 5
                cv2.line(overlay, (px - r, py - r), (px + r, py + r), (*bgr, 160), 2)
                cv2.line(overlay, (px + r, py - r), (px - r, py + r), (*bgr, 160), 2)

        return overlay
