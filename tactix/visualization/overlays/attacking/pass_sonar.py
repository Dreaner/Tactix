"""
Project: Tactix
File Created: 2026-02-11
Author: Xingnan Zhu
File Name: pass_sonar.py  (overlay renderer)
Description:
    RGBA overlay renderer for PassSonar analytics data.

    For each player with recorded passes, draws 8 radial spokes around their
    minimap position. Spoke length is proportional to pass frequency in that
    direction. Color matches team color.
"""

import math

import cv2
import numpy as np

from tactix.analytics.attacking.pass_sonar import PassSonar
from tactix.config import Colors
from tactix.core.types import TeamID

_PITCH_W = 105.0
_PITCH_H = 68.0
_NUM_SECTORS = 8
_SECTOR_DEG = 360.0 / _NUM_SECTORS
_MAX_SPOKE_PX = 20


class PassSonarOverlay:
    """Stateless renderer — call render() to produce an RGBA overlay."""

    @staticmethod
    def render(
        data: PassSonar,
        canvas_w: int,
        canvas_h: int,
        pitch_w: float = _PITCH_W,
        pitch_h: float = _PITCH_H,
    ) -> np.ndarray:
        overlay = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)

        for pid, counts in data.counts.items():
            if pid not in data.positions:
                continue
            x_m, y_m, team = data.positions[pid]
            # Convert pitch metres → canvas pixels here (not in analytics)
            cx = int(x_m / pitch_w * canvas_w)
            cy = int(y_m / pitch_h * canvas_h)

            total = sum(counts)
            if total == 0:
                continue

            team_rgb = Colors.TEAM_A if team == TeamID.A else Colors.TEAM_B
            bgr = Colors.to_bgr(team_rgb)
            max_count = max(counts)

            for sector, count in enumerate(counts):
                if count == 0:
                    continue
                length = int(count / max_count * _MAX_SPOKE_PX)
                angle_rad = math.radians(sector * _SECTOR_DEG)
                ex = cx + int(math.cos(angle_rad) * length)
                ey = cy - int(math.sin(angle_rad) * length)  # Y flipped for screen
                alpha = min(160 + int(count / max_count * 95), 255)
                cv2.line(overlay, (cx, cy), (ex, ey), (*bgr, alpha), 1, cv2.LINE_AA)

        return overlay
