"""
Project: Tactix
File Created: 2026-02-11
Author: Xingnan Zhu
File Name: compactness.py  (overlay renderer)
Description:
    Team compactness (convex hull) overlay renderer.
"""

import cv2
import numpy as np
from scipy.spatial import ConvexHull

from tactix.config import Colors
from tactix.core.types import FrameData, PitchConfig, TeamID


class CompactnessOverlay:
    """Stateless renderer — call render() to produce an RGBA overlay."""

    @staticmethod
    def render(frame_data: FrameData) -> np.ndarray:
        w, h = PitchConfig.PIXEL_WIDTH, PitchConfig.PIXEL_HEIGHT
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        CompactnessOverlay._draw_hull(frame_data, TeamID.A, overlay, Colors.to_bgr(Colors.TEAM_A))
        CompactnessOverlay._draw_hull(frame_data, TeamID.B, overlay, Colors.to_bgr(Colors.TEAM_B))
        return overlay

    @staticmethod
    def _draw_hull(frame_data: FrameData, team: TeamID, overlay: np.ndarray, color: tuple) -> None:
        pts = np.array([
            [int(p.pitch_position.x), int(p.pitch_position.y)]
            for p in frame_data.players
            if p.team == team and p.pitch_position
        ])
        if len(pts) < 3:
            return
        try:
            hull = ConvexHull(pts)
            hull_pts = pts[hull.vertices].reshape((-1, 1, 2))
            mask = np.zeros((PitchConfig.PIXEL_HEIGHT, PitchConfig.PIXEL_WIDTH), dtype=np.uint8)
            cv2.fillPoly(mask, [hull_pts], 255)
            overlay[mask == 255] = (*color, 40)
            cv2.polylines(overlay, [hull_pts], True, (*color, 200), 2, cv2.LINE_AA)
        except Exception:
            pass
