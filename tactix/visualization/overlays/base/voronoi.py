"""
Project: Tactix
File Created: 2026-02-11
Author: Xingnan Zhu
File Name: voronoi.py  (overlay renderer)
Description:
    Voronoi space-control overlay renderer.
    Computes and draws Voronoi facets colored by team.
"""

import cv2
import numpy as np

from tactix.config import Colors
from tactix.core.types import FrameData, PitchConfig, TeamID


class VoronoiOverlay:
    """Stateless renderer — call render() to produce an RGBA overlay."""

    @staticmethod
    def render(frame_data: FrameData) -> np.ndarray:
        w, h = PitchConfig.PIXEL_WIDTH, PitchConfig.PIXEL_HEIGHT
        overlay = np.zeros((h, w, 4), dtype=np.uint8)

        points, colors = [], []
        COLOR_A = Colors.to_bgr(Colors.TEAM_A)
        COLOR_B = Colors.to_bgr(Colors.TEAM_B)

        for p in frame_data.players:
            if p.pitch_position and p.team in (TeamID.A, TeamID.B):
                px = int(np.clip(p.pitch_position.x, 0, w - 1))
                py = int(np.clip(p.pitch_position.y, 0, h - 1))
                points.append([px, py])
                colors.append(COLOR_A if p.team == TeamID.A else COLOR_B)

        if len(points) < 4:
            return overlay

        subdiv = cv2.Subdiv2D((0, 0, w, h))
        for pt in points:
            subdiv.insert((float(pt[0]), float(pt[1])))

        facets, centers = subdiv.getVoronoiFacetList([])

        for i, facet in enumerate(facets):
            cx, cy = centers[i]
            best = min(range(len(points)), key=lambda j: (points[j][0]-cx)**2 + (points[j][1]-cy)**2)
            color = colors[best]
            poly = np.array(facet, dtype=np.int32)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [poly], 255)
            overlay[mask == 255] = (*color, 80)
            cv2.polylines(overlay, [poly], True, (255, 255, 255, 150), 1, cv2.LINE_AA)

        return overlay
