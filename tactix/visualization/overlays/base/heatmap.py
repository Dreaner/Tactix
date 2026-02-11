"""
Project: Tactix
File Created: 2026-02-11
Author: Xingnan Zhu
File Name: heatmap.py  (overlay renderer)
Description:
    RGBA overlay renderer for HeatmapAccumulator data.
    Applies Gaussian blur and JET colormap to the accumulated grids.
"""

import cv2
import numpy as np

from tactix.analytics.base.heatmap import HeatmapAccumulator
from tactix.core.types import PitchConfig, TeamID


class HeatmapOverlay:
    """Stateless renderer — call render() to produce an RGBA overlay."""

    @staticmethod
    def render(
        data: HeatmapAccumulator,
        team: TeamID = None,
    ) -> np.ndarray:
        """
        Returns RGBA ndarray (H×W×4).

        Args:
            data: HeatmapAccumulator instance with accumulated grids.
            team: If None, combine both teams. Pass TeamID.A or TeamID.B
                  for a single-team view.
        """
        if team == TeamID.A:
            grid = data.heatmap_a
        elif team == TeamID.B:
            grid = data.heatmap_b
        else:
            grid = data.heatmap_a + data.heatmap_b

        max_val = np.max(grid)
        if max_val == 0:
            return np.zeros(
                (PitchConfig.PIXEL_HEIGHT, PitchConfig.PIXEL_WIDTH, 4), dtype=np.uint8
            )

        norm_grid = (grid / max_val * 255).astype(np.uint8)
        blurred = cv2.GaussianBlur(norm_grid, (0, 0), sigmaX=3, sigmaY=3)
        colored = cv2.applyColorMap(blurred, cv2.COLORMAP_JET)

        alpha = blurred.copy()
        alpha[alpha < 30] = 0
        alpha = cv2.normalize(alpha, None, 0, 180, cv2.NORM_MINMAX)

        rgba_small = np.dstack((colored, alpha))
        rgba_full = cv2.resize(
            rgba_small,
            (PitchConfig.PIXEL_WIDTH, PitchConfig.PIXEL_HEIGHT),
            interpolation=cv2.INTER_LINEAR,
        )
        return rgba_full
