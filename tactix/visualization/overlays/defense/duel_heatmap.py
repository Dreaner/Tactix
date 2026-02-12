"""
Project: Tactix
File Created: 2026-02-12
Author: Xingnan Zhu
File Name: duel_heatmap.py  (overlay renderer)
Description:
    RGBA overlay renderer for DuelHeatmap analytics data.

    Renders the total-duel grid with a HOT colormap (red = high frequency).
    If winner data is available, also draws a subtle win-ratio tint per cell:
        Team A dominance → blue shift
        Team B dominance → red shift
"""

import cv2
import numpy as np

from tactix.analytics.defense.duel_heatmap import DuelHeatmap
from tactix.core.types import PitchConfig


class DuelHeatmapOverlay:
    """Stateless renderer — call render() to produce an RGBA overlay."""

    @staticmethod
    def render(data: DuelHeatmap) -> np.ndarray:
        w, h = PitchConfig.PIXEL_WIDTH, PitchConfig.PIXEL_HEIGHT

        if data.grid_total.max() == 0:
            return np.zeros((h, w, 4), dtype=np.uint8)

        # ── Base heatmap (HOT colormap) ────────────────────────────────
        norm = (data.grid_total / data.grid_total.max() * 255).astype(np.uint8)
        blurred = cv2.GaussianBlur(norm, (0, 0), sigmaX=3, sigmaY=3)
        colored = cv2.applyColorMap(blurred, cv2.COLORMAP_HOT)

        # Alpha: transparent where there are no duels
        alpha = blurred.copy()
        alpha[alpha < 20] = 0
        alpha = cv2.normalize(alpha, None, 0, 200, cv2.NORM_MINMAX)

        rgba_small = np.dstack((colored, alpha))

        # ── Team dominance tint (A vs B involvement ratio) ────────────
        duel_sum = data.grid_a + data.grid_b
        if duel_sum.max() > 0:
            # ratio in [-1, 1]: +1 = A dominates, -1 = B dominates
            ratio = np.where(
                duel_sum > 0,
                (data.grid_a - data.grid_b) / duel_sum,
                0.0
            )
            # Team A → blue tint, Team B → green tint (distinct from HOT red)
            tint = np.zeros((*ratio.shape, 3), dtype=np.uint8)
            tint[ratio > 0.2, 0] = 180   # blue channel for A
            tint[ratio < -0.2, 1] = 180  # green channel for B
            tint_alpha = (np.abs(ratio) * 80).astype(np.uint8)
            tint_rgba = np.dstack((tint, tint_alpha))
            rgba_small = cv2.addWeighted(rgba_small, 0.8, tint_rgba, 0.2, 0)

        # ── Scale to canvas ────────────────────────────────────────────
        return cv2.resize(rgba_small, (w, h), interpolation=cv2.INTER_LINEAR)
