"""
Project: Tactix
File Created: 2026-02-11
Author: Xingnan Zhu
File Name: zone_14.py  (overlay renderer)
Description:
    RGBA overlay renderer for ZoneAnalyzer analytics data.

    Draws on the minimap:
      - Semi-transparent zone rectangle
      - Internal frequency heatmap (hot colormap)
      - Zone boundary outline + pass count label
"""

import cv2
import numpy as np

from tactix.analytics.attacking.zone_analyzer import ZoneAnalyzer
from tactix.core.types import TeamID

_PITCH_W = 105.0
_PITCH_H = 68.0


class Zone14Overlay:
    """Stateless renderer — call render() to produce an RGBA overlay."""

    @staticmethod
    def render(
        data: ZoneAnalyzer,
        canvas_w: int,
        canvas_h: int,
        pitch_w: float = _PITCH_W,
        pitch_h: float = _PITCH_H,
    ) -> np.ndarray:
        overlay = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
        x_min, x_max, y_min, y_max = data.zone

        px_min = int(x_min / pitch_w * canvas_w)
        px_max = int(x_max / pitch_w * canvas_w)
        py_min = int(y_min / pitch_h * canvas_h)
        py_max = int(y_max / pitch_h * canvas_h)
        zone_w = max(px_max - px_min, 1)
        zone_h = max(py_max - py_min, 1)

        # Semi-transparent fill
        cv2.rectangle(overlay, (px_min, py_min), (px_max, py_max), (255, 255, 200, 40), -1)

        # Internal frequency heatmap
        combined = data.grid[TeamID.A] + data.grid[TeamID.B]
        if combined.max() > 0:
            norm = (combined / combined.max() * 200).astype(np.uint8)
            heat_small = cv2.resize(norm, (zone_w, zone_h), interpolation=cv2.INTER_LINEAR)
            heat_rgb = cv2.applyColorMap(heat_small, cv2.COLORMAP_HOT)
            heat_bgra = cv2.cvtColor(heat_rgb, cv2.COLOR_BGR2BGRA)
            heat_bgra[:, :, 3] = (heat_small * 0.7).astype(np.uint8)
            overlay[py_min:py_max, px_min:px_max] = cv2.addWeighted(
                overlay[py_min:py_max, px_min:px_max], 0.3,
                heat_bgra, 0.7, 0
            )

        # Border
        cv2.rectangle(overlay, (px_min, py_min), (px_max, py_max), (255, 255, 0, 200), 2)

        # Label
        a = data.counts[TeamID.A]
        b = data.counts[TeamID.B]
        cv2.putText(
            overlay, f"Z14 A:{a} B:{b}",
            (px_min + 2, py_min - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.3,
            (255, 255, 0, 220), 1, cv2.LINE_AA
        )

        return overlay
