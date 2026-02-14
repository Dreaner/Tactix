"""
Project: Tactix
File Created: 2026-02-14
Author: Xingnan Zhu
File Name: formation.py
Description:
    RGBA overlay renderer for formation labels on the minimap.
    Pure rendering — no tactical logic.
    See tactix.analytics.formation.formation_detector for the data accumulator.
"""

import cv2
import numpy as np

from tactix.analytics.formation.formation_detector import FormationDetector
from tactix.config import Colors
from tactix.core.types import TeamID


class FormationOverlay:
    """Renders formation labels for both teams as text on a transparent canvas."""

    @staticmethod
    def render(data: FormationDetector,
               canvas_w: int, canvas_h: int) -> np.ndarray:
        """
        Draw formation labels on a transparent RGBA canvas.

        Team A label at top-left, Team B label at top-right.
        Includes confidence percentage if < 100%.

        :param data: FormationDetector instance with .current and .confidence
        :param canvas_w: Minimap canvas width in pixels
        :param canvas_h: Minimap canvas height in pixels
        :return: RGBA numpy array (canvas_h × canvas_w × 4)
        """
        canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 2
        margin = 20

        # --- Team A (top-left) ---
        formation_a = data.current.get(TeamID.A, "Unknown")
        conf_a = data.confidence.get(TeamID.A, 0.0)
        label_a = f"A: {formation_a}"
        if conf_a < 1.0:
            label_a += f" ({int(conf_a * 100)}%)"

        color_a = Colors.TEAM_A  # RGB
        _draw_label(canvas, label_a, (margin, margin + 30),
                    font, font_scale, thickness, color_a)

        # --- Team B (top-right) ---
        formation_b = data.current.get(TeamID.B, "Unknown")
        conf_b = data.confidence.get(TeamID.B, 0.0)
        label_b = f"B: {formation_b}"
        if conf_b < 1.0:
            label_b += f" ({int(conf_b * 100)}%)"

        color_b = Colors.TEAM_B  # RGB
        (tw, _), _ = cv2.getTextSize(label_b, font, font_scale, thickness)
        x_b = canvas_w - margin - tw
        _draw_label(canvas, label_b, (x_b, margin + 30),
                    font, font_scale, thickness, color_b)

        return canvas


def _draw_label(canvas: np.ndarray,
                text: str,
                org: tuple,
                font: int,
                font_scale: float,
                thickness: int,
                color_rgb: tuple) -> None:
    """
    Draw text with a dark shadow for readability on the RGBA canvas.
    Color is provided as RGB; converted to BGR for cv2.putText.
    Alpha channel is set to 255 on drawn pixels.
    """
    x, y = org
    # Shadow (offset by 2px)
    cv2.putText(canvas, text, (x + 2, y + 2), font, font_scale,
                (0, 0, 0, 220), thickness + 1, cv2.LINE_AA)
    # Main text — write into BGR channels, set alpha to 255
    bgr = (color_rgb[2], color_rgb[1], color_rgb[0], 255)
    cv2.putText(canvas, text, (x, y), font, font_scale,
                bgr, thickness, cv2.LINE_AA)
