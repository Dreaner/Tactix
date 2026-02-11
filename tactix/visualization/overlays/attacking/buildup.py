"""
Project: Tactix
File Created: 2026-02-11
Author: Xingnan Zhu
File Name: buildup.py  (overlay renderer)
Description:
    RGBA overlay renderer for BuildupTracker analytics data.

    Draws recent completed build-up sequences as fading arrow chains,
    plus the currently active build-up in full brightness.
"""

from typing import Tuple

import cv2
import numpy as np

from tactix.analytics.attacking.buildup_tracker import BuildupSequence, BuildupTracker, BuildupState
from tactix.config import Colors
from tactix.core.types import Point, TeamID

_PITCH_W = 105.0
_PITCH_H = 68.0
_HISTORY_DISPLAY = 5


class BuildupOverlay:
    """Stateless renderer — call render() to produce an RGBA overlay."""

    @staticmethod
    def render(
        data: BuildupTracker,
        canvas_w: int,
        canvas_h: int,
        pitch_w: float = _PITCH_W,
        pitch_h: float = _PITCH_H,
    ) -> np.ndarray:
        overlay = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)

        def to_px(pt: Point) -> Tuple[int, int]:
            return (int(pt.x / pitch_w * canvas_w), int(pt.y / pitch_h * canvas_h))

        def draw_sequence(seq: BuildupSequence, alpha: int) -> None:
            if len(seq.waypoints) < 2:
                return
            team_rgb = Colors.TEAM_A if seq.team == TeamID.A else Colors.TEAM_B
            bgr = Colors.to_bgr(team_rgb)
            pts = [to_px(w) for w in seq.waypoints]
            for i in range(len(pts) - 1):
                cv2.arrowedLine(
                    overlay, pts[i], pts[i + 1],
                    (*bgr, alpha), 1, cv2.LINE_AA, tipLength=0.3
                )

        # Recent completed sequences (fading)
        recent = data.sequences[-_HISTORY_DISPLAY:]
        for idx, seq in enumerate(recent):
            fade = int(60 + 80 * (idx + 1) / max(len(recent), 1))
            draw_sequence(seq, fade)

        # Active build-up in full brightness
        if data.state == BuildupState.BUILDING and data.active is not None:
            draw_sequence(data.active, 220)

        return overlay
