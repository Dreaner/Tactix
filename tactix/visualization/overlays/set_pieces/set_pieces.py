"""
Project: Tactix
File Created: 2026-02-12
Author: Xingnan Zhu
File Name: set_pieces.py  (overlay renderer)
Description:
    Combined RGBA overlay renderer for set-piece analytics data.

    Corner kicks:
        Dot at corner location, color = team, size = outcome severity.
        ● filled + white ring = goal
        ○ ring               = shot
        · small dot          = other

    Free kicks:
        Dot at FK location, color = team.
        Size proportional to xG estimate.
        ● filled + white ring = goal | ○ ring = shot | · = no shot

    Both types label with a count badge when multiple events cluster.
"""

import cv2
import numpy as np

from tactix.analytics.set_pieces.set_piece_analyzer import (
    CornerAnalyzer,
    FreeKickAnalyzer,
)
from tactix.config import Colors
from tactix.core.types import TeamID

_PITCH_W = 105.0
_PITCH_H = 68.0


class SetPiecesOverlay:
    """Stateless renderer — call render() to produce an RGBA overlay."""

    @staticmethod
    def render(
        corners: CornerAnalyzer,
        free_kicks: FreeKickAnalyzer,
        canvas_w: int,
        canvas_h: int,
        pitch_w: float = _PITCH_W,
        pitch_h: float = _PITCH_H,
    ) -> np.ndarray:
        overlay = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)

        def to_px(pt) -> tuple:
            return (int(pt.x / pitch_w * canvas_w), int(pt.y / pitch_h * canvas_h))

        # ── Corner kicks ──────────────────────────────────────────────
        for seq in corners.sequences:
            px, py = to_px(seq.location)
            bgr = Colors.to_bgr(Colors.TEAM_A if seq.team == TeamID.A else Colors.TEAM_B)
            if seq.outcome == "goal":
                cv2.circle(overlay, (px, py), 8, (*bgr, 220), -1)
                cv2.circle(overlay, (px, py), 8, (255, 255, 255, 255), 2)
            elif seq.shot_taken:
                cv2.circle(overlay, (px, py), 6, (*bgr, 180), 2)
            else:
                cv2.circle(overlay, (px, py), 3, (*bgr, 140), -1)

        # Active corner marker
        if corners.active is not None:
            px, py = to_px(corners.active.location)
            bgr = Colors.to_bgr(Colors.TEAM_A if corners.active.team == TeamID.A else Colors.TEAM_B)
            cv2.circle(overlay, (px, py), 10, (*bgr, 255), 2)
            cv2.putText(overlay, "CK", (px + 8, py - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (*bgr, 255), 1, cv2.LINE_AA)

        # ── Free kicks ────────────────────────────────────────────────
        for seq in free_kicks.sequences:
            px, py = to_px(seq.location)
            bgr = Colors.to_bgr(Colors.TEAM_A if seq.team == TeamID.A else Colors.TEAM_B)
            # Size proportional to xG (min 4, max 12)
            radius = max(4, min(12, int(4 + seq.xg * 80)))
            if seq.outcome == "goal":
                cv2.circle(overlay, (px, py), radius, (*bgr, 220), -1)
                cv2.circle(overlay, (px, py), radius, (255, 255, 255, 255), 2)
            elif seq.shot_taken:
                cv2.circle(overlay, (px, py), radius, (*bgr, 180), 2)
            else:
                cv2.circle(overlay, (px, py), 3, (*bgr, 120), -1)

        # Active free kick marker
        if free_kicks.active is not None:
            px, py = to_px(free_kicks.active.location)
            bgr = Colors.to_bgr(Colors.TEAM_A if free_kicks.active.team == TeamID.A else Colors.TEAM_B)
            cv2.circle(overlay, (px, py), 10, (*bgr, 255), 2)
            cv2.putText(overlay, "FK", (px + 8, py - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (*bgr, 255), 1, cv2.LINE_AA)

        return overlay
