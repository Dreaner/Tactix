"""
Project: Tactix
File Created: 2026-02-12
Author: Xingnan Zhu
File Name: transition.py  (overlay renderer)
Description:
    RGBA overlay renderer for TransitionTracker data.

    ATTACK sequences (recovery → shot/loss):
        Drawn as bright arrow chains in team color.
        Outcome marker at the final waypoint:
          ★ = shot | ✦ = goal | ✕ = lost | ⊙ = timeout

    DEFEND sequences (ball-loss → recovery):
        Drawn as dashed lines (lower alpha) in team color.

    Only the most recent HISTORY_DISPLAY sequences of each type are shown;
    older ones fade by alpha.
"""

from typing import Tuple

import cv2
import numpy as np

from tactix.analytics.transition.transition_tracker import (
    TransitionPhase,
    TransitionSequence,
    TransitionTracker,
)
from tactix.config import Colors
from tactix.core.types import Point, TeamID

_PITCH_W = 105.0
_PITCH_H = 68.0
_HISTORY = 4   # max completed sequences per (team × phase) to display


class TransitionOverlay:
    """Stateless renderer — call render() to produce an RGBA overlay."""

    @staticmethod
    def render(
        data: TransitionTracker,
        canvas_w: int,
        canvas_h: int,
        pitch_w: float = _PITCH_W,
        pitch_h: float = _PITCH_H,
    ) -> np.ndarray:
        overlay = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)

        def to_px(pt: Point) -> Tuple[int, int]:
            return (int(pt.x / pitch_w * canvas_w), int(pt.y / pitch_h * canvas_h))

        def draw_sequence(seq: TransitionSequence, alpha: int) -> None:
            if len(seq.waypoints) < 2:
                return
            team_rgb = Colors.TEAM_A if seq.team == TeamID.A else Colors.TEAM_B
            bgr = Colors.to_bgr(team_rgb)
            pts = [to_px(w) for w in seq.waypoints]

            if seq.phase == TransitionPhase.ATTACK:
                for i in range(len(pts) - 1):
                    cv2.arrowedLine(
                        overlay, pts[i], pts[i + 1],
                        (*bgr, alpha), 1, cv2.LINE_AA, tipLength=0.3
                    )
                # Outcome marker at last point
                px, py = pts[-1]
                marker_color = (*bgr, min(alpha + 40, 255))
                if seq.outcome == "goal":
                    cv2.circle(overlay, (px, py), 6, marker_color, -1)
                    cv2.circle(overlay, (px, py), 6, (255, 255, 255, 220), 2)
                elif seq.outcome == "shot":
                    cv2.circle(overlay, (px, py), 5, marker_color, 2)
                else:  # lost / timeout
                    r = 4
                    cv2.line(overlay, (px - r, py - r), (px + r, py + r), marker_color, 1)
                    cv2.line(overlay, (px + r, py - r), (px - r, py + r), marker_color, 1)
            else:
                # DEFEND: dashed appearance via short segments
                defend_alpha = max(alpha - 40, 30)
                for i in range(len(pts) - 1):
                    if i % 2 == 0:  # skip every other segment for dashed look
                        cv2.line(overlay, pts[i], pts[i + 1], (*bgr, defend_alpha), 1, cv2.LINE_AA)

        # ── Draw completed sequences (fading) ──────────────────
        for team in (TeamID.A, TeamID.B):
            for phase in TransitionPhase:
                recent = [
                    s for s in data.sequences
                    if s.team == team and s.phase == phase
                ][-_HISTORY:]
                n = len(recent)
                for idx, seq in enumerate(recent):
                    fade = int(50 + 130 * (idx + 1) / max(n, 1))
                    draw_sequence(seq, fade)

        # ── Draw active sequences (bright) ─────────────────────
        for team in (TeamID.A, TeamID.B):
            for phase, store in (
                (TransitionPhase.ATTACK, data.active_attack),
                (TransitionPhase.DEFEND, data.active_defend),
            ):
                seq = store.get(team)
                if seq is not None:
                    draw_sequence(seq, 210)

        return overlay
