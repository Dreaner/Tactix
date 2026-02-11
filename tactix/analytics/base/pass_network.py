"""
Project: Tactix
File Created: 2026-02-02
Author: Xingnan Zhu
File Name: pass_network.py
Description:
    Identifies the ball carrier and computes pass-line candidates to
    teammates. Returns a list of (start_px, end_px, opacity) tuples;
    the caller is responsible for drawing them.

    Also writes ball.owner_id on each call.
"""

from typing import List, Optional, Tuple

import numpy as np

from tactix.core.types import FrameData, Player


class PassNetwork:
    def __init__(self, max_pass_dist: int = 300, ball_owner_dist: int = 50) -> None:
        self.max_pass_dist = max_pass_dist
        self.ball_owner_dist = ball_owner_dist

    def analyze(
        self, frame_data: FrameData
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int], float]]:
        """
        Returns [(start_xy, end_xy, opacity), ...] for potential pass lines.
        Side-effect: sets frame_data.ball.owner_id.
        """
        if not frame_data.ball or not frame_data.players:
            return []

        ball_center = np.array(frame_data.ball.center)
        owner: Optional[Player] = None
        min_dist = float("inf")

        for p in frame_data.players:
            dist = np.linalg.norm(np.array(p.anchor) - ball_center)
            if dist < min_dist:
                min_dist = dist
                owner = p

        frame_data.ball.owner_id = owner.id if owner else None

        if owner is None or min_dist > self.ball_owner_dist:
            return []

        teammates = [
            p for p in frame_data.players
            if p.team == owner.team and p.id != owner.id
        ]

        lines = []
        for mate in teammates:
            dist = np.linalg.norm(np.array(owner.anchor) - np.array(mate.anchor))
            if dist < self.max_pass_dist:
                opacity = max(0.2, 1.0 - dist / self.max_pass_dist)
                lines.append((owner.anchor, mate.anchor, opacity))

        return lines
