"""
Project: Tactix
File Created: 2026-02-10
Author: Xingnan Zhu
File Name: registry.py
Description:
    Persistent cross-frame state registries.

    PlayerRegistry: Maps tracker_id → team assignment across frames.
    Accumulates color evidence via voting so team labels stabilise
    after a few frames and survive short tracking gaps.

    BallStateTracker: Fills short detection gaps with linear extrapolation
    so downstream modules always receive a reasonable ball position.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Deque, List, Optional, Tuple

import numpy as np

from tactix.core.types import Ball, TeamID


# ==========================================
# 1. Player Registry
# ==========================================

@dataclass
class PlayerRecord:
    """Persistent state for one tracked player, keyed by ByteTrack tracker_id."""
    tracker_id: int
    team: TeamID = TeamID.UNKNOWN
    team_confidence: float = 0.0        # Ratio of majority votes (0.0 – 1.0)
    frames_seen: int = 0
    color_history: Deque = field(default_factory=lambda: deque(maxlen=20))
    team_vote_counts: Dict[TeamID, int] = field(default_factory=lambda: {TeamID.A: 0, TeamID.B: 0})
    confirmed: bool = False             # True once confidence threshold is met


class PlayerRegistry:
    """
    Cross-frame registry that maps tracker_id → PlayerRecord.

    The key improvement over the previous per-frame K-Means prediction:
    - Team assignment accumulates across frames via voting.
    - Once "confirmed" (enough frames + strong majority), the team is frozen
      and K-Means is no longer called for that player.
    - GK / Referee roles are hard-set via override_team() and skip voting.
    """

    # How many frames a player must be seen before their team is confirmed.
    CONFIRM_FRAMES: int = 5
    # Fraction of votes that must agree before the team is confirmed (0.0 – 1.0).
    CONFIRM_VOTE_RATIO: float = 0.70

    def __init__(self) -> None:
        self._records: Dict[int, PlayerRecord] = {}

    # ------------------------------------------------------------------
    # Record access
    # ------------------------------------------------------------------

    def get_or_create(self, tracker_id: int) -> PlayerRecord:
        """Returns the existing record for this ID, or creates a new one."""
        if tracker_id not in self._records:
            self._records[tracker_id] = PlayerRecord(tracker_id=tracker_id)
        return self._records[tracker_id]

    def get(self, tracker_id: int) -> Optional[PlayerRecord]:
        """Returns None if this ID has never been seen."""
        return self._records.get(tracker_id)

    # ------------------------------------------------------------------
    # Evidence accumulation
    # ------------------------------------------------------------------

    def record_color_sample(self, tracker_id: int, color: np.ndarray) -> None:
        """Appends a raw BGR color observation for this player."""
        record = self.get_or_create(tracker_id)
        record.color_history.append(color)
        record.frames_seen += 1

    def record_team_vote(self, tracker_id: int, team: TeamID) -> None:
        """
        Accumulates a K-Means prediction vote for this player.
        Recomputes confidence and updates the confirmed flag.
        """
        record = self.get_or_create(tracker_id)
        if team in (TeamID.A, TeamID.B):
            record.team_vote_counts[team] += 1

        total = sum(record.team_vote_counts.values())
        if total == 0:
            return

        best_team = max(record.team_vote_counts, key=record.team_vote_counts.get)
        best_count = record.team_vote_counts[best_team]
        ratio = best_count / total

        record.team = best_team
        record.team_confidence = ratio

        if record.frames_seen >= self.CONFIRM_FRAMES and ratio >= self.CONFIRM_VOTE_RATIO:
            record.confirmed = True

    def override_team(self, tracker_id: int, team: TeamID) -> None:
        """
        Hard-sets team for GOALKEEPER or REFEREE without going through voting.
        Marks the record as confirmed immediately.
        """
        record = self.get_or_create(tracker_id)
        record.team = team
        record.confirmed = True
        record.team_confidence = 1.0

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def is_confirmed(self, tracker_id: int) -> bool:
        record = self._records.get(tracker_id)
        return record is not None and record.confirmed

    def get_team(self, tracker_id: int) -> TeamID:
        record = self._records.get(tracker_id)
        if record is None:
            return TeamID.UNKNOWN
        return record.team

    def get_color_samples(self, tracker_id: int) -> List[np.ndarray]:
        record = self._records.get(tracker_id)
        return list(record.color_history) if record else []

    def all_records(self) -> Dict[int, PlayerRecord]:
        return self._records


# ==========================================
# 2. Ball State Tracker
# ==========================================

class BallStateTracker:
    """
    Fills short ball-detection gaps with forward-projected positions.

    When the ball disappears for ≤ MAX_GAP consecutive frames, the tracker
    returns a synthetic Ball whose position is extrapolated using the
    velocity estimated from the last two real detections.

    The synthetic Ball has score=0.0 as a sentinel; downstream code can
    check ball.score == 0.0 to distinguish real from interpolated detections.
    """

    MAX_GAP: int = 10   # Frames to tolerate before giving up on interpolation
    HISTORY_LEN: int = 10

    def __init__(self) -> None:
        self._history: Deque[Tuple[int, float, float]] = deque(maxlen=self.HISTORY_LEN)
        self._last_ball: Optional[Ball] = None
        self._frames_missing: int = 0

    def update(self, frame_index: int, ball: Optional[Ball]) -> Optional[Ball]:
        """
        Call every frame with the raw detector output.

        Returns:
            - The original ball if detected this frame.
            - A synthetic Ball if within MAX_GAP missing frames.
            - None if the gap is too large or no prior detection exists.
        """
        if ball is not None:
            cx, cy = ball.center
            self._history.append((frame_index, float(cx), float(cy)))
            self._last_ball = ball
            self._frames_missing = 0
            return ball

        # Ball not detected this frame
        self._frames_missing += 1

        if self._frames_missing > self.MAX_GAP or self._last_ball is None:
            return None

        # Project forward using velocity from the last two known positions
        if len(self._history) >= 2:
            f1, x1, y1 = self._history[-2]
            f2, x2, y2 = self._history[-1]
            if f2 > f1:
                steps = float(self._frames_missing)
                vx = (x2 - x1) / (f2 - f1)
                vy = (y2 - y1) / (f2 - f1)
                proj_x = x2 + vx * steps
                proj_y = y2 + vy * steps
            else:
                proj_x, proj_y = float(x2), float(y2)
        else:
            _, proj_x, proj_y = self._history[-1]

        r = 5.0
        synthetic_rect = (proj_x - r, proj_y - r, proj_x + r, proj_y + r)
        return Ball(rect=synthetic_rect, score=0.0)

    def get_velocity(self) -> Optional[Tuple[float, float]]:
        """Returns (vx, vy) in pixels/frame, or None if insufficient history."""
        if len(self._history) < 2:
            return None
        f1, x1, y1 = self._history[-2]
        f2, x2, y2 = self._history[-1]
        if f2 == f1:
            return None
        return (x2 - x1) / (f2 - f1), (y2 - y1) / (f2 - f1)
