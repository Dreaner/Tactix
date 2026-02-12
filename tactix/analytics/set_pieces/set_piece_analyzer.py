"""
Project: Tactix
File Created: 2026-02-12
Author: Xingnan Zhu
File Name: set_piece_analyzer.py
Description:
    Accumulates and analyzes corner kicks and free kicks.

    CornerAnalyzer
    ──────────────
    On CornerEvent: opens a tracking window (SET_PIECE_WINDOW_FRAMES).
    Within the window: accumulates passes + first shot.
    Ends on: first shot | window expiry.
    Outcome: "goal" | "shot" | "possession" (no shot, but kept ball) | "lost"

    FreeKickAnalyzer
    ────────────────
    On FreeKickEvent: records position + xG estimate.
    Opens a short tracking window; ends on shot or window expiry.
    Outcome: "goal" | "shot" | "no_shot"

    No rendering — see tactix.visualization.overlays.set_pieces.set_pieces
"""

from dataclasses import dataclass, field
from typing import List, Optional

from tactix.config import Config
from tactix.core.events import CornerEvent, FreeKickEvent, FrameEvents
from tactix.core.types import Point, TeamID


# ─────────────────────────────────────────────
# Corner
# ─────────────────────────────────────────────

@dataclass
class CornerSequence:
    """One corner kick attempt and its outcome."""
    team: TeamID
    side: str               # "left" | "right"  (which goal-line end)
    location: Point         # ball position at corner detection
    frame_index: int
    pass_count: int = 0
    shot_taken: bool = False
    xg: float = 0.0
    outcome: str = "unknown"   # "goal" | "shot" | "possession" | "lost" | "unknown"
    end_frame: int = -1


class CornerAnalyzer:
    """
    Detects corner sequences and accumulates pass/shot outcomes.

    Public attributes:
        .sequences — List[CornerSequence]
        .active    — Optional[CornerSequence]
    """

    def __init__(self, cfg: Config) -> None:
        self._window = getattr(cfg, "SET_PIECE_WINDOW_FRAMES", 300)
        self.sequences: List[CornerSequence] = []
        self.active: Optional[CornerSequence] = None
        self._active_start: int = -1

    def update(self, frame_index: int, events: FrameEvents) -> None:
        # New corner event
        if events.corner:
            if self.active is not None:
                self._finish(self.active, "unknown", frame_index)
            evt = events.corner
            side = "left" if (evt.location and evt.location.x < 52.5) else "right"
            self.active = CornerSequence(
                team=evt.team,
                side=side,
                location=evt.location or Point(0.0, 0.0),
                frame_index=frame_index,
            )
            self._active_start = frame_index

        if self.active is None:
            return

        # Accumulate passes
        for pass_evt in events.passes:
            if pass_evt.team == self.active.team:
                self.active.pass_count += 1

        # First shot ends the sequence
        for shot in events.shots:
            if shot.team == self.active.team:
                self.active.shot_taken = True
                self.active.xg = shot.xg
                outcome = "goal" if shot.outcome == "goal" else "shot"
                self._finish(self.active, outcome, frame_index)
                self.active = None
                return

        # Possession change ends the sequence as "lost"
        if events.possession_change:
            gaining = events.possession_change.gaining_team
            if gaining != self.active.team:
                outcome = "possession" if self.active.pass_count > 0 else "lost"
                self._finish(self.active, outcome, frame_index)
                self.active = None
                return

        # Window expiry
        if frame_index - self._active_start > self._window:
            outcome = "possession" if self.active.pass_count > 0 else "unknown"
            self._finish(self.active, outcome, frame_index)
            self.active = None

    def _finish(self, seq: CornerSequence, outcome: str, frame_index: int) -> None:
        seq.outcome = outcome
        seq.end_frame = frame_index
        self.sequences.append(seq)

    def summary(self) -> dict:
        total = len(self.sequences)
        if total == 0:
            return {}
        return {
            "total": total,
            "goals": sum(1 for s in self.sequences if s.outcome == "goal"),
            "shots": sum(1 for s in self.sequences if s.shot_taken),
            "avg_passes": sum(s.pass_count for s in self.sequences) / total,
        }


# ─────────────────────────────────────────────
# Free Kick
# ─────────────────────────────────────────────

@dataclass
class FreeKickSequence:
    """One free kick attempt and its outcome."""
    team: TeamID
    location: Point         # where the FK was taken
    frame_index: int
    xg: float = 0.0        # from EventDetector estimate
    wall_size: int = 0
    shot_taken: bool = False
    outcome: str = "no_shot"  # "goal" | "shot" | "no_shot"
    end_frame: int = -1


class FreeKickAnalyzer:
    """
    Detects free kick sequences and records shot outcomes.

    Public attributes:
        .sequences — List[FreeKickSequence]
        .active    — Optional[FreeKickSequence]
    """

    def __init__(self, cfg: Config) -> None:
        self._window = getattr(cfg, "SET_PIECE_WINDOW_FRAMES", 300)
        self.sequences: List[FreeKickSequence] = []
        self.active: Optional[FreeKickSequence] = None
        self._active_start: int = -1

    def update(self, frame_index: int, events: FrameEvents) -> None:
        # New free kick event
        if events.free_kick:
            if self.active is not None:
                self._finish(self.active, "no_shot", frame_index)
            evt = events.free_kick
            self.active = FreeKickSequence(
                team=evt.team,
                location=evt.location or Point(0.0, 0.0),
                frame_index=frame_index,
                xg=evt.xg,
                wall_size=evt.wall_size,
            )
            self._active_start = frame_index

        if self.active is None:
            return

        # Shot check
        for shot in events.shots:
            if shot.team == self.active.team:
                self.active.shot_taken = True
                self.active.xg = max(self.active.xg, shot.xg)
                outcome = "goal" if shot.outcome == "goal" else "shot"
                self._finish(self.active, outcome, frame_index)
                self.active = None
                return

        # Window expiry
        if frame_index - self._active_start > self._window:
            self._finish(self.active, "no_shot", frame_index)
            self.active = None

    def _finish(self, seq: FreeKickSequence, outcome: str, frame_index: int) -> None:
        seq.outcome = outcome
        seq.end_frame = frame_index
        self.sequences.append(seq)

    def summary(self) -> dict:
        total = len(self.sequences)
        if total == 0:
            return {}
        return {
            "total": total,
            "goals": sum(1 for s in self.sequences if s.outcome == "goal"),
            "shots": sum(1 for s in self.sequences if s.shot_taken),
            "avg_xg": sum(s.xg for s in self.sequences) / total,
        }
