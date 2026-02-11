"""
Project: Tactix
File Created: 2026-02-11
Author: Xingnan Zhu
File Name: buildup_tracker.py
Description:
    Pure-data state-machine tracker for attacking build-up sequences.
    No rendering logic — see tactix.visualization.overlays.attacking.buildup
    for the corresponding RGBA overlay renderer.

    A build-up starts when a team gains possession in their defensive third
    and ends via shot, possession loss, territorial advance, or timeout.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

from tactix.config import Config
from tactix.core.events import FrameEvents
from tactix.core.types import Point, TeamID


class BuildupState(Enum):
    IDLE = auto()
    BUILDING = auto()


@dataclass
class BuildupSequence:
    """One completed or active build-up sequence."""
    team: TeamID
    start_frame: int
    waypoints: List[Point] = field(default_factory=list)
    pass_count: int = 0
    x_start: float = 0.0
    x_max: float = 0.0
    outcome: str = "unknown"    # "shot" | "lost" | "advanced" | "timeout"
    end_frame: int = -1

    @property
    def duration_frames(self) -> int:
        return max(self.end_frame - self.start_frame, 0)

    @property
    def x_advance(self) -> float:
        return self.x_max - self.x_start


class BuildupTracker:
    """
    Tracks one active build-up per team simultaneously.
    Completed sequences are stored in .sequences.

    Public attributes for the renderer:
        .sequences  — List[BuildupSequence]   all completed sequences
        .active     — Optional[BuildupSequence]  currently building (or None)
        .state      — BuildupState
    """

    MAX_FRAMES: int = 600    # ~20 s at 30 fps

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self.sequences: List[BuildupSequence] = []
        self.active: Optional[BuildupSequence] = None
        self.state: BuildupState = BuildupState.IDLE
        self._active_team: Optional[TeamID] = None

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def update(self, frame_index: int, events: FrameEvents) -> None:
        """Feed current frame events into the state machine."""
        if events.possession_change:
            self._handle_possession_change(frame_index, events.possession_change)

        if self.state == BuildupState.BUILDING and self.active is not None:
            for p in events.passes:
                if p.team == self._active_team and p.origin is not None:
                    self.active.waypoints.append(p.origin)
                    self.active.pass_count += 1
                    if p.origin.x > self.active.x_max:
                        self.active.x_max = p.origin.x

            for s in events.shots:
                if s.team == self._active_team:
                    self._finish("shot", frame_index)
                    return

            if (self.active.waypoints
                    and self.active.waypoints[-1].x > self._cfg.ATTACKING_THIRD_X):
                self._finish("advanced", frame_index)
                return

            if frame_index - self.active.start_frame > self.MAX_FRAMES:
                self._finish("timeout", frame_index)

    def _handle_possession_change(self, frame_index: int, evt) -> None:
        gaining = evt.gaining_team
        location = evt.location

        if self.state == BuildupState.BUILDING and gaining != self._active_team:
            self._finish("lost", frame_index)

        if location is not None:
            in_defensive = (
                (gaining == TeamID.A and location.x < self._cfg.DEFENSIVE_THIRD_X)
                or (gaining == TeamID.B and location.x > (105.0 - self._cfg.DEFENSIVE_THIRD_X))
            )
            if in_defensive and gaining in (TeamID.A, TeamID.B):
                self.active = BuildupSequence(
                    team=gaining,
                    start_frame=frame_index,
                    waypoints=[location],
                    x_start=location.x,
                    x_max=location.x,
                )
                self._active_team = gaining
                self.state = BuildupState.BUILDING

    def _finish(self, outcome: str, frame_index: int) -> None:
        if self.active is not None:
            self.active.outcome = outcome
            self.active.end_frame = frame_index
            self.sequences.append(self.active)
        self.active = None
        self._active_team = None
        self.state = BuildupState.IDLE

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        by_team = {TeamID.A: [], TeamID.B: []}
        for seq in self.sequences:
            if seq.team in by_team:
                by_team[seq.team].append(seq)

        result = {}
        for team, seqs in by_team.items():
            if not seqs:
                result[team] = {}
                continue
            result[team] = {
                "total": len(seqs),
                "avg_passes": sum(s.pass_count for s in seqs) / len(seqs),
                "avg_advance_m": sum(s.x_advance for s in seqs) / len(seqs),
                "outcomes": {o: sum(1 for s in seqs if s.outcome == o)
                             for o in ("shot", "lost", "advanced", "timeout")},
            }
        return result
