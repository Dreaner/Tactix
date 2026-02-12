"""
Project: Tactix
File Created: 2026-02-12
Author: Xingnan Zhu
File Name: transition_tracker.py
Description:
    Tracks two complementary transition sequences simultaneously:

    ATTACK  (Recovery → Shot / loss)
        Triggered when a team GAINS possession.
        Ends when: shot taken | possession lost | timeout (ATTACK_MAX_FRAMES).
        Measures: how quickly the team progresses toward a chance after winning the ball.

    DEFEND  (Ball-loss → Recovery)
        Triggered when a team LOSES possession.
        Ends when: possession regained | timeout (DEFEND_MAX_FRAMES).
        Measures: how quickly the team recovers structural shape after losing the ball.

    For each frame tick, call:
        tracker.update(frame_index, events)

    Completed sequences are stored in .sequences (both phases, both teams).
    Renderer: tactix.visualization.overlays.transition.transition
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional

from tactix.config import Config
from tactix.core.events import FrameEvents
from tactix.core.types import Point, TeamID


# ─────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────

class TransitionPhase(Enum):
    ATTACK = auto()   # recovery-to-shot
    DEFEND = auto()   # ball-loss-to-recovery


@dataclass
class TransitionSequence:
    team: TeamID
    phase: TransitionPhase
    start_frame: int
    waypoints: List[Point] = field(default_factory=list)  # ball positions (passes)
    pass_count: int = 0
    shots_taken: int = 0
    outcome: str = "unknown"
    # attack outcomes : "shot" | "goal" | "lost" | "timeout"
    # defend outcomes : "regained" | "timeout"
    end_frame: int = -1

    @property
    def duration_frames(self) -> int:
        return max(self.end_frame - self.start_frame, 0)

    @property
    def x_advance(self) -> float:
        """Metres gained along the x-axis during the sequence."""
        if len(self.waypoints) < 2:
            return 0.0
        return self.waypoints[-1].x - self.waypoints[0].x


# ─────────────────────────────────────────────
# Tracker
# ─────────────────────────────────────────────

class TransitionTracker:
    """
    Tracks one ATTACK + one DEFEND sequence per team simultaneously.

    Public attributes for the renderer:
        .sequences  — List[TransitionSequence]  all completed sequences
        .active_attack  — Dict[TeamID, Optional[TransitionSequence]]
        .active_defend  — Dict[TeamID, Optional[TransitionSequence]]
    """

    # Default timeouts; overridden by cfg if provided
    _ATTACK_MAX = 450   # ≈ 15 s at 30 fps
    _DEFEND_MAX = 900   # ≈ 30 s at 30 fps

    def __init__(self, cfg: Config) -> None:
        self._attack_max = getattr(cfg, "TRANSITION_ATTACK_MAX_FRAMES", self._ATTACK_MAX)
        self._defend_max = getattr(cfg, "TRANSITION_DEFEND_MAX_FRAMES", self._DEFEND_MAX)

        self.sequences: List[TransitionSequence] = []
        self.active_attack: Dict[TeamID, Optional[TransitionSequence]] = {
            TeamID.A: None, TeamID.B: None
        }
        self.active_defend: Dict[TeamID, Optional[TransitionSequence]] = {
            TeamID.A: None, TeamID.B: None
        }

    # ─── Per-frame entry point ────────────────

    def update(self, frame_index: int, events: FrameEvents) -> None:
        """Feed current-frame events into both state machines."""
        if events.possession_change:
            self._handle_possession_change(frame_index, events.possession_change)

        for pass_evt in events.passes:
            self._accumulate_pass(pass_evt)

        for shot in events.shots:
            self._handle_shot(frame_index, shot)

        self._check_timeouts(frame_index)

    # ─── Event handlers ───────────────────────

    def _handle_possession_change(self, frame_index: int, evt) -> None:
        gaining = evt.gaining_team
        location = evt.location

        # Determine losing team
        if gaining == TeamID.A:
            losing = TeamID.B
        elif gaining == TeamID.B:
            losing = TeamID.A
        else:
            return

        # ── LOSING team ──────────────────────
        # End their attack sequence
        if self.active_attack[losing] is not None:
            self._finish(self.active_attack[losing], "lost", frame_index)
            self.active_attack[losing] = None

        # Start their defend sequence
        seq = TransitionSequence(
            team=losing, phase=TransitionPhase.DEFEND,
            start_frame=frame_index,
            waypoints=[location] if location else [],
        )
        self.active_defend[losing] = seq

        # ── GAINING team ─────────────────────
        # End their defend sequence
        if self.active_defend[gaining] is not None:
            self._finish(self.active_defend[gaining], "regained", frame_index)
            self.active_defend[gaining] = None

        # Start their attack sequence
        seq = TransitionSequence(
            team=gaining, phase=TransitionPhase.ATTACK,
            start_frame=frame_index,
            waypoints=[location] if location else [],
        )
        self.active_attack[gaining] = seq

    def _accumulate_pass(self, pass_evt) -> None:
        team = pass_evt.team
        if pass_evt.origin is None:
            return
        if self.active_attack.get(team) is not None:
            self.active_attack[team].waypoints.append(pass_evt.origin)
            self.active_attack[team].pass_count += 1

    def _handle_shot(self, frame_index: int, shot) -> None:
        team = shot.team
        seq = self.active_attack.get(team)
        if seq is not None:
            seq.shots_taken += 1
            outcome = "goal" if shot.outcome == "goal" else "shot"
            self._finish(seq, outcome, frame_index)
            self.active_attack[team] = None

    def _check_timeouts(self, frame_index: int) -> None:
        for team in (TeamID.A, TeamID.B):
            seq = self.active_attack[team]
            if seq and frame_index - seq.start_frame > self._attack_max:
                self._finish(seq, "timeout", frame_index)
                self.active_attack[team] = None

            seq = self.active_defend[team]
            if seq and frame_index - seq.start_frame > self._defend_max:
                self._finish(seq, "timeout", frame_index)
                self.active_defend[team] = None

    # ─── Helpers ──────────────────────────────

    def _finish(self, seq: TransitionSequence, outcome: str, frame_index: int) -> None:
        seq.outcome = outcome
        seq.end_frame = frame_index
        self.sequences.append(seq)

    # ─── Stats ────────────────────────────────

    def summary(self) -> dict:
        result: dict = {}
        for team in (TeamID.A, TeamID.B):
            for phase in TransitionPhase:
                seqs = [
                    s for s in self.sequences
                    if s.team == team and s.phase == phase
                ]
                if not seqs:
                    result[(team, phase)] = {}
                    continue
                result[(team, phase)] = {
                    "total": len(seqs),
                    "avg_duration_frames": sum(s.duration_frames for s in seqs) / len(seqs),
                    "avg_passes": sum(s.pass_count for s in seqs) / len(seqs),
                    "outcomes": {
                        o: sum(1 for s in seqs if s.outcome == o)
                        for o in ("shot", "goal", "lost", "regained", "timeout")
                    },
                }
        return result
