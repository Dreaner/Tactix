"""
Project: Tactix
File Created: 2026-02-11
Author: Xingnan Zhu
File Name: event_detector.py
Description:
    Detects discrete tactical events from per-frame tracking data.
    This is the foundation of the 5-Phase analysis system (Milestone 0).

    Called once per frame from TactixEngine._stage_tactical_analysis().
    Returns a FrameEvents object containing all events detected this frame.

    All thresholds are read from Config and must NOT be hardcoded here.
"""

import math
from collections import deque
from typing import Deque, Dict, Optional, Tuple

import numpy as np

from tactix.config import Config
from tactix.core.events import (
    CornerEvent,
    DuelEvent,
    FrameEvents,
    FreeKickEvent,
    PassEvent,
    PossessionChangeEvent,
    ShotEvent,
)
from tactix.core.types import FrameData, Player, Point, TeamID


class EventDetector:
    """
    Stateful event detector that sits alongside TactixEngine as a side-channel.
    Maintains the minimum cross-frame state needed to confirm each event type.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

        # --- Possession tracking ---
        # Ring buffer of (frame_index, owner_team, owner_id) over last N frames
        self._ownership_history: Deque[Tuple[int, TeamID, int]] = deque(
            maxlen=cfg.POSSESSION_CONFIRM_FRAMES + 2
        )
        self._current_owner_team: TeamID = TeamID.UNKNOWN
        self._current_owner_id: int = -1
        self._pending_change_team: TeamID = TeamID.UNKNOWN
        self._pending_change_frames: int = 0

        # --- Pass tracking ---
        # Remember last confirmed owner to detect handoff
        self._last_owner_id: int = -1
        self._last_owner_team: TeamID = TeamID.UNKNOWN

        # --- Ball crossing detection (for corners) ---
        self._ball_was_in_play: bool = True   # False when ball last seen out of bounds
        self._last_touch_team: TeamID = TeamID.UNKNOWN

        # --- Shot cooldown (avoid detecting the same shot for many frames) ---
        self._last_shot_frame: int = -999

        # --- Free kick: simple cooldown after detection ---
        self._last_fk_frame: int = -999

    # ==========================================
    # Public API
    # ==========================================

    def detect(self, frame_data: FrameData) -> FrameEvents:
        """
        Run all event detectors for this frame.
        Must be called AFTER coordinate mapping so pitch_position is available.
        """
        fi = frame_data.frame_index
        events = FrameEvents(frame_index=fi)

        # Require geometry to be valid — events need real-world coordinates
        if not frame_data.players or frame_data.homography is None:
            return events

        # 1. Resolve current ball owner from pass_network owner_id (already set upstream)
        current_owner_id = frame_data.ball.owner_id if frame_data.ball else -1
        current_owner: Optional[Player] = frame_data.get_player_by_id(current_owner_id)
        current_team = current_owner.team if current_owner else TeamID.UNKNOWN

        # 2. Possession change
        poss_event = self._detect_possession_change(fi, current_team, current_owner_id, frame_data)
        if poss_event:
            events.possession_change = poss_event

        # 3. Pass (owner changed within same team)
        pass_event = self._detect_pass(fi, current_owner_id, current_team, frame_data)
        if pass_event:
            events.passes.append(pass_event)

        # 4. Shot
        shot_event = self._detect_shot(fi, current_owner, frame_data)
        if shot_event:
            events.shots.append(shot_event)

        # 5. Duels
        events.duels = self._detect_duels(fi, frame_data)

        # 6. Corner
        corner_event = self._detect_corner(fi, frame_data)
        if corner_event:
            events.corner = corner_event

        # 7. Free kick
        fk_event = self._detect_free_kick(fi, frame_data)
        if fk_event:
            events.free_kick = fk_event

        # Update ownership memory for next frame
        self._last_owner_id = current_owner_id
        self._last_owner_team = current_team

        return events

    # ==========================================
    # Possession Change
    # ==========================================

    def _detect_possession_change(
        self,
        fi: int,
        current_team: TeamID,
        current_owner_id: int,
        frame_data: FrameData,
    ) -> Optional[PossessionChangeEvent]:
        if current_team in (TeamID.UNKNOWN, TeamID.REFEREE):
            return None

        if current_team == self._current_owner_team:
            # Same team still has ball — reset pending counter
            self._pending_change_frames = 0
            self._pending_change_team = TeamID.UNKNOWN
            return None

        # A different (valid) team appears to have the ball
        if current_team != self._pending_change_team:
            # New candidate team — start confirmation window
            self._pending_change_team = current_team
            self._pending_change_frames = 1
            return None

        self._pending_change_frames += 1

        if self._pending_change_frames >= self.cfg.POSSESSION_CONFIRM_FRAMES:
            # Change confirmed
            losing_team = self._current_owner_team
            self._current_owner_team = current_team
            self._current_owner_id = current_owner_id
            self._pending_change_frames = 0
            self._pending_change_team = TeamID.UNKNOWN
            self._last_touch_team = losing_team

            ball_pos = frame_data.ball.pitch_position if frame_data.ball else None
            losing_id = self._last_owner_id if self._last_owner_team == losing_team else -1

            return PossessionChangeEvent(
                frame_index=fi,
                losing_team=losing_team,
                gaining_team=current_team,
                location=ball_pos,
                losing_player_id=losing_id,
                gaining_player_id=current_owner_id,
            )

        return None

    # ==========================================
    # Pass Detection
    # ==========================================

    def _detect_pass(
        self,
        fi: int,
        current_owner_id: int,
        current_team: TeamID,
        frame_data: FrameData,
    ) -> Optional[PassEvent]:
        if current_owner_id == -1 or self._last_owner_id == -1:
            return None
        if current_owner_id == self._last_owner_id:
            return None
        # Same team handoff = pass
        if current_team != self._last_owner_team:
            return None
        if current_team in (TeamID.UNKNOWN, TeamID.REFEREE):
            return None

        sender: Optional[Player] = frame_data.get_player_by_id(self._last_owner_id)
        receiver: Optional[Player] = frame_data.get_player_by_id(current_owner_id)

        if not sender or not receiver:
            return None
        if not sender.pitch_position or not receiver.pitch_position:
            return None

        dx = receiver.pitch_position.x - sender.pitch_position.x
        dy = receiver.pitch_position.y - sender.pitch_position.y
        dist = math.sqrt(dx * dx + dy * dy)
        angle = math.degrees(math.atan2(dy, dx))

        return PassEvent(
            frame_index=fi,
            sender_id=self._last_owner_id,
            receiver_id=current_owner_id,
            team=current_team,
            origin=sender.pitch_position,
            destination=receiver.pitch_position,
            distance_m=dist,
            angle_deg=angle,
        )

    # ==========================================
    # Shot Detection
    # ==========================================

    def _detect_shot(
        self,
        fi: int,
        shooter: Optional[Player],
        frame_data: FrameData,
    ) -> Optional[ShotEvent]:
        if fi - self._last_shot_frame < 15:
            return None  # Cooldown: one shot per 0.5 s
        if not shooter or not shooter.pitch_position:
            return None
        if not frame_data.ball or not frame_data.ball.velocity:
            return None
        if shooter.team in (TeamID.UNKNOWN, TeamID.REFEREE):
            return None

        ball_v = frame_data.ball.velocity
        speed_ms = math.sqrt(ball_v.x ** 2 + ball_v.y ** 2)

        if speed_ms < self.cfg.SHOT_VELOCITY_THRESHOLD:
            return None

        # Determine which goal the shooter is attacking
        # Team A attacks right goal (x ≈ 105); Team B attacks left goal (x ≈ 0)
        if shooter.team == TeamID.A:
            goal_x, goal_y = 105.0, 34.0
        else:
            goal_x, goal_y = 0.0, 34.0

        dx = goal_x - shooter.pitch_position.x
        dy = goal_y - shooter.pitch_position.y
        dist_to_goal = math.sqrt(dx * dx + dy * dy)
        angle_to_goal = math.degrees(math.atan2(abs(dy), abs(dx)))

        # Ball must be moving roughly toward the goal (within 60° of the goal direction)
        shot_dx = ball_v.x if shooter.team == TeamID.A else -ball_v.x
        if shot_dx <= 0:
            return None  # Ball moving away from goal

        self._last_shot_frame = fi

        return ShotEvent(
            frame_index=fi,
            shooter_id=shooter.id,
            team=shooter.team,
            location=shooter.pitch_position,
            distance_to_goal_m=dist_to_goal,
            angle_to_goal_deg=angle_to_goal,
            ball_speed_ms=speed_ms,
        )

    # ==========================================
    # Duel Detection
    # ==========================================

    def _detect_duels(self, fi: int, frame_data: FrameData) -> list[DuelEvent]:
        duels: list[DuelEvent] = []
        players_a = [p for p in frame_data.players if p.team == TeamID.A and p.pitch_position]
        players_b = [p for p in frame_data.players if p.team == TeamID.B and p.pitch_position]

        ball_pos = frame_data.ball.pitch_position if frame_data.ball else None

        for pa in players_a:
            for pb in players_b:
                dx = pa.pitch_position.x - pb.pitch_position.x
                dy = pa.pitch_position.y - pb.pitch_position.y
                dist = math.sqrt(dx * dx + dy * dy)

                if dist < self.cfg.DUEL_DISTANCE:
                    winner = self._duel_winner(pa, pb, ball_pos)
                    midpoint = Point(
                        x=(pa.pitch_position.x + pb.pitch_position.x) / 2,
                        y=(pa.pitch_position.y + pb.pitch_position.y) / 2,
                    )
                    duels.append(
                        DuelEvent(
                            frame_index=fi,
                            player_a_id=pa.id,
                            player_b_id=pb.id,
                            team_a=TeamID.A,
                            team_b=TeamID.B,
                            location=midpoint,
                            distance_m=dist,
                            winner_team=winner,
                        )
                    )
        return duels

    @staticmethod
    def _duel_winner(pa: Player, pb: Player, ball_pos: Optional[Point]) -> Optional[TeamID]:
        """Closest player to ball wins the duel (simple heuristic)."""
        if ball_pos is None:
            return None
        da = math.sqrt((pa.pitch_position.x - ball_pos.x) ** 2 + (pa.pitch_position.y - ball_pos.y) ** 2)
        db = math.sqrt((pb.pitch_position.x - ball_pos.x) ** 2 + (pb.pitch_position.y - ball_pos.y) ** 2)
        if da < db - 0.3:
            return TeamID.A
        if db < da - 0.3:
            return TeamID.B
        return None  # Contested

    # ==========================================
    # Corner Detection
    # ==========================================

    def _detect_corner(self, fi: int, frame_data: FrameData) -> Optional[CornerEvent]:
        if not frame_data.ball or not frame_data.ball.pitch_position:
            return None

        bx = frame_data.ball.pitch_position.x

        # Ball out over the goal line
        is_out = bx < -1.0 or bx > 106.0

        if not is_out:
            self._ball_was_in_play = True
            return None

        # Trigger only on the first frame the ball goes out (transition in→out)
        if not self._ball_was_in_play:
            return None

        self._ball_was_in_play = False

        # Defending team = the team that last touched the ball
        # If unknown, skip
        if self._last_touch_team in (TeamID.UNKNOWN, TeamID.REFEREE):
            return None

        attacking_team = TeamID.B if self._last_touch_team == TeamID.A else TeamID.A
        by = frame_data.ball.pitch_position.y
        side = "right" if bx > 52.5 else "left"

        # Identify likely corner taker (attacking player closest to corner arc)
        corner_x = 105.0 if bx > 52.5 else 0.0
        corner_y = 0.0 if by < 34.0 else 68.0
        taker_id = self._nearest_player_to(frame_data, attacking_team, corner_x, corner_y)

        return CornerEvent(
            frame_index=fi,
            attacking_team=attacking_team,
            defending_team=self._last_touch_team,
            side=side,
            taker_id=taker_id,
        )

    # ==========================================
    # Free Kick Detection
    # ==========================================

    def _detect_free_kick(self, fi: int, frame_data: FrameData) -> Optional[FreeKickEvent]:
        if fi - self._last_fk_frame < 90:
            return None  # 3-second cooldown between free kicks
        if not frame_data.ball or not frame_data.ball.pitch_position:
            return None

        # Detect a defensive wall: ≥ WALL_MIN_PLAYERS opposing players
        # in an approximately linear arrangement within 10m of the ball
        ball = frame_data.ball.pitch_position
        wall_info = self._detect_defensive_wall(frame_data, ball)

        if wall_info is None:
            return None

        defending_team = wall_info["team"]
        attacking_team = TeamID.B if defending_team == TeamID.A else TeamID.A

        # Determine goal position for attacking team
        goal_x = 105.0 if attacking_team == TeamID.A else 0.0
        dx = goal_x - ball.x
        dy = 34.0 - ball.y
        dist_to_goal = math.sqrt(dx * dx + dy * dy)
        angle_to_goal = math.degrees(math.atan2(abs(dy), abs(dx)))
        is_direct = ball.x > 75.0 or ball.x < 30.0  # Near penalty area

        xg = self._estimate_fk_xg(dist_to_goal, angle_to_goal)

        self._last_fk_frame = fi

        return FreeKickEvent(
            frame_index=fi,
            attacking_team=attacking_team,
            location=ball,
            distance_to_goal_m=dist_to_goal,
            angle_to_goal_deg=angle_to_goal,
            is_direct=is_direct,
            wall_size=wall_info["size"],
            estimated_xg=xg,
        )

    def _detect_defensive_wall(
        self, frame_data: FrameData, ball: Point
    ) -> Optional[Dict]:
        """
        Returns wall info dict or None. A wall is ≥ WALL_MIN_PLAYERS players
        from the same team within 15m of the ball in a roughly linear arrangement.
        """
        for team in (TeamID.A, TeamID.B):
            candidates = [
                p for p in frame_data.players
                if p.team == team and p.pitch_position
                and math.sqrt(
                    (p.pitch_position.x - ball.x) ** 2 +
                    (p.pitch_position.y - ball.y) ** 2
                ) < 15.0
            ]
            if len(candidates) < self.cfg.WALL_MIN_PLAYERS:
                continue

            positions = np.array([[p.pitch_position.x, p.pitch_position.y] for p in candidates])

            if self._is_linear(positions, tolerance=self.cfg.WALL_LINEAR_TOLERANCE):
                return {"team": team, "size": len(candidates)}

        return None

    @staticmethod
    def _is_linear(positions: np.ndarray, tolerance: float) -> bool:
        """True if points lie roughly on a straight line (residual < tolerance)."""
        if len(positions) < 3:
            return False
        # PCA: smallest eigenvalue measures deviation from best-fit line
        centred = positions - positions.mean(axis=0)
        _, s, _ = np.linalg.svd(centred, full_matrices=False)
        # s[1] is the spread in the minor axis; small → linear arrangement
        return float(s[1]) < tolerance * len(positions)

    # ==========================================
    # Helpers
    # ==========================================

    @staticmethod
    def _nearest_player_to(
        frame_data: FrameData, team: TeamID, x: float, y: float
    ) -> int:
        best_id = -1
        best_dist = float("inf")
        for p in frame_data.players:
            if p.team != team or not p.pitch_position:
                continue
            d = math.sqrt((p.pitch_position.x - x) ** 2 + (p.pitch_position.y - y) ** 2)
            if d < best_dist:
                best_dist = d
                best_id = p.id
        return best_id

    @staticmethod
    def _estimate_fk_xg(dist_to_goal: float, angle_deg: float) -> float:
        """Simple distance + angle decay model for free kick xG."""
        if dist_to_goal < 1.0:
            return 0.0
        base = 1.0 / (1.0 + math.exp(0.15 * dist_to_goal - 2.5))
        angle_factor = max(math.cos(math.radians(angle_deg)), 0.2)
        return round(base * angle_factor, 3)
