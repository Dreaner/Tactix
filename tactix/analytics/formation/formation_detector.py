"""
Project: Tactix
File Created: 2026-02-14
Author: Xingnan Zhu
File Name: formation_detector.py
Description:
    Pure-data accumulator for per-team tactical formation detection.
    No rendering logic — see tactix.visualization.overlays.formation.formation
    for the corresponding RGBA overlay renderer.

    Uses K-Means clustering on player X-coordinates to detect defensive /
    midfield / attacking lines, counts players per line, and matches the
    resulting tuple against a dictionary of canonical formation names.
    A sliding-window mode vote provides temporal smoothing.
"""

from collections import Counter, deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from tactix.config import Config
from tactix.core.types import FrameData, Player, TeamID

# ------------------------------------------------------------------
# Canonical formation templates
# ------------------------------------------------------------------

FORMATION_NAMES: Dict[Tuple[int, ...], str] = {
    # 3-line formations
    (3, 4, 3): "3-4-3",
    (3, 5, 2): "3-5-2",
    (3, 3, 4): "3-3-4",
    (4, 3, 3): "4-3-3",
    (4, 4, 2): "4-4-2",
    (4, 5, 1): "4-5-1",
    (5, 3, 2): "5-3-2",
    (5, 4, 1): "5-4-1",
    (5, 2, 3): "5-2-3",
    # 4-line formations
    (3, 4, 2, 1): "3-4-2-1",
    (3, 4, 1, 2): "3-4-1-2",
    (4, 2, 3, 1): "4-2-3-1",
    (4, 1, 4, 1): "4-1-4-1",
    (4, 1, 2, 3): "4-1-2-3",
    (4, 2, 1, 3): "4-2-1-3",
    (4, 3, 2, 1): "4-3-2-1",
    (4, 1, 3, 2): "4-1-3-2",
    (4, 2, 2, 2): "4-2-2-2",
    (3, 1, 4, 2): "3-1-4-2",
    (5, 2, 2, 1): "5-2-2-1",
    # 2-line formations (unusual but possible)
    (5, 5): "5-5",
    (6, 4): "6-4",
    (4, 6): "4-6",
}


def _nearest_formation(line_counts: Tuple[int, ...]) -> str:
    """Return the canonical name if it exists, otherwise build a dash string."""
    if line_counts in FORMATION_NAMES:
        return FORMATION_NAMES[line_counts]
    return "-".join(str(c) for c in line_counts)


def _detect_lines(x_coords: np.ndarray) -> Tuple[int, ...]:
    """
    Cluster X-coordinates into 2, 3, or 4 lines and return the player
    count per line sorted from defensive (smaller X) to attacking.

    Uses silhouette score to choose the best K among {2, 3, 4}.
    """
    n = len(x_coords)
    if n < 4:
        return (n,)

    X = x_coords.reshape(-1, 1)
    best_k = 3
    best_score = -1.0

    for k in (2, 3, 4):
        if k >= n:
            continue
        km = KMeans(n_clusters=k, n_init=5, max_iter=50, random_state=0)
        labels = km.fit_predict(X)
        # silhouette_score requires >= 2 labels
        unique_labels = set(labels)
        if len(unique_labels) < 2:
            continue
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score = score
            best_k = k

    km = KMeans(n_clusters=best_k, n_init=5, max_iter=50, random_state=0)
    labels = km.fit_predict(X)
    centers = km.cluster_centers_.flatten()

    # Sort clusters by ascending X (defensive → attacking)
    order = np.argsort(centers)
    counts: List[int] = []
    for cluster_idx in order:
        counts.append(int(np.sum(labels == cluster_idx)))

    return tuple(counts)


class FormationDetector:
    """
    Detects tactical formation per team using positional clustering
    with temporal smoothing via a sliding window mode vote.

    Public attributes for the renderer:
        .current    — Dict[TeamID, str]          e.g. {TeamID.A: "4-3-3"}
        .raw_lines  — Dict[TeamID, Tuple[int,...]]  e.g. {TeamID.A: (4, 3, 3)}
        .confidence — Dict[TeamID, float]         smoothing confidence 0.0–1.0
    """

    def __init__(self, cfg: Config) -> None:
        self._window_size: int = cfg.FORMATION_WINDOW_FRAMES
        self._min_players: int = cfg.FORMATION_MIN_PLAYERS

        # Sliding windows keyed by team
        self._buffers: Dict[TeamID, deque] = {
            TeamID.A: deque(maxlen=self._window_size),
            TeamID.B: deque(maxlen=self._window_size),
        }

        # Public state
        self.current: Dict[TeamID, str] = {TeamID.A: "Unknown", TeamID.B: "Unknown"}
        self.raw_lines: Dict[TeamID, Tuple[int, ...]] = {}
        self.confidence: Dict[TeamID, float] = {TeamID.A: 0.0, TeamID.B: 0.0}

        # History for summary / PDF export
        self._history: List[Tuple[int, TeamID, str]] = []

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def update(self, frame_data: FrameData) -> None:
        """Cluster player positions and update smoothed formation."""
        for team in (TeamID.A, TeamID.B):
            self._update_team(frame_data, team)

    def _update_team(self, frame_data: FrameData, team: TeamID) -> None:
        """Detect formation for a single team."""
        # Gather outfield players with valid pitch positions
        outfield = [
            p for p in frame_data.players
            if p.team == team
            and p.pitch_position is not None
        ]
        # Also include goalkeepers assigned to this team for counting,
        # but exclude them from line detection
        # (GK will be excluded from the 10 outfield; we only cluster outfield)
        # GK are TeamID.GOALKEEPER, not TeamID.A/B, so they're already excluded.

        if len(outfield) < self._min_players:
            return  # Not enough data — keep last known formation

        # Extract X coordinates (depth on the pitch)
        x_coords = np.array([p.pitch_position.x for p in outfield], dtype=np.float64)

        # Flip Team B so their defensive line has *smaller* X values
        # (Team B defends the right goal, x ≈ 105 → flip to x ≈ 0)
        if team == TeamID.B:
            x_coords = 105.0 - x_coords

        # Cluster into lines
        line_counts = _detect_lines(x_coords)
        self.raw_lines[team] = line_counts

        # Push into sliding window
        buf = self._buffers[team]
        buf.append(line_counts)

        # Mode vote
        counter = Counter(buf)
        most_common, freq = counter.most_common(1)[0]
        total = len(buf)
        conf = freq / total

        formation_name = _nearest_formation(most_common)
        self.current[team] = formation_name
        self.confidence[team] = conf

        # Record to history if formation changed
        if (not self._history
                or self._history[-1][1] != team
                or self._history[-1][2] != formation_name):
            self._history.append((frame_data.frame_index, team, formation_name))

    # ------------------------------------------------------------------
    # Query / export
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return formation stats for export / PDF report."""
        return {
            "team_a": self.current.get(TeamID.A, "Unknown"),
            "team_b": self.current.get(TeamID.B, "Unknown"),
            "team_a_confidence": round(self.confidence.get(TeamID.A, 0.0), 2),
            "team_b_confidence": round(self.confidence.get(TeamID.B, 0.0), 2),
            "history": [
                {"frame": f, "team": t.name, "formation": fm}
                for f, t, fm in self._history
            ],
        }
