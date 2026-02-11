"""
Project: Tactix
File Created: 2026-02-04
Author: Xingnan Zhu
File Name: heatmap.py
Description:
    Cross-frame accumulator for player position heatmaps.
    Maintains two float32 grids (Team A and Team B) that grow
    over the course of the video.

    No rendering — see tactix.visualization.overlays.base.heatmap
    for the HeatmapOverlay renderer.
"""

import numpy as np

from tactix.core.types import FrameData, PitchConfig, TeamID


class HeatmapAccumulator:
    """
    Accumulates player positions into a downscaled grid every frame.

    Public attributes for the renderer:
        .heatmap_a  — np.ndarray (float32) for Team A
        .heatmap_b  — np.ndarray (float32) for Team B
        .grid_w / .grid_h — grid dimensions
        .scale_factor     — downscale ratio vs. PitchConfig canvas
    """

    SCALE_FACTOR: float = 0.1

    def __init__(self) -> None:
        self.scale_factor = self.SCALE_FACTOR
        self.grid_w = int(PitchConfig.PIXEL_WIDTH * self.scale_factor)
        self.grid_h = int(PitchConfig.PIXEL_HEIGHT * self.scale_factor)
        self.heatmap_a = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        self.heatmap_b = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)

    def update(self, frame_data: FrameData) -> None:
        """Accumulate player positions for this frame."""
        for p in frame_data.players:
            if p.pitch_position and p.team in (TeamID.A, TeamID.B):
                gx = int(p.pitch_position.x * PitchConfig.X_SCALE * self.scale_factor)
                gy = int(p.pitch_position.y * PitchConfig.Y_SCALE * self.scale_factor)
                if 0 <= gx < self.grid_w and 0 <= gy < self.grid_h:
                    if p.team == TeamID.A:
                        self.heatmap_a[gy, gx] += 1.0
                    else:
                        self.heatmap_b[gy, gx] += 1.0

    def reset(self) -> None:
        """Clear all accumulated data."""
        self.heatmap_a[:] = 0.0
        self.heatmap_b[:] = 0.0
