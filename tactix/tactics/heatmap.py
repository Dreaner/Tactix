"""
Project: Tactix
File Created: 2026-02-04 18:00:00
Author: Xingnan Zhu
File Name: heatmap.py
Description:
    Generates a cumulative heatmap of player movements over the course of the video.
    It maintains a 2D grid representing the pitch and accumulates player positions,
    applying Gaussian blur and color-mapping for visualization.
"""

import cv2
import numpy as np
from tactix.core.types import FrameData, TeamID, PitchConfig

class HeatmapGenerator:
    def __init__(self):
        # Use a smaller grid size for performance, scale up when rendering
        # Real pitch 105x68m -> Grid 105x68 (1m/cell) or finer
        # Here we use 1/10 of the tactical board pixel size, preserving detail while being fast
        self.scale_factor = 0.1
        self.grid_w = int(PitchConfig.PIXEL_WIDTH * self.scale_factor)
        self.grid_h = int(PitchConfig.PIXEL_HEIGHT * self.scale_factor)
        
        # Initialize two heatmap grids (Team A and Team B separate)
        # float32 for accumulation
        self.heatmap_a = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        self.heatmap_b = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)

    def update(self, frame_data: FrameData):
        """
        Called every frame to accumulate player positions
        """
        for p in frame_data.players:
            if p.pitch_position and p.team in [TeamID.A, TeamID.B]:
                # Map to grid coordinates
                gx = int(p.pitch_position.x * PitchConfig.X_SCALE * self.scale_factor)
                gy = int(p.pitch_position.y * PitchConfig.Y_SCALE * self.scale_factor)
                
                # Boundary check
                if 0 <= gx < self.grid_w and 0 <= gy < self.grid_h:
                    # Accumulate value (can be weighted by speed: fast moving = low heat, standing = high heat)
                    # Here we simply add 1 per frame
                    if p.team == TeamID.A:
                        self.heatmap_a[gy, gx] += 1.0
                    else:
                        self.heatmap_b[gy, gx] += 1.0

    def generate_overlay(self, team: TeamID = None) -> np.ndarray:
        """
        Generate visualized heatmap overlay (RGBA)
        :param team: Specify which team to generate, None for both combined
        """
        if team == TeamID.A:
            grid = self.heatmap_a
        elif team == TeamID.B:
            grid = self.heatmap_b
        else:
            grid = self.heatmap_a + self.heatmap_b

        # 1. Normalize (0-255)
        # Avoid division by zero
        max_val = np.max(grid)
        if max_val == 0:
            return np.zeros((PitchConfig.PIXEL_HEIGHT, PitchConfig.PIXEL_WIDTH, 4), dtype=np.uint8)
            
        norm_grid = (grid / max_val * 255).astype(np.uint8)

        # 2. Gaussian Blur (Turn dots into clouds)
        # sigma determines the spread range
        blurred = cv2.GaussianBlur(norm_grid, (0, 0), sigmaX=3, sigmaY=3)

        # 3. Apply Pseudo-color (Colormap)
        # COLORMAP_JET is the classic heatmap color (Blue->Cyan->Yellow->Red)
        colored_map = cv2.applyColorMap(blurred, cv2.COLORMAP_JET)

        # 4. Handle Transparency
        # We want low heat areas (blue background) to be transparent, high heat opaque
        # blurred itself is the brightness value, can be used as base for Alpha channel
        
        # Create Alpha channel
        # Threshold filter: Too cold areas become transparent
        alpha = blurred.copy()
        alpha[alpha < 30] = 0 
        # Enhance visibility slightly
        alpha = cv2.normalize(alpha, None, 0, 180, cv2.NORM_MINMAX)

        # Merge channels
        rgba_small = np.dstack((colored_map, alpha))

        # 5. Scale back to original size
        rgba_full = cv2.resize(rgba_small, (PitchConfig.PIXEL_WIDTH, PitchConfig.PIXEL_HEIGHT), interpolation=cv2.INTER_LINEAR)
        
        return rgba_full
