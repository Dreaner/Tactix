"""
Project: Tactix
File Created: 2026-02-07 16:02:42
Author: Xingnan Zhu
File Name: team_centroid.py
Description:
    Calculates and visualizes the geometric center (centroid) of each team.
    This helps in analyzing the team's average position on the pitch.
"""

import cv2
import numpy as np
from tactix.core.types import FrameData, TeamID, PitchConfig
from tactix.config import Colors

class TeamCentroid:
    def __init__(self):
        self.w = PitchConfig.PIXEL_WIDTH
        self.h = PitchConfig.PIXEL_HEIGHT

    def generate_overlay(self, frame_data: FrameData) -> np.ndarray:
        """
        Generate Team Centroid overlay (RGBA).
        """
        overlay = np.zeros((self.h, self.w, 4), dtype=np.uint8)
        
        self._draw_centroid(frame_data, TeamID.A, overlay, Colors.to_bgr(Colors.TEAM_A))
        self._draw_centroid(frame_data, TeamID.B, overlay, Colors.to_bgr(Colors.TEAM_B))
        
        return overlay

    def _draw_centroid(self, frame_data: FrameData, team: TeamID, overlay: np.ndarray, color: tuple):
        """
        Calculate and draw centroid for a specific team.
        """
        x_sum = 0.0
        y_sum = 0.0
        count = 0
        
        for p in frame_data.players:
            if p.team == team and p.pitch_position:
                x_sum += p.pitch_position.x
                y_sum += p.pitch_position.y
                count += 1
        
        if count == 0:
            return

        # Calculate average position (Meters)
        avg_x = x_sum / count
        avg_y = y_sum / count
        
        # Convert to Pixels
        px = int(avg_x * PitchConfig.X_SCALE)
        py = int(avg_y * PitchConfig.Y_SCALE)
        
        # Draw 'X' marker
        # Thickness 3, Size 20
        marker_size = 20
        # Line 1
        cv2.line(overlay, (px - marker_size, py - marker_size), (px + marker_size, py + marker_size), (*color, 255), 3, cv2.LINE_AA)
        # Line 2
        cv2.line(overlay, (px + marker_size, py - marker_size), (px - marker_size, py + marker_size), (*color, 255), 3, cv2.LINE_AA)
        
        # Optional: Draw a circle around it
        cv2.circle(overlay, (px, py), marker_size + 5, (*color, 255), 2, cv2.LINE_AA)
