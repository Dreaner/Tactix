"""
Project: Tactix
File Created: 2026-02-07 16:02:54
Author: Xingnan Zhu
File Name: team_width_length.py
Description:
    Calculates and visualizes the team's width and length (depth).
    It draws a bounding box around the team to show their spatial spread.
"""

import cv2
import numpy as np
from tactix.core.types import FrameData, TeamID, PitchConfig
from tactix.config import Colors

class TeamWidthLength:
    def __init__(self):
        self.w = PitchConfig.PIXEL_WIDTH
        self.h = PitchConfig.PIXEL_HEIGHT

    def generate_overlay(self, frame_data: FrameData) -> np.ndarray:
        """
        Generate Team Width & Length overlay (RGBA).
        """
        overlay = np.zeros((self.h, self.w, 4), dtype=np.uint8)
        
        self._draw_bounding_box(frame_data, TeamID.A, overlay, Colors.to_bgr(Colors.TEAM_A))
        self._draw_bounding_box(frame_data, TeamID.B, overlay, Colors.to_bgr(Colors.TEAM_B))
        
        return overlay

    def _draw_bounding_box(self, frame_data: FrameData, team: TeamID, overlay: np.ndarray, color: tuple):
        """
        Calculate and draw bounding box for a specific team.
        """
        xs = []
        ys = []
        
        for p in frame_data.players:
            if p.team == team and p.pitch_position:
                xs.append(p.pitch_position.x)
                ys.append(p.pitch_position.y)
        
        if len(xs) < 2:
            return

        # Calculate bounds (Meters)
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        # Convert to Pixels
        px1 = int(min_x * PitchConfig.X_SCALE)
        py1 = int(min_y * PitchConfig.Y_SCALE)
        px2 = int(max_x * PitchConfig.X_SCALE)
        py2 = int(max_y * PitchConfig.Y_SCALE)
        
        # Draw dashed rectangle
        # OpenCV doesn't support dashed lines natively in rectangle, so we draw lines manually or just use solid thin line
        # Let's use a thin solid line with low opacity
        
        # Box
        cv2.rectangle(overlay, (px1, py1), (px2, py2), (*color, 150), 1, cv2.LINE_AA)
        
        # Optional: Draw dimensions text
        length_m = max_x - min_x
        width_m = max_y - min_y
        
        # Text settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        text_color = (*color, 255)
        
        # Draw Length (Horizontal)
        label_l = f"{length_m:.1f}m"
        cv2.putText(overlay, label_l, (int((px1+px2)/2) - 20, py2 + 15), font, scale, text_color, thickness)
        
        # Draw Width (Vertical)
        label_w = f"{width_m:.1f}m"
        cv2.putText(overlay, label_w, (px2 + 5, int((py1+py2)/2)), font, scale, text_color, thickness)
