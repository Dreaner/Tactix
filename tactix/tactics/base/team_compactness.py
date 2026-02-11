"""
Project: Tactix
File Created: 2026-02-06 15:35:15
Author: Xingnan Zhu
File Name: team_compactness.py
Description:
    Analyzes team compactness using Convex Hull.
    It calculates the polygon area covered by each team and visualizes it
    on the minimap to show defensive/offensive shape.
"""

import cv2
import numpy as np
from scipy.spatial import ConvexHull
from tactix.core.types import FrameData, TeamID, PitchConfig
from tactix.config import Colors

class TeamCompactness:
    def __init__(self):
        self.w = PitchConfig.PIXEL_WIDTH
        self.h = PitchConfig.PIXEL_HEIGHT

    def generate_overlay(self, frame_data: FrameData) -> np.ndarray:
        """
        Generate Convex Hull overlay (RGBA) for both teams.
        """
        overlay = np.zeros((self.h, self.w, 4), dtype=np.uint8)
        
        # Process Team A and Team B
        self._draw_team_hull(frame_data, TeamID.A, overlay, Colors.to_bgr(Colors.TEAM_A))
        self._draw_team_hull(frame_data, TeamID.B, overlay, Colors.to_bgr(Colors.TEAM_B))
        
        return overlay

    def _draw_team_hull(self, frame_data: FrameData, team: TeamID, overlay: np.ndarray, color: tuple):
        """
        Calculate and draw convex hull for a specific team.
        """
        points = []
        for p in frame_data.players:
            if p.team == team and p.pitch_position:
                # Use pixel coordinates directly
                px = int(p.pitch_position.x)
                py = int(p.pitch_position.y)
                points.append([px, py])
        
        points = np.array(points)
        
        # Need at least 3 points to form a polygon
        if len(points) < 3:
            return

        try:
            hull = ConvexHull(points)
            
            # Get the vertices of the hull
            # hull.vertices contains indices of points forming the hull
            hull_points = points[hull.vertices]
            hull_points = hull_points.reshape((-1, 1, 2)) # Format for cv2.fillPoly
            
            # Draw filled polygon (Semi-transparent)
            # We draw on a temp mask first to handle alpha
            mask = np.zeros((self.h, self.w), dtype=np.uint8)
            cv2.fillPoly(mask, [hull_points], 255)
            
            # Apply color and alpha
            # Alpha = 40 (Very faint, so it doesn't block Voronoi/Heatmap too much)
            overlay[mask == 255] = (*color, 40) 
            
            # Draw border line (Solid color, slightly thicker)
            cv2.polylines(overlay, [hull_points], True, (*color, 200), 2, cv2.LINE_AA)
            
            # Optional: Calculate and print area (in sq meters)
            # Area in pixels
            # area_px = hull.volume # In 2D, volume is area
            # area_m2 = area_px / (PitchConfig.X_SCALE * PitchConfig.Y_SCALE)
            # print(f"Team {team.name} Compactness: {area_m2:.1f} m²")

        except Exception as e:
            # ConvexHull might fail if points are collinear
            pass
