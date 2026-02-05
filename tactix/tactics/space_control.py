"""
Project: Tactix
File Created: 2026-02-04 17:00:00
Author: Xingnan Zhu
File Name: space_control.py
Description:
    Implements advanced tactical analysis algorithms including Voronoi diagrams
    for space control visualization. It calculates which player controls which
    area of the pitch based on their positions.
"""

import cv2
import numpy as np
from scipy.spatial import Voronoi
from typing import List, Tuple, Dict

from tactix.core.types import FrameData, TeamID, PitchConfig
from tactix.config import Colors

class SpaceControl:
    def __init__(self):
        # Tactical board dimensions
        self.w = PitchConfig.PIXEL_WIDTH
        self.h = PitchConfig.PIXEL_HEIGHT
        
        # Define pitch boundary rectangle (for clipping Voronoi)
        self.rect = (0, 0, self.w, self.h)

    def generate_voronoi(self, frame_data: FrameData) -> np.ndarray:
        """
        Generate Voronoi overlay (RGBA), can be directly superimposed on the minimap
        """
        # 1. Collect all players with position info
        points = []
        colors = []
        
        # Team colors (BGR) - Semi-transparent fill
        COLOR_A = Colors.to_bgr(Colors.TEAM_A)
        COLOR_B = Colors.to_bgr(Colors.TEAM_B)
        
        for p in frame_data.players:
            if p.pitch_position and p.team in [TeamID.A, TeamID.B]:
                # Convert to tactical board pixel coordinates
                px = int(p.pitch_position.x)
                py = int(p.pitch_position.y)
                
                # Boundary protection
                px = np.clip(px, 0, self.w - 1)
                py = np.clip(py, 0, self.h - 1)
                
                points.append([px, py])
                colors.append(COLOR_A if p.team == TeamID.A else COLOR_B)

        # If too few players, cannot draw
        if len(points) < 4:
            return np.zeros((self.h, self.w, 4), dtype=np.uint8)

        # 2. Calculate Voronoi
        subdiv = cv2.Subdiv2D(self.rect)
        for p in points:
            subdiv.insert((float(p[0]), float(p[1])))
            
        # Get Voronoi cells
        (facets, centers) = subdiv.getVoronoiFacetList([])

        # 3. Draw
        overlay = np.zeros((self.h, self.w, 4), dtype=np.uint8)
        
        for i, facet in enumerate(facets):
            cx, cy = centers[i]
            min_dist = float('inf')
            match_idx = -1
            
            for j, pt in enumerate(points):
                dist = (pt[0]-cx)**2 + (pt[1]-cy)**2
                if dist < min_dist:
                    min_dist = dist
                    match_idx = j
            
            if match_idx != -1:
                color = colors[match_idx]
                facet_poly = np.array(facet, dtype=np.int32)
                
                # Fill semi-transparent color (Alpha = 100)
                mask = np.zeros((self.h, self.w), dtype=np.uint8)
                cv2.fillPoly(mask, [facet_poly], 255)
                
                # Assign color
                overlay[mask == 255] = (*color, 80) # Alpha = 80 (Faint)
                
                # Draw boundary lines (White, thin)
                cv2.polylines(overlay, [facet_poly], True, (255, 255, 255, 150), 1, cv2.LINE_AA)

        return overlay
