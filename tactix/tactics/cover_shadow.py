"""
Project: Tactix
File Created: 2026-02-07 15:43:16
Author: Xingnan Zhu
File Name: cover_shadow.py
Description:
    Calculates the Cover Shadow (Pass Blocking) for defensive players.
    It identifies the area behind a defender that is blocked from the ball carrier's perspective.
"""

import numpy as np
import cv2
from typing import List, Tuple
from tactix.core.types import FrameData, TeamID, PitchConfig
from tactix.config import Colors

class CoverShadow:
    def __init__(self, shadow_length: float = 20.0, shadow_angle: float = 20.0):
        """
        :param shadow_length: Length of the shadow in meters.
        :param shadow_angle: Spread angle of the shadow cone in degrees.
        """
        self.shadow_length = shadow_length
        self.shadow_angle = np.radians(shadow_angle)
        self.w = PitchConfig.PIXEL_WIDTH
        self.h = PitchConfig.PIXEL_HEIGHT

    def generate_overlay(self, frame_data: FrameData) -> np.ndarray:
        """
        Generate Cover Shadow overlay (RGBA).
        """
        overlay = np.zeros((self.h, self.w, 4), dtype=np.uint8)
        
        # 1. Find Ball Carrier
        ball = frame_data.ball
        if not ball or not ball.pitch_position or ball.owner_id is None:
            return overlay
            
        owner = frame_data.get_player_by_id(ball.owner_id)
        if not owner or not owner.pitch_position:
            return overlay
            
        bx, by = owner.pitch_position.x, owner.pitch_position.y
        
        # 2. Identify Defenders
        # Defenders are players from the opposite team
        defenders = [p for p in frame_data.players 
                     if p.team != owner.team and p.team in [TeamID.A, TeamID.B] and p.pitch_position]
        
        # 3. Calculate Shadow for each defender
        for defender in defenders:
            dx, dy = defender.pitch_position.x, defender.pitch_position.y
            
            # Vector from Ball to Defender
            vec_x = dx - bx
            vec_y = dy - by
            dist = np.sqrt(vec_x**2 + vec_y**2)
            
            if dist < 0.1: continue # Too close
            
            # Normalize vector
            unit_x = vec_x / dist
            unit_y = vec_y / dist
            
            # Calculate shadow vertices (Triangle)
            # Vertex 1: Defender position
            v1 = (dx, dy)
            
            # Rotate vector by +angle/2 and -angle/2 to get cone edges
            cos_a = np.cos(self.shadow_angle / 2)
            sin_a = np.sin(self.shadow_angle / 2)
            
            # Edge 1 vector
            e1_x = unit_x * cos_a - unit_y * sin_a
            e1_y = unit_x * sin_a + unit_y * cos_a
            
            # Edge 2 vector
            e2_x = unit_x * cos_a + unit_y * sin_a
            e2_y = -unit_x * sin_a + unit_y * cos_a
            
            # Vertex 2 & 3: End of shadow
            v2 = (dx + e1_x * self.shadow_length, dy + e1_y * self.shadow_length)
            v3 = (dx + e2_x * self.shadow_length, dy + e2_y * self.shadow_length)
            
            # Convert to pixels
            triangle_cnt = np.array([
                [int(v1[0] * PitchConfig.X_SCALE), int(v1[1] * PitchConfig.Y_SCALE)],
                [int(v2[0] * PitchConfig.X_SCALE), int(v2[1] * PitchConfig.Y_SCALE)],
                [int(v3[0] * PitchConfig.X_SCALE), int(v3[1] * PitchConfig.Y_SCALE)]
            ], dtype=np.int32)
            
            # Draw Shadow
            # Color: Dark Grey/Black, Semi-transparent
            color = (50, 50, 50) 
            
            mask = np.zeros((self.h, self.w), dtype=np.uint8)
            cv2.fillPoly(mask, [triangle_cnt], 255)
            
            overlay[mask == 255] = (*color, 100) # Alpha = 100
            
        return overlay
