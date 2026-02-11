"""
Project: Tactix
File Created: 2026-02-11
Author: Xingnan Zhu
File Name: cover_shadow.py  (overlay renderer)
Description:
    Cover shadow (defensive cone) overlay renderer.
    Draws triangular shadow cones behind defenders relative to the ball carrier.
"""

import cv2
import numpy as np

from tactix.core.types import FrameData, PitchConfig, TeamID


class CoverShadowOverlay:
    def __init__(self, shadow_length: float = 20.0, shadow_angle: float = 20.0) -> None:
        self.shadow_length = shadow_length
        self.shadow_angle = np.radians(shadow_angle)

    def render(self, frame_data: FrameData) -> np.ndarray:
        w, h = PitchConfig.PIXEL_WIDTH, PitchConfig.PIXEL_HEIGHT
        overlay = np.zeros((h, w, 4), dtype=np.uint8)

        ball = frame_data.ball
        if not ball or not ball.pitch_position or ball.owner_id is None:
            return overlay

        owner = frame_data.get_player_by_id(ball.owner_id)
        if not owner or not owner.pitch_position:
            return overlay

        bx, by = owner.pitch_position.x, owner.pitch_position.y
        defenders = [
            p for p in frame_data.players
            if p.team != owner.team and p.team in (TeamID.A, TeamID.B) and p.pitch_position
        ]

        for d in defenders:
            dx, dy = d.pitch_position.x, d.pitch_position.y
            vec_x, vec_y = dx - bx, dy - by
            dist = np.sqrt(vec_x ** 2 + vec_y ** 2)
            if dist < 0.1:
                continue

            ux, uy = vec_x / dist, vec_y / dist
            half = self.shadow_angle / 2
            cos_a, sin_a = np.cos(half), np.sin(half)

            e1x = ux * cos_a - uy * sin_a
            e1y = ux * sin_a + uy * cos_a
            e2x = ux * cos_a + uy * sin_a
            e2y = -ux * sin_a + uy * cos_a

            v1 = (dx, dy)
            v2 = (dx + e1x * self.shadow_length, dy + e1y * self.shadow_length)
            v3 = (dx + e2x * self.shadow_length, dy + e2y * self.shadow_length)

            tri = np.array([
                [int(v1[0] * PitchConfig.X_SCALE), int(v1[1] * PitchConfig.Y_SCALE)],
                [int(v2[0] * PitchConfig.X_SCALE), int(v2[1] * PitchConfig.Y_SCALE)],
                [int(v3[0] * PitchConfig.X_SCALE), int(v3[1] * PitchConfig.Y_SCALE)],
            ], dtype=np.int32)

            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [tri], 255)
            overlay[mask == 255] = (50, 50, 50, 100)

        return overlay
