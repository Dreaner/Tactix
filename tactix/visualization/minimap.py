"""
Project: Tactix
File Created: 2026-02-03 09:54:59
Author: Xingnan Zhu
File Name: minimap.py
Description:
    Renders the 2D tactical minimap.
    It draws the pitch background, players, ball, and overlays advanced
    visualizations like Voronoi diagrams, Heatmaps, and velocity vectors.
"""


import cv2
import numpy as np

from tactix.core.types import FrameData, TeamID, PitchConfig
from tactix.config import Colors


class MinimapRenderer:
    def __init__(self, bg_image_path: str):
        # Load background image
        self.bg_image = cv2.imread(bg_image_path)
        if self.bg_image is None:
            # If image not found, create a default green background
            w, h = PitchConfig.PIXEL_WIDTH, PitchConfig.PIXEL_HEIGHT
            print(f"⚠️ Warning: Minimap background not found at {bg_image_path}. Using green canvas.")
            self.bg_image = np.zeros((h, w, 3), dtype=np.uint8)
            self.bg_image[:] = (50, 150, 50) # Green
            
        self.h, self.w = self.bg_image.shape[:2]
        
        # Force update PitchConfig dimensions to match the actual loaded image
        if self.w != PitchConfig.PIXEL_WIDTH or self.h != PitchConfig.PIXEL_HEIGHT:
            print(f"⚠️ Updating PitchConfig dimensions from {PitchConfig.PIXEL_WIDTH}x{PitchConfig.PIXEL_HEIGHT} to {self.w}x{self.h}")
            PitchConfig.PIXEL_WIDTH = self.w
            PitchConfig.PIXEL_HEIGHT = self.h
            # Recalculate scale
            PitchConfig.X_SCALE = self.w / PitchConfig.LENGTH
            PitchConfig.Y_SCALE = self.h / PitchConfig.WIDTH
            
        # Predefined color map (BGR)
        self.colors = {
            TeamID.A: Colors.to_bgr(Colors.TEAM_A),
            TeamID.B: Colors.to_bgr(Colors.TEAM_B),
            TeamID.GOALKEEPER: Colors.to_bgr(Colors.GOALKEEPER),
            TeamID.REFEREE: Colors.to_bgr(Colors.REFEREE),
            TeamID.UNKNOWN: Colors.to_bgr(Colors.UNKNOWN)
        }

    def draw(self, frame_data: FrameData, voronoi_overlay: np.ndarray = None, heatmap_overlay: np.ndarray = None) -> np.ndarray:
        """
        Draws the minimap for the current frame.
        :param frame_data: Frame data
        :param voronoi_overlay: Pre-calculated Voronoi RGBA layer (optional)
        :param heatmap_overlay: Pre-calculated Heatmap RGBA layer (optional)
        """
        # Copy background
        minimap = self.bg_image.copy()

        # 0. Overlay Heatmap layer (if any) - Bottom layer
        if heatmap_overlay is not None:
            self._overlay_image(minimap, heatmap_overlay)

        # 0.5 Overlay Voronoi layer (if any)
        if voronoi_overlay is not None:
            self._overlay_image(minimap, voronoi_overlay)

        # 1. Draw Players
        for p in frame_data.players:
            if p.pitch_position:
                # Convert coordinates: pitch_position is already in pixels
                mx = int(p.pitch_position.x)
                my = int(p.pitch_position.y)
                
                color = self.colors.get(p.team, self.colors[TeamID.UNKNOWN])
                
                # --- A. Draw Velocity Vector ---
                if p.velocity:
                    # Velocity vector length scale factor (e.g., 1m/s drawn as 20px long)
                    scale_factor = 20.0 
                    vx_px = int(p.velocity.x * scale_factor)
                    vy_px = int(p.velocity.y * scale_factor)
                    
                    # Only draw if speed is significant
                    if abs(vx_px) > 2 or abs(vy_px) > 2:
                        end_x = mx + vx_px
                        end_y = my + vy_px
                        cv2.arrowedLine(minimap, (mx, my), (end_x, end_y), Colors.to_bgr(Colors.TEXT), 2, tipLength=0.3)

                # --- B. Draw Dot ---
                # Outer circle (White border)
                cv2.circle(minimap, (mx, my), 14, Colors.to_bgr(Colors.TEXT), -1)
                # Inner circle (Team color)
                cv2.circle(minimap, (mx, my), 12, color, -1)
                
                # Draw Number
                if p.id != -1:
                    text = str(p.id)
                    font_scale = 0.8
                    thickness = 2
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    # Center text
                    tx = mx - tw // 2
                    ty = my + th // 2 - 2
                    
                    # Text color: White for most, Black for light backgrounds (like Yellow Referee)
                    text_color = Colors.to_bgr(Colors.TEXT)
                    if p.team == TeamID.REFEREE: 
                        text_color = (0, 0, 0)
                    
                    cv2.putText(minimap, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

        # 2. Draw Ball
        if frame_data.ball and frame_data.ball.pitch_position:
            bx = int(frame_data.ball.pitch_position.x)
            by = int(frame_data.ball.pitch_position.y)
            
            ball_color = Colors.to_bgr(Colors.BALL)
            
            # Shadow
            cv2.circle(minimap, (bx+2, by+2), 10, (0, 0, 0, 100), -1)
            # Body
            cv2.circle(minimap, (bx, by), 10, ball_color, -1) 
            cv2.circle(minimap, (bx, by), 10, (0, 0, 0), 2)      # Black border

        return minimap

    @staticmethod
    def _overlay_image(background, overlay):
        """
        Helper function to overlay an RGBA image onto an RGB background
        """
        # Check dimensions match
        bg_h, bg_w = background.shape[:2]
        ov_h, ov_w = overlay.shape[:2]
        
        if bg_h != ov_h or bg_w != ov_w:
            overlay = cv2.resize(overlay, (bg_w, bg_h))

        alpha_channel = overlay[:, :, 3] / 255.0
        rgb_channels = overlay[:, :, :3]
        
        for c in range(3):
            background[:, :, c] = (rgb_channels[:, :, c] * alpha_channel + 
                                   background[:, :, c] * (1 - alpha_channel))
