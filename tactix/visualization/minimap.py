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


class MinimapRenderer:
    def __init__(self, bg_image_path: str):
        # 加载背景图
        self.bg_image = cv2.imread(bg_image_path)
        if self.bg_image is None:
            # 如果找不到图，创建一个默认的绿色背景
            w, h = PitchConfig.PIXEL_WIDTH, PitchConfig.PIXEL_HEIGHT
            print(f"⚠️ Warning: Minimap background not found at {bg_image_path}. Using green canvas.")
            self.bg_image = np.zeros((h, w, 3), dtype=np.uint8)
            self.bg_image[:] = (50, 150, 50) # Green
            
        self.h, self.w = self.bg_image.shape[:2]
        
        # 强制更新 PitchConfig 的尺寸，以匹配实际加载的图片
        # 这样后续的计算（如 Voronoi, Heatmap）都会基于正确的尺寸
        if self.w != PitchConfig.PIXEL_WIDTH or self.h != PitchConfig.PIXEL_HEIGHT:
            print(f"⚠️ Updating PitchConfig dimensions from {PitchConfig.PIXEL_WIDTH}x{PitchConfig.PIXEL_HEIGHT} to {self.w}x{self.h}")
            PitchConfig.PIXEL_WIDTH = self.w
            PitchConfig.PIXEL_HEIGHT = self.h
            # 重新计算比例尺
            PitchConfig.X_SCALE = self.w / PitchConfig.LENGTH
            PitchConfig.Y_SCALE = self.h / PitchConfig.WIDTH
            
        # 预定义的颜色表 (BGR)
        self.colors = {
            TeamID.A: (70, 57, 230),       # 红 (BGR: 230, 57, 70 -> RGB) -> 这里用 BGR: (70, 57, 230)
            TeamID.B: (157, 123, 69),      # 蓝 (BGR: 69, 123, 157 -> RGB) -> 这里用 BGR: (157, 123, 69)
            TeamID.GOALKEEPER: (0, 255, 255), # 黄
            TeamID.REFEREE: (50, 50, 50),   # 深灰
            TeamID.UNKNOWN: (200, 200, 200) # 浅灰
        }

    def draw(self, frame_data: FrameData, voronoi_overlay: np.ndarray = None, heatmap_overlay: np.ndarray = None) -> np.ndarray:
        """
        绘制当前帧的小地图
        :param frame_data: 帧数据
        :param voronoi_overlay: 预先计算好的 Voronoi RGBA 图层 (可选)
        :param heatmap_overlay: 预先计算好的 Heatmap RGBA 图层 (可选)
        """
        # 复制背景
        minimap = self.bg_image.copy()

        # 0. 叠加 Heatmap 图层 (如果有) - 放在最底层
        if heatmap_overlay is not None:
            self._overlay_image(minimap, heatmap_overlay)

        # 0.5 叠加 Voronoi 图层 (如果有)
        if voronoi_overlay is not None:
            self._overlay_image(minimap, voronoi_overlay)

        # 1. 画球员
        for p in frame_data.players:
            if p.pitch_position:
                # 转换坐标: 这里的 pitch_position 已经是像素坐标了，不需要再乘比例尺
                # 因为 ViewTransformer 使用的 KEY_POINTS 是像素单位
                mx = int(p.pitch_position.x)
                my = int(p.pitch_position.y)
                
                color = self.colors.get(p.team, self.colors[TeamID.UNKNOWN])
                
                # --- A. 画速度向量 (Velocity Vector) ---
                # 假设 p.velocity 已经是 (vx, vy) 单位是 m/s
                # 我们把它放大一点画出来
                if p.velocity:
                    # 速度向量长度放大系数 (比如 1m/s 画成 20px 长)
                    scale_factor = 20.0 
                    vx_px = int(p.velocity.x * scale_factor)
                    vy_px = int(p.velocity.y * scale_factor)
                    
                    # 只有速度够大才画
                    if abs(vx_px) > 2 or abs(vy_px) > 2:
                        end_x = mx + vx_px
                        end_y = my + vy_px
                        cv2.arrowedLine(minimap, (mx, my), (end_x, end_y), (255, 255, 255), 2, tipLength=0.3)

                # --- B. 画圆点 ---
                # 外圈 (白色描边)
                cv2.circle(minimap, (mx, my), 14, (255, 255, 255), -1)
                # 内圈 (队伍颜色)
                cv2.circle(minimap, (mx, my), 12, color, -1)
                
                # 画号码
                if p.id != -1:
                    text = str(p.id)
                    font_scale = 0.8
                    thickness = 2
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    # 文字居中
                    tx = mx - tw // 2
                    ty = my + th // 2 - 2
                    # 只有当颜色比较深时用白字，否则用黑字 (这里简化全用白字，除了裁判)
                    text_color = (255, 255, 255)
                    if p.team == TeamID.GOALKEEPER: text_color = (0, 0, 0)
                    
                    cv2.putText(minimap, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

        # 2. 画球
        if frame_data.ball and frame_data.ball.pitch_position:
            # 同样，球的坐标也是像素坐标
            bx = int(frame_data.ball.pitch_position.x)
            by = int(frame_data.ball.pitch_position.y)
            
            # 阴影
            cv2.circle(minimap, (bx+2, by+2), 10, (0, 0, 0, 100), -1)
            # 本体
            cv2.circle(minimap, (bx, by), 10, (0, 255, 255), -1) # 黄球
            cv2.circle(minimap, (bx, by), 10, (0, 0, 0), 2)      # 黑边

        return minimap

    def _overlay_image(self, background, overlay):
        """
        Helper function to overlay an RGBA image onto an RGB background
        """
        # 检查尺寸是否匹配，如果不匹配则调整 overlay 大小
        bg_h, bg_w = background.shape[:2]
        ov_h, ov_w = overlay.shape[:2]
        
        if bg_h != ov_h or bg_w != ov_w:
            # print(f"⚠️ Resizing overlay from {ov_w}x{ov_h} to {bg_w}x{bg_h}")
            overlay = cv2.resize(overlay, (bg_w, bg_h))

        alpha_channel = overlay[:, :, 3] / 255.0
        rgb_channels = overlay[:, :, :3]
        
        for c in range(3):
            background[:, :, c] = (rgb_channels[:, :, c] * alpha_channel + 
                                   background[:, :, c] * (1 - alpha_channel))
