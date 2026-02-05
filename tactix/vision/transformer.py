"""
Project: Tactix
File Created: 2026-02-02 23:22:57
Author: Xingnan Zhu
File Name: transformer.py
Description:
    Handles the perspective transformation (Homography) between the video frame
    and the 2D tactical board. It maps detected keypoints to their real-world
    counterparts to compute the transformation matrix.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple
from tactix.core.types import PitchConfig, Player
from tactix.core.keypoints import YOLO_INDEX_MAP 
from tactix.core.geometry import WORLD_POINTS

class ViewTransformer:
    def __init__(self):
        self.homography_matrix = None
        self.scale_x = PitchConfig.PIXEL_WIDTH / PitchConfig.LENGTH
        self.scale_y = PitchConfig.PIXEL_HEIGHT / PitchConfig.WIDTH

    def update(self, keypoints: np.ndarray, confs: np.ndarray, threshold: float = 0.5) -> bool:
        """
        Attempts to update the homography matrix.
        Returns: bool (Whether a valid matrix is available, new or old)
        """
        if keypoints is None: 
            # If no points, check if we have an old matrix to fallback on
            return self.homography_matrix is not None

        src_pts = [] 
        dst_pts = [] 

        for i, (x, y) in enumerate(keypoints):
            if confs[i] < threshold: continue
            
            name = YOLO_INDEX_MAP.get(i)
            if name and name in WORLD_POINTS:
                src_pts.append([x, y])
                world_x, world_y = WORLD_POINTS[name]
                target_x = int(world_x * self.scale_x)
                target_y = int(world_y * self.scale_y)
                dst_pts.append([target_x, target_y])

        # 🔥 Core modification: If not enough points, don't crash, don't clear, just use the old matrix
        if len(src_pts) < 4:
            return self.homography_matrix is not None

        src_arr = np.array(src_pts).reshape(-1, 1, 2)
        dst_arr = np.array(dst_pts).reshape(-1, 1, 2)

        # RANSAC Calculation

        h, mask = cv2.findHomography(src_arr, dst_arr, cv2.RANSAC, 5.0)
        
        if h is not None:
             # Secondary validation
             inliers = np.sum(mask)
             if inliers >= 4:
                 self.homography_matrix = h # Update to new matrix
                 return True
        
        # If new calculation is bad, continue using old one
        return self.homography_matrix is not None

    def transform_point(self, xy: Tuple[float, float]) -> Optional[Tuple[int, int]]:
        # As long as there is a matrix (even an old one), calculate it!
        if self.homography_matrix is None: return None
        
        point_arr = np.array([[[xy[0], xy[1]]]], dtype=np.float32)
        try:
            transformed = cv2.perspectiveTransform(point_arr, self.homography_matrix)[0][0]
            
            # 🔥 Extra protection: Check if coordinates flew off the earth
            # If calculated coordinates are negative or huge, matrix is bad, return None to avoid drawing errors
            tx, ty = int(transformed[0]), int(transformed[1])
            if -500 < tx < 3000 and -500 < ty < 2000: # Tolerant boundaries
                return tx, ty
        except cv2.error as e:
            print(f"OpenCV Transformation Error: {e}")
        except Exception as e:
            print(f"Unexpected Transformation Error: {e}")
            
        return None

    def transform_players(self, players: List[Player]):
        for p in players:
            # Use bottom_center (feet) for more accurate transformation, otherwise use center
            # Assuming Player.rect is [x1, y1, x2, y2]
            # anchor_x = (x1 + x2) / 2
            # anchor_y = y2 (feet)
            result = self.transform_point(p.anchor)
            
            if result:
                # Assignment depends on Point definition in types.py
                # If p.pitch_position is Point type:
                from tactix.core.types import Point
                p.pitch_position = Point(x=result[0], y=result[1])
            else:
                p.pitch_position = None