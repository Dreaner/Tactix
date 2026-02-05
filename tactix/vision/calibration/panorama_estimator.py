"""
Project: Tactix
File Created: 2026-02-05 17:46:31
Author: Xingnan Zhu
File Name: panorama_estimator.py
Description:
    Implements a Panorama-based pitch estimator.
    It uses manual calibration for the first frame, then tracks global camera motion
    (using ECC or Feature Matching) to update the homography matrix for subsequent frames.
    This allows tracking even when the original keypoints move out of view.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
from tactix.vision.calibration.base import BasePitchEstimator
from tactix.core.geometry import WORLD_POINTS
from tactix.core.keypoints import YOLO_INDEX_MAP, to_px

class PanoramaPitchEstimator(BasePitchEstimator):
    def __init__(self, initial_points: List[Tuple[int, int, int]]):
        """
        :param initial_points: List of (x, y, keypoint_index) from manual calibration
        """
        print(f"🌐 Initializing Panorama Pitch Estimator...")
        self.initial_points = initial_points
        self.prev_frame_gray = None
        self.H_curr = None # Current Homography Matrix (World -> Screen)
        
        # Calculate initial H from manual points
        self._init_homography()

    def _init_homography(self):
        if len(self.initial_points) < 4:
            print("❌ Not enough points for Panorama initialization!")
            return

        src_pts = [] # Screen points (from manual click)
        dst_pts = [] # World points (meters)

        for x, y, idx in self.initial_points:
            name = YOLO_INDEX_MAP.get(idx)
            if name and name in WORLD_POINTS:
                src_pts.append([x, y])
                # Note: We map to World Coordinates (Meters) directly here, 
                # because we want H to be World -> Screen (or Screen -> World)
                # But wait, ViewTransformer expects Screen Keypoints -> Tactical Board Pixels
                # Let's stick to the standard pipeline: Output Keypoints on Screen.
                # So we don't need H here? 
                # Actually, for Panorama, we maintain H_total = H_initial * H_motion
                pass
        
        # Strategy:
        # We don't need to calculate H here. We just need to track camera motion.
        # But to output 27 keypoints, we need to know where they are.
        # So we DO need an initial H to project all 27 points onto the first frame.
        
        # Let's calculate H_world_to_screen for frame 0
        # src: World (Meters), dst: Screen (Pixels)
        world_pts = []
        screen_pts = []
        for x, y, idx in self.initial_points:
            name = YOLO_INDEX_MAP.get(idx)
            if name in WORLD_POINTS:
                wx, wy = WORLD_POINTS[name]
                world_pts.append([wx, wy])
                screen_pts.append([x, y])
                
        if len(screen_pts) >= 4:
            world_arr = np.array(world_pts).reshape(-1, 1, 2)
            screen_arr = np.array(screen_pts).reshape(-1, 1, 2)
            self.H_curr, _ = cv2.findHomography(world_arr, screen_arr, cv2.RANSAC, 5.0)
            print("✅ Initial Homography Calculated.")

    def predict(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        curr_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. First Frame
        if self.prev_frame_gray is None:
            self.prev_frame_gray = curr_frame_gray
            return self._project_all_keypoints(frame.shape)

        # 2. Calculate Global Motion (Frame t-1 -> Frame t)
        # We use ECC (Enhanced Correlation Coefficient) for high accuracy, or ORB for speed.
        # ECC is slow but accurate for small motions. ORB+RANSAC is robust for large motions.
        # Let's use ORB features + RANSAC for robustness and speed.
        
        # Detect features
        orb = cv2.ORB_create(nfeatures=500)
        kp1, des1 = orb.detectAndCompute(self.prev_frame_gray, None)
        kp2, des2 = orb.detectAndCompute(curr_frame_gray, None)
        
        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            # Motion estimation failed, keep previous H
            pass
        else:
            # Match features
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Keep top matches
            good_matches = matches[:int(len(matches)*0.5)]
            
            if len(good_matches) >= 4:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # Calculate transform M (Frame t-1 -> Frame t)
                # We assume the pitch is a plane, so Homography is appropriate
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                if M is not None:
                    # Update Global H: H_new = M * H_old
                    # World -> Screen_t = (Screen_t-1 -> Screen_t) * (World -> Screen_t-1)
                    self.H_curr = np.dot(M, self.H_curr)

        # Update history
        self.prev_frame_gray = curr_frame_gray
        
        return self._project_all_keypoints(frame.shape)

    def _project_all_keypoints(self, frame_shape) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project all 27 world keypoints onto the current screen using H_curr.
        """
        if self.H_curr is None:
            return np.zeros((27, 2)), np.zeros(27)

        output_xy = np.zeros((27, 2), dtype=float)
        output_conf = np.zeros(27, dtype=float)
        
        h, w = frame_shape[:2]
        
        # Iterate all standard keypoints
        for idx, name in YOLO_INDEX_MAP.items():
            if name in WORLD_POINTS:
                wx, wy = WORLD_POINTS[name]
                
                # Project World -> Screen
                pt_arr = np.array([[[wx, wy]]], dtype=np.float32)
                try:
                    projected = cv2.perspectiveTransform(pt_arr, self.H_curr)[0][0]
                    px, py = projected[0], projected[1]
                    
                    # Check if point is within frame (or close to it)
                    # We can return points slightly outside frame too, Transformer handles it
                    output_xy[idx] = [px, py]
                    
                    # Confidence: 1.0 if inside frame, 0.5 if outside but tracked
                    if 0 <= px < w and 0 <= py < h:
                        output_conf[idx] = 1.0
                    else:
                        output_conf[idx] = 0.5 
                        
                except Exception:
                    output_conf[idx] = 0.0
                    
        return output_xy, output_conf