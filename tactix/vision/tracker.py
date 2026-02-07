"""
Project: Tactix
File Created: 2026-02-02 12:13:21
Author: Xingnan Zhu
File Name: tracker.py
Description:
    Implements object tracking using ByteTrack.
    It assigns unique IDs to detected players across frames and calculates
    their velocity vectors based on position changes.
"""


import supervision as sv
from tactix.core.types import FrameData, Point
import numpy as np
from collections import deque

class Tracker:
    def __init__(self):
        # ByteTrack parameters adapted for latest supervision version
        # track_activation_threshold: Only boxes with confidence > 0.25 are tracked
        # minimum_matching_threshold: Matching similarity threshold
        # lost_track_buffer: Keep ID for 30 frames after loss
        self.tracker = sv.ByteTrack(
            frame_rate=30,
            track_activation_threshold=0.1,
            minimum_matching_threshold=0.8,
            lost_track_buffer=30
        )
        
        # Velocity calculation cache: {player_id: deque([(frame_idx, x, y), ...])}
        # Stores the last 5 frames of positions for smooth velocity calculation
        self.position_history = {} 

    def update(self, detections: sv.Detections, frame_data: FrameData):
        """
        Uses IoU matching to accurately assign tracker IDs back to frame_data.players.
        Also calculates velocity vectors.
        """
        # 1. Get tracking results
        tracked_detections = self.tracker.update_with_detections(detections)
        
        if len(tracked_detections.tracker_id) == 0:
            return

        # 2. Matching logic: Assign IDs from tracked_detections to frame_data.players
        # We cannot rely on exact coordinate equality (float error), so we use IoU overlap
        
        # Extract coordinates
        tracked_xyxy = tracked_detections.xyxy
        original_xyxy = np.array([p.rect for p in frame_data.players])

        # If no original detections, return
        if len(original_xyxy) == 0:
            return

        # Calculate IoU matrix (Supervision tools can be tricky, so we implement simple matching)
        # Logic: For each tracked_box, find the original_box with highest overlap
        
        for i, t_box in enumerate(tracked_xyxy):
            t_id = tracked_detections.tracker_id[i]
            
            # Calculate IoU between t_box and all original_boxes
            ious = self._box_iou_batch(t_box, original_xyxy)
            
            # Find index with highest overlap
            best_match_idx = np.argmax(ious)
            max_iou = ious[best_match_idx]

            # If overlap > 0.5, consider it the same person and assign ID
            if max_iou > 0.5:
                player = frame_data.players[best_match_idx]
                player.id = t_id
                
                # --- Calculate Velocity Vector ---
                # Only possible if player has pitch_position (real coordinates)
                # But Tracker runs before Transformer, so we can only store pixel coords or wait for next frame
                # Better approach: Tracker only handles IDs, velocity calculation moves to system.py after Stage 3,
                # or store here but pass pitch_position.
                # Due to timing, we cannot calculate m/s velocity here directly.
                # 
                # Alternative: In system.py, after Transformer calculates pitch_position,
                # call tracker.update_velocity(frame_data)
                pass

    def update_velocity(self, frame_data: FrameData):
        """
        Call this method after Transformer calculates pitch_position to compute velocity.
        """
        current_frame = frame_data.frame_index
        dt = 1.0 / 30.0 # Assume 30fps, 0.033s per frame
        
        for p in frame_data.players:
            if p.id == -1 or p.pitch_position is None:
                continue
                
            pid = p.id
            cx, cy = p.pitch_position.x, p.pitch_position.y
            
            # Initialize history
            if pid not in self.position_history:
                self.position_history[pid] = deque(maxlen=5)
            
            history = self.position_history[pid]
            history.append((current_frame, cx, cy))
            
            # Need at least 3 frames of datasets for stable velocity
            if len(history) >= 3:
                # Calculate displacement between oldest frame and current frame
                # e.g., history[0] is t-4, history[-1] is t
                start_frame, sx, sy = history[0]
                end_frame, ex, ey = history[-1]
                
                time_diff = (end_frame - start_frame) * dt
                if time_diff > 0:
                    vx = (ex - sx) / time_diff
                    vy = (ey - sy) / time_diff
                    
                    # Simple low-pass filter: Limit max speed (e.g., Bolt is ~12m/s)
                    speed = np.sqrt(vx**2 + vy**2)
                    if speed < 15.0: 
                        p.velocity = Point(x=vx, y=vy)
                        p.speed = speed
                    else:
                        # Outlier, likely a jump, keep previous velocity or zero
                        p.velocity = Point(0, 0)
                        p.speed = 0

    @staticmethod
    def _box_iou_batch(box_a, boxes_b):
        """
        Calculate IoU between one box (box_a) and a set of boxes (boxes_b).
        """
        # box_a: [x1, y1, x2, y2]
        # boxes_b: [[x1, y1, x2, y2], ...]
        
        x_a = np.maximum(box_a[0], boxes_b[:, 0])
        y_a = np.maximum(box_a[1], boxes_b[:, 1])
        x_b = np.minimum(box_a[2], boxes_b[:, 2])
        y_b = np.minimum(box_a[3], boxes_b[:, 3])

        inter_area = np.maximum(0, x_b - x_a) * np.maximum(0, y_b - y_a)

        box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        boxes_b_area = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

        iou = inter_area / (box_a_area + boxes_b_area - inter_area)
        return iou