"""
Project: Tactix
File Created: 2026-02-02 12:12:35
Author: Xingnan Zhu
File Name: detector.py
Description: xxx...
"""

import numpy as np
from ultralytics import YOLO
import supervision as sv
from typing import Dict, Optional, List, Tuple

from tactix.core.types import Player, Ball, FrameData, TeamID

class Detector:
    def __init__(
        self, 
        model_weights: str, 
        device: str = 'mps',
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.7
    ):
        print(f"👁️ Loading Detector: {model_weights} on {device}...")
        self.model = YOLO(model_weights)
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # 类别映射 (根据你的模型)
        self.CLASS_MAP = {
            0: 'ball',
            1: 'goalkeeper',
            2: 'player',
            3: 'referee'
        }

    def detect(self, frame: np.ndarray, frame_index: int) -> FrameData:
        # 1. 开启 TTA 增强模式 + 高分辨率
        results = self.model(
            frame, 
            device=self.device, 
            verbose=False, 
            conf=self.conf_threshold, # 这里用基础阈值
            iou=self.iou_threshold,
            imgsz=1280,   # 高清模式
            augment=True  # TTA 增强
        )[0]
        
        detections = sv.Detections.from_ultralytics(results)
        frame_data = FrameData(frame_index=frame_index, image_shape=frame.shape[:2])

        # 临时列表：先把球和人分开存，最后再做“双标”判断
        ball_candidates = [] # 存 (rect, score)
        player_boxes = []    # 存 [x1, y1, x2, y2] 用于计算重叠

        # --- 第一遍循环：先处理所有物体 ---
        for i, class_id in enumerate(detections.class_id):
            xyxy = detections.xyxy[i]
            rect = tuple(xyxy.tolist())
            confidence = float(detections.confidence[i])
            class_name = self.CLASS_MAP.get(class_id, 'unknown')

            # [纠错逻辑] 宽高比过滤
            x1, y1, x2, y2 = xyxy
            width, height = x2 - x1, y2 - y1
            area = width * height
            ratio = width / height if height > 0 else 0

            # 纠错: 极小且方的东西 -> 强制认为是球
            if class_name != 'ball' and area < 900 and ratio > 0.7:
                class_name = 'ball'
            
            # 纠错: 太大或太扁的东西 -> 肯定不是球
            if class_name == 'ball':
                if area > 900 or ratio < 0.6 or ratio > 1.5:
                    continue

            # 分类存储
            if class_name == 'ball':
                ball_candidates.append((rect, confidence))
            elif class_name in ['player', 'goalkeeper', 'referee']:
                # 直接存入 frame_data
                player = Player(
                    id=-1,
                    rect=rect,
                    class_id=class_id,
                    team=TeamID.UNKNOWN
                )
                if class_name == 'referee': player.team = TeamID.REFEREE
                elif class_name == 'goalkeeper': player.team = TeamID.GOALKEEPER
                
                frame_data.players.append(player)
                player_boxes.append(xyxy) # 记录人的位置

        # --- 第二遍循环：用“双重标准”筛选球 ---
        best_ball = None
        best_score = -1.0

        for rect, score in ball_candidates:
            # 1. 检查这个球是不是在某人的脚下 (重叠检测)
            is_touching_player = False
            ball_x = (rect[0] + rect[2]) / 2
            ball_y = (rect[1] + rect[3]) / 2

            for p_box in player_boxes:
                # 简单判断：球心在人的框内，且靠下半部分
                px1, py1, px2, py2 = p_box
                if px1 < ball_x < px2 and py1 < ball_y < py2:
                    is_touching_player = True
                    break
            
            # 2. 动态阈值 (Dynamic Threshold)
            # 如果在人脚下，要求极高 (0.6)；如果在空地，要求极低 (0.1)
            threshold = 0.6 if is_touching_player else 0.1
            
            if score > threshold:
                if score > best_score:
                    best_score = score
                    best_ball = Ball(rect=rect, score=score)

        if best_ball:
            frame_data.ball = best_ball

        return frame_data