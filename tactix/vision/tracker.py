"""
Project: Tactix
File Created: 2026-02-02 12:13:21
Author: Xingnan Zhu
File Name: tracker.py
Description: xxx...
"""


import supervision as sv
from tactix.core.types import FrameData
import numpy as np

class Tracker:
    def __init__(self):
        # ByteTrack 参数名适配最新版 supervision
        # track_activation_threshold: 只有置信度高于 0.25 的框才参与跟踪
        # minimum_matching_threshold: 匹配相似度阈值
        # lost_track_buffer: 丢失 30 帧内还能找回 ID
        self.tracker = sv.ByteTrack(
            frame_rate=30,
            track_activation_threshold=0.1,
            minimum_matching_threshold=0.8,
            lost_track_buffer=30
        )

    def update(self, detections: sv.Detections, frame_data: FrameData):
        """
        使用 IoU 匹配，将跟踪器的 ID 准确填回 frame_data.players
        """
        # 1. 获取跟踪结果
        tracked_detections = self.tracker.update_with_detections(detections)
        
        if len(tracked_detections.tracker_id) == 0:
            return

        # 2. 建立匹配逻辑：把 tracked_detections 的 ID 赋给 frame_data.players
        # 我们不能直接依赖坐标相等 (float 误差)，我们要计算 IoU 重叠度
        
        # 提取两组坐标
        tracked_xyxy = tracked_detections.xyxy
        original_xyxy = np.array([p.rect for p in frame_data.players])

        # 如果没有原始检测，直接返回
        if len(original_xyxy) == 0:
            return

        # 计算 IoU 矩阵 (Supervision 内置工具有时候不好调，我们手写一个简单的匹配)
        # 这里的逻辑是：对于每一个 tracked_box，找到和它重叠度最高的 original_box
        
        for i, t_box in enumerate(tracked_xyxy):
            t_id = tracked_detections.tracker_id[i]
            
            # 计算 t_box 与所有 original_boxes 的 IoU
            ious = self._box_iou_batch(t_box, original_xyxy)
            
            # 找到重叠度最高的那个索引
            best_match_idx = np.argmax(ious)
            max_iou = ious[best_match_idx]

            # 如果重叠度大于 0.5，我们就认为是同一个人，把 ID 填进去
            if max_iou > 0.5:
                frame_data.players[best_match_idx].id = t_id

    @staticmethod
    def _box_iou_batch(box_a, boxes_b):
        """
        计算一个框 (box_a) 与 一组框 (boxes_b) 的 IoU
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