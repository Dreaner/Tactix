"""
Project: Tactix
File Created: 2026-02-02 12:13:14
Author: Xingnan Zhu
File Name: team.py
Description: xxx...
"""

from typing import List, Tuple, Optional

import numpy as np
from sklearn.cluster import KMeans

from tactix.core.types import Player, TeamID, FrameData


class TeamClassifier:
    def __init__(self, device='cpu'):
        self.device = device
        # 聚类器：我们需要分出 2 类 (Team A, Team B)
        self.kmeans = None
        self.team_colors = {} # {TeamID.A: (R,G,B), TeamID.B: (R,G,B)}

    def fit(self, frame: np.ndarray, players: List[Player]):
        """
        [初始化] 在视频前几帧调用。
        采集所有未知阵营球员的球衣颜色，训练 K-Means 模型。
        """
        player_colors = []
        
        # 只选取还未被分类的球员（避开裁判和门将，如果 detector 已经分出来的话）
        candidates = [p for p in players if p.team == TeamID.UNKNOWN]

        for p in candidates:
            color = self._extract_shirt_color(frame, p.rect)
            if color is not None:
                player_colors.append(color)
        
        if not player_colors:
            return

        # 训练 KMeans 分成 2 组
        # n_init=10 意味着多跑几次找最优解
        data = np.array(player_colors)
        if len(data) < 2: 
            return # 人太少没法聚类

        self.kmeans = KMeans(n_clusters=2, n_init=10, random_state=0)
        self.kmeans.fit(data)
        
        # 保存两个队的中心颜色，用于绘图或调试
        self.team_colors[TeamID.A] = self.kmeans.cluster_centers_[0]
        self.team_colors[TeamID.B] = self.kmeans.cluster_centers_[1]
        
        print(f"✅ Team Colors Learned: A={self.team_colors[TeamID.A]}, B={self.team_colors[TeamID.B]}")

    def predict(self, frame: np.ndarray, frame_data: FrameData):
        """
        [实时] 每一帧调用。给 frame_data 里还是 UNKNOWN 的球员打上标签。
        """
        if self.kmeans is None:
            return # 还没初始化

        for p in frame_data.players:
            # 只处理还不知道队伍的人
            if p.team == TeamID.UNKNOWN:
                color = self._extract_shirt_color(frame, p.rect)
                if color is not None:
                    # 预测类别 (0 或 1)
                    label = self.kmeans.predict([color])[0]
                    p.team = TeamID.A if label == 0 else TeamID.B

    @staticmethod
    def _extract_shirt_color(frame: np.ndarray, rect: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
        """
        辅助函数：从边界框中提取球衣颜色。
        技巧：只取上半身中间部分，避开草地、头和短裤。
        """
        x1, y1, x2, y2 = map(int, rect)
        
        # 边界检查
        h_img, w_img, _ = frame.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)

        # 裁剪出图像
        img = frame[y1:y2, x1:x2]
        h, w, _ = img.shape
        if h < 5 or w < 5: return None # 太小了

        # --- 关键策略：只取上半身中心区域 ---
        # y: 从 15% 处开始到 50% 处 (避开头和短裤)
        # x: 从 25% 处开始到 75% 处 (避开背景草地)
        crop = img[int(h*0.15):int(h*0.50), int(w*0.25):int(w*0.75)]
        
        if crop.size == 0: return None

        # 计算平均颜色
        avg_color_row = np.average(crop, axis=0)
        avg_color = np.average(avg_color_row, axis=0)
        
        return avg_color
