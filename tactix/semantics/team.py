"""
Project: Tactix
File Created: 2026-02-02 12:13:14
Author: Xingnan Zhu
File Name: team.py
Description:
    Implements team classification logic using K-Means clustering.
    It extracts the dominant shirt color from detected players and groups them
    into two teams (Team A and Team B).
"""

from typing import List, Tuple, Optional

import numpy as np
from sklearn.cluster import KMeans

from tactix.core.types import Player, TeamID, FrameData


class TeamClassifier:
    def __init__(self, device='cpu'):
        self.device = device
        # Clusterer: We need to separate into 2 classes (Team A, Team B)
        self.kmeans = None
        self.team_colors = {} # {TeamID.A: (R,G,B), TeamID.B: (R,G,B)}

    def fit(self, frame: np.ndarray, players: List[Player]):
        """
        [Initialization] Called in the first few frames of the video.
        Collects shirt colors from all unknown players to train the K-Means model.
        """
        player_colors = []
        
        # Only select players not yet classified (avoid referee and goalkeeper if detector already found them)
        candidates = [p for p in players if p.team == TeamID.UNKNOWN]

        for p in candidates:
            color = self._extract_shirt_color(frame, p.rect)
            if color is not None:
                player_colors.append(color)
        
        if not player_colors:
            return

        # Train KMeans to split into 2 groups
        # n_init=10 means running multiple times to find optimal centroids
        data = np.array(player_colors)
        if len(data) < 2: 
            return # Not enough players to cluster

        self.kmeans = KMeans(n_clusters=2, n_init=10, random_state=0)
        self.kmeans.fit(data)
        
        # Save center colors for both teams for debugging/drawing
        self.team_colors[TeamID.A] = self.kmeans.cluster_centers_[0]
        self.team_colors[TeamID.B] = self.kmeans.cluster_centers_[1]
        
        print(f"✅ Team Colors Learned: A={self.team_colors[TeamID.A]}, B={self.team_colors[TeamID.B]}")

    def predict(self, frame: np.ndarray, frame_data: FrameData):
        """
        [Real-time] Called every frame. Assigns labels to UNKNOWN players in frame_data.
        """
        if self.kmeans is None:
            return # Not initialized yet

        for p in frame_data.players:
            # Only process players with unknown team
            if p.team == TeamID.UNKNOWN:
                color = self._extract_shirt_color(frame, p.rect)
                if color is not None:
                    # Predict class (0 or 1)
                    label = self.kmeans.predict([color])[0]
                    p.team = TeamID.A if label == 0 else TeamID.B

    @staticmethod
    def _extract_shirt_color(frame: np.ndarray, rect: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
        """
        Helper function: Extracts shirt color from bounding box.
        Technique: Only take the upper body center part to avoid grass, head, and shorts.
        """
        x1, y1, x2, y2 = map(int, rect)
        
        # Boundary check
        h_img, w_img, _ = frame.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)

        # Crop image
        img = frame[y1:y2, x1:x2]
        h, w, _ = img.shape
        if h < 5 or w < 5: return None # Too small

        # --- Key Strategy: Crop Upper Body Center ---
        # y: From 15% to 50% (Avoid head and shorts)
        # x: From 25% to 75% (Avoid background grass)
        crop = img[int(h*0.15):int(h*0.50), int(w*0.25):int(w*0.75)]
        
        if crop.size == 0: return None

        # Calculate average color
        avg_color_row = np.average(crop, axis=0)
        avg_color = np.average(avg_color_row, axis=0)
        
        return avg_color
