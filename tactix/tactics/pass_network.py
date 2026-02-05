"""
Project: Tactix
File Created: 2026-02-02 16:34:42
Author: Xingnan Zhu
File Name: pass_network.py
Description:
    Analyzes player positions and ball ownership to visualize potential
    passing networks. It calculates distances between players and the ball
    to determine the ball carrier and draws passing lines to teammates.
"""


from typing import List, Tuple, Optional

import numpy as np

from tactix.core.types import FrameData, Player


class PassNetwork:
    def __init__(self, max_pass_dist=300, ball_owner_dist=50):
        self.max_pass_dist = max_pass_dist # Max pixel distance to draw a line
        self.ball_owner_dist = ball_owner_dist # How close the ball must be to be considered "owned"
    
    def analyze(self, frame_data: FrameData) -> List[Tuple[Tuple[int,int], Tuple[int,int], float]]:
        """
        Returns a list of lines to draw: [(start_xy, end_xy, opacity), ...]
        """
        if not frame_data.ball or not frame_data.players:
            return []

        ball_center = np.array(frame_data.ball.center)
        # === Debug Print 1 ===
        print(f"Ball detected at {ball_center}")

        owner: Optional[Player] = None
        min_dist = float('inf')

        for p in frame_data.players:
            # Must be close enough to the ball, and we must know their team
            # if p.team == TeamID.UNKNOWN or p.team == TeamID.REFEREE:
            #     continue
                
            # Use anchor point (feet) for more accurate distance
            dist = np.linalg.norm(np.array(p.anchor) - ball_center)
            if dist < min_dist:
                min_dist = dist
                owner = p

        # === Debug Print 2 ===
        print(f"Nearest player dist: {min_dist:.1f} px")
        
        # Record owner ID in the ball object
        frame_data.ball.owner_id = owner.id
        
        # Relax distance limit! Change from 50 to 100 or even 150 to test
        # self.ball_owner_dist can be passed in system.py, or hardcoded here for testing
        effective_limit = max(self.ball_owner_dist, 100) 
        
        if min_dist > effective_limit:
            return []
            
        # === Debug Print 3 ===
        print(f"✅ Owner Found! ID: {owner.id}, Team: {owner.team}")

        # 2. Calculate passing routes
        # Find all teammates
        teammates = [p for p in frame_data.players if p.team == owner.team and p.id != owner.id]
        
        lines_to_draw = []
        
        for mate in teammates:
            # Calculate distance
            dist = np.linalg.norm(np.array(owner.anchor) - np.array(mate.anchor))
            
            # Only draw lines within range
            if dist < self.max_pass_dist:
                # Closer distance = brighter line (higher opacity)
                opacity = 1.0 - (dist / self.max_pass_dist)
                # Set minimum opacity so it's not too faint
                opacity = max(0.2, opacity)
                
                lines_to_draw.append((owner.anchor, mate.anchor, opacity))
                
        return lines_to_draw
