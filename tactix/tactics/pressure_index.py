"""
Project: Tactix
File Created: 2026-02-07 00:00:48
Author: Xingnan Zhu
File Name: pressure_index.py
Description:
    Calculates the Pressure Index for each player.
    Pressure is determined by the proximity and number of opposing players.
"""

import numpy as np
from tactix.core.types import FrameData, TeamID, Player

class PressureIndex:
    def __init__(self, pressure_radius: float = 8.0):
        """
        :param pressure_radius: Radius in meters to consider an opponent as applying pressure.
        """
        self.pressure_radius = pressure_radius

    def calculate(self, frame_data: FrameData):
        """
        Calculates pressure for each player and updates the player.pressure attribute.
        """
        players = frame_data.players
        
        for p in players:
            # Only calculate for players with valid position and team
            if p.pitch_position is None or p.team in [TeamID.UNKNOWN, TeamID.REFEREE]:
                p.pressure = 0.0
                continue
                
            pressure_score = 0.0
            
            # Find opponents
            opponents = [op for op in players if op.team != p.team and op.team in [TeamID.A, TeamID.B] and op.pitch_position]
            
            for op in opponents:
                # Calculate distance in meters
                dist = np.sqrt((p.pitch_position.x - op.pitch_position.x)**2 + 
                               (p.pitch_position.y - op.pitch_position.y)**2)
                
                if dist < self.pressure_radius:
                    # Simple inverse distance model: closer = more pressure
                    # Avoid division by zero
                    weight = 1.0 / (dist + 0.5) 
                    pressure_score += weight
            
            # Normalize pressure (heuristic)
            # Assume max pressure is around 3.0 (e.g., 3 players within 1m)
            normalized_pressure = min(pressure_score / 3.0, 1.0)
            p.pressure = normalized_pressure
