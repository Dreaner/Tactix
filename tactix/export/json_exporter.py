"""
Project: Tactix
File Created: 2026-02-05 18:37:42
Author: Xingnan Zhu
File Name: json_exporter.py
Description:
    Exports tracking datasets to a JSON file.
    The structure includes metadata and a frame-by-frame list of player/ball positions.
"""

import json
import os
from typing import List, Dict, Any
from tactix.export.base import BaseExporter
from tactix.core.types import FrameData, TeamID

class JsonExporter(BaseExporter):
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.frames_data: List[Dict[str, Any]] = []
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def add_frame(self, frame_data: FrameData):
        """
        Convert FrameData object to a dictionary and append to buffer.
        """
        frame_dict = {
            "frame_index": frame_data.frame_index,
            "players": [],
            "ball": None
        }

        # 1. Process Players
        for p in frame_data.players:
            # Only export players with valid pitch positions
            if p.pitch_position:
                player_dict = {
                    "id": p.id,
                    "team": p.team.name, # Enum to string (A, B, REFEREE...)
                    "x": round(p.pitch_position.x, 2),
                    "y": round(p.pitch_position.y, 2)
                }
                
                # Optional: Add velocity if available
                if p.velocity:
                    player_dict["vx"] = round(p.velocity.x, 2)
                    player_dict["vy"] = round(p.velocity.y, 2)
                    player_dict["speed"] = round(p.speed, 2)
                
                frame_dict["players"].append(player_dict)

        # 2. Process Ball
        if frame_data.ball and frame_data.ball.pitch_position:
            ball_dict = {
                "x": round(frame_data.ball.pitch_position.x, 2),
                "y": round(frame_data.ball.pitch_position.y, 2),
                "owner_id": frame_data.ball.owner_id
            }
            frame_dict["ball"] = ball_dict

        self.frames_data.append(frame_dict)

    def save(self):
        """
        Write the buffered datasets to a JSON file.
        """
        output_data = {
            "meta": {
                "version": "1.0",
                "pitch_size": [105, 68],
                "total_frames": len(self.frames_data)
            },
            "frames": self.frames_data
        }

        try:
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2) # indent=2 for readability, remove for smaller size
            print(f"✅ Data exported to {self.output_path}")
        except Exception as e:
            print(f"❌ Failed to export JSON: {e}")
