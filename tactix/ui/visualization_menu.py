"""
Project: Tactix
File Created: 2026-02-06 15:55:37
Author: Xingnan Zhu
File Name: visualization_menu.py
Description:
    Provides a terminal-based menu for users to toggle visualization layers
    (Voronoi, Heatmap, etc.) before starting the analysis.
"""

import os
from tactix.config import Config

class VisualizationMenu:
    def __init__(self, config: Config):
        self.cfg = config
        # Map menu index to config attribute and display name
        self.options = {
            "1": ("SHOW_VORONOI", "Voronoi Diagram (Space Control)"),
            "2": ("SHOW_HEATMAP", "Heatmap (Movement History)"),
            "3": ("SHOW_COMPACTNESS", "Team Compactness (Convex Hull)"),
            "4": ("SHOW_PASS_NETWORK", "Passing Network"),
            "5": ("SHOW_VELOCITY", "Velocity Vectors"),
            "6": ("SHOW_PRESSURE", "Pressure Index"),
            "7": ("SHOW_COVER_SHADOW", "Cover Shadow (Pass Blocking)"),
            "8": ("SHOW_TEAM_CENTROID", "Team Centroid"),
            "9": ("SHOW_TEAM_WIDTH_LENGTH", "Team Width & Length"),
            "0": ("SHOW_DEBUG_KEYPOINTS", "Debug Keypoints")
        }

    def run(self):
        """
        Runs the interactive menu loop.
        """
        while True:
            self._clear_screen()
            print("\n" + "="*50)
            print("🎨 VISUALIZATION SETTINGS")
            print("="*50)
            print("Toggle layers by entering their number:")
            print("-" * 50)

            # Sort options by key to display in order
            sorted_keys = sorted(self.options.keys(), key=lambda x: int(x) if x.isdigit() else 99)
            
            for key in sorted_keys:
                attr, desc = self.options[key]
                # Get current state from config instance
                is_on = getattr(self.cfg, attr)
                status = "✅ [ON]" if is_on else "❌ [OFF]"
                print(f"[{key}] {desc:<35} {status}")

            print("-" * 50)
            print("[r] Run Analysis")
            print("[q] Quit")
            print("="*50)

            choice = input("Enter choice: ").strip().lower()

            if choice == 'r':
                print("🚀 Starting analysis...")
                break
            elif choice == 'q':
                print("👋 Exiting...")
                exit(0)
            elif choice in self.options:
                attr, _ = self.options[choice]
                # Toggle boolean value
                current_val = getattr(self.cfg, attr)
                setattr(self.cfg, attr, not current_val)
            else:
                # Invalid input, just refresh
                pass

    def _clear_screen(self):
        # Cross-platform clear screen
        os.system('cls' if os.name == 'nt' else 'clear')
