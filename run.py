"""
Project: Tactix
File Created: 2026-02-04 16:14:41
Author: Xingnan Zhu
File Name: run.py
Description:
    The entry point for the Tactix application.
    It handles initialization, optional interactive calibration, visualization settings,
    and starts the main engine loop.
"""


import sys
import os

# Ensure tactix package can be found
sys.path.append(os.getcwd())

from tactix.engine.system import TactixEngine
from tactix.config import Config
from tactix.ui.calibration import CalibrationUI
from tactix.ui.visualization_menu import VisualizationMenu

if __name__ == "__main__":
    # 1. Initialize Config
    # We create a temporary Config instance to check flags and store settings
    cfg = Config()
    
    # 2. Interactive Calibration (Optional)
    manual_points = None
    if cfg.INTERACTIVE_MODE:
        print("🔧 Launching Interactive Calibration Tool...")
        calib_ui = CalibrationUI(cfg.INPUT_VIDEO)
        manual_points = calib_ui.run()

        if manual_points:
            print(f"✅ Calibration successful! Captured {len(manual_points)} points.")
        else:
            print("⚠️ Calibration cancelled or failed. Falling back to default configuration.")

    # 3. Visualization Menu (Optional but recommended)
    # Allows user to toggle layers before starting
    viz_menu = VisualizationMenu(cfg)
    viz_menu.run()

    # 4. Start the Engine
    engine = TactixEngine(cfg=cfg, manual_keypoints=manual_points)
    engine.run()