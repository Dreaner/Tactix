"""
Project: Tactix
File Created: 2026-02-04 16:14:41
Author: Xingnan Zhu
File Name: run.py
Description:
    The entry point for the Tactix application.
    It handles initialization, optional interactive calibration, and starts the main engine loop.
"""


import sys
import os

# Ensure tactix package can be found
sys.path.append(os.getcwd())

from tactix.engine.system import TactixEngine
from tactix.config import Config
from tactix.ui.calibration import CalibrationUI

if __name__ == "__main__":
    # 1. Check if Interactive Calibration is needed
    # We create a temporary Config instance to check the flag
    cfg = Config()
    
    manual_points = None
    if cfg.INTERACTIVE_MODE:
        print("🔧 Launching Interactive Calibration Tool...")
        calib_ui = CalibrationUI(cfg.INPUT_VIDEO)
        manual_points = calib_ui.run()
        
        if manual_points:
            print(f"✅ Calibration successful! Captured {len(manual_points)} points.")
        else:
            print("⚠️ Calibration cancelled or failed. Falling back to default configuration.")

    # 2. Start the Engine
    # Pass the manual points (if any) to the engine
    engine = TactixEngine(manual_keypoints=manual_points)
    engine.run()