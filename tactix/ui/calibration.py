"""
Project: Tactix
File Created: 2026-02-05 14:54:47
Author: Xingnan Zhu
File Name: calibration.py
Description:
    Provides an interactive UI for manual pitch calibration.
    Allows the user to click on keypoints in the video frame and assign them
    to standard pitch landmarks, generating a calibration configuration.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from tactix.core.keypoints import YOLO_INDEX_MAP

class CalibrationUI:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.points = [] # List of (x, y, id)
        self.current_frame = None
        self.window_name = "Tactix Calibration Tool"
        
        # Reverse mapping: Name -> ID
        self.name_to_id = {v: k for k, v in YOLO_INDEX_MAP.items()}

    def run(self) -> List[Tuple[int, int, int]]:
        """
        Runs the interactive calibration loop.
        Returns: List of (x, y, keypoint_id)
        """
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print(f"❌ Error: Could not read video {self.video_path}")
            return []

        self.current_frame = frame
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        print("\n" + "="*50)
        print("🎯 INTERACTIVE CALIBRATION MODE")
        print("="*50)
        print("1. Click on a keypoint in the image.")
        print("2. Select the corresponding landmark from the console list.")
        print("3. Repeat for at least 4 points.")
        print("4. Press 'q' or 'Esc' to finish.")
        print("-" * 50)

        while True:
            # Draw points
            display_img = self.current_frame.copy()
            for x, y, pid in self.points:
                # Draw crosshair
                cv2.drawMarker(display_img, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
                # Draw label
                label = YOLO_INDEX_MAP.get(pid, str(pid))
                cv2.putText(display_img, label, (x + 10, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow(self.window_name, display_img)
            
            key = cv2.waitKey(20) & 0xFF
            if key == 27 or key == ord('q'): # Esc or q
                break

        cv2.destroyAllWindows()
        
        if len(self.points) < 4:
            print("⚠️ Warning: Less than 4 points selected. Calibration might fail.")
            
        return self.points

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # 1. Draw a temporary marker to show where user clicked
            temp_img = self.current_frame.copy()
            cv2.circle(temp_img, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow(self.window_name, temp_img)
            cv2.waitKey(1) # Force UI update

            # 2. Ask user for input in console
            print(f"\n📍 Point clicked at ({x}, {y})")
            print("Select landmark ID:")
            
            # Print available options in columns
            sorted_items = sorted(YOLO_INDEX_MAP.items())
            for i in range(0, len(sorted_items), 2):
                item1 = sorted_items[i]
                str1 = f"{item1[0]}: {item1[1]}"
                if i + 1 < len(sorted_items):
                    item2 = sorted_items[i+1]
                    str2 = f"{item2[0]}: {item2[1]}"
                    print(f"{str1:<35} | {str2}")
                else:
                    print(f"{str1}")
            
            try:
                user_input = input("Enter ID (or 'c' to cancel): ").strip()
                if user_input.lower() == 'c':
                    print("Cancelled.")
                    return

                pid = int(user_input)
                if pid in YOLO_INDEX_MAP:
                    self.points.append((x, y, pid))
                    print(f"✅ Added: {YOLO_INDEX_MAP[pid]}")
                else:
                    print("❌ Invalid ID.")
            except ValueError:
                print("❌ Invalid input.")
