# Agent Context for Project Tactix

## 1. Project Overview
Tactix is an automated football tactical analysis engine using Computer Vision.
It processes broadcast videos to generate 2D tactical minimaps, passing networks, and player tracking data.

## 2. Tech Stack (Strictly Follow)
- **Language:** Python 3.12+
- **Core Vision:** OpenCV (cv2), Ultralytics (YOLOv8), Supervision (roboflow/supervision)
- **Data:** NumPy (heavy usage for matrix ops)
- **UI/Video:** OpenCV for drawing, sv.VideoSink for saving.

## 3. Coding Guidelines
- **Type Hints:** MANDATORY for all function arguments and return values.
  - Bad: `def process(frame):`
  - Good: `def process(frame: np.ndarray) -> FrameData:`
- **Config:** NEVER hardcode thresholds or paths. Always load from `tactix.config.Config`.
- **Comments:** Use concise English or Chinese for complex logic.
- **Error Handling:** Do not crash on a single bad frame. Use `try-except` blocks in the main loop.

## 4. Key Architecture Rules
- `tactix/vision/transformer.py`: Handles Homography (Pixel <-> Meter conversion).
- `tactix/engine/system.py`: The main pipeline loop. Avoid adding heavy logic here; delegate to sub-modules.
- `tactix/core/types.py`: All data classes (Player, Ball, FrameData) are defined here. Do not use raw dictionaries.

## 5. Specific Constraints
- **Coordinate System:** Origin (0,0) is TOP-LEFT corner.
- **Minimap:** When drawing the minimap, do not create a new canvas every frame if possible; optimize for performance.
- **Mocking:** If `Config.USE_MOCK_PITCH` is True, bypass the YOLO model and use fixed keypoints.
- 