# Tactix: Automated Football Tactical Analysis Engine

Tactix is a computer vision-based system designed to analyze football broadcast videos. It automatically detects players, tracks their movements, classifies teams, and generates tactical visualizations such as 2D minimaps, passing networks, and heatmaps.

## 🚀 Features

*   **Player Detection & Tracking**: Uses YOLOv8 and ByteTrack to detect and track players, referees, and the ball.
*   **Team Classification**: Automatically groups players into two teams based on jersey color using K-Means clustering.
*   **Pitch Calibration**:
    *   **AI Mode**: Automatic keypoint detection using a custom YOLO-Pose model.
    *   **Manual Mode**: Interactive tool to manually select keypoints for high-precision calibration.
    *   **Panorama Mode**: Tracks camera movement (Pan/Tilt/Zoom) to maintain calibration even when keypoints move out of view.
*   **Tactical Analysis**:
    *   **Minimap Generation**: Real-time 2D top-down view of the game.
    *   **Passing Network**: Visualizes potential passing lanes between teammates.
    *   **Space Control (Voronoi)**: Visualizes which team controls which areas of the pitch.
    *   **Heatmaps**: Cumulative heatmap of player movement.
    *   **Velocity Vectors**: Displays player movement direction and speed.
*   **Data Export**: Exports detailed tracking data (coordinates, teams, velocity) to JSON for external analysis.

## 🛠️ Tech Stack

*   **Language**: Python 3.12+
*   **Computer Vision**: OpenCV, Ultralytics YOLOv8, Supervision
*   **Data Processing**: NumPy, Scikit-learn, SciPy

## 🚦 Getting Started

### 1. Installation

Ensure you have Python 3.12+ installed.

```bash
pip install -r requirements.txt
```

*(Note: You may need to install `ultralytics`, `supervision`, `opencv-python`, `numpy`, `scikit-learn`, `scipy`, `tqdm`)*

### 2. Configuration

Edit `tactix/config.py` to set your video paths and preferences:

*   `INPUT_VIDEO`: Path to your input football video.
*   `OUTPUT_VIDEO`: Path to save the analyzed video.
*   `CALIBRATION_MODE`: Choose between `CalibrationMode.AI_ONLY`, `MANUAL_FIXED`, or `PANORAMA`.
*   `INTERACTIVE_MODE`: Set to `True` to enable the manual calibration tool at startup.

### 3. Running

```bash
python run.py
```

If `INTERACTIVE_MODE` is enabled:
1.  A window will appear showing the first frame of the video.
2.  Click on a distinct pitch landmark (e.g., corner flag, penalty box corner).
3.  Look at the console and enter the corresponding ID for that landmark.
4.  Repeat for at least 4 points.
5.  Press `q` or `Esc` to start the analysis.

## 📊 Output

*   **Video**: An annotated video showing the tactical overlay, minimap, and player tracking.
*   **JSON Data**: A `tracking_data.json` file containing frame-by-frame player coordinates and metadata.
