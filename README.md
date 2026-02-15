# Tactix — Automated Football Tactical Analysis Engine

<p align="center">
  <strong>Turn broadcast footage into tactical intelligence.</strong><br>
  Computer vision pipeline that detects players, tracks movements, classifies teams,<br>
  and generates 2D tactical minimaps with rich analytical overlays.
</p>

---

## Features

### Core Vision Pipeline

- **Player & Ball Detection** — YOLOv8 / YOLO26x detects players, goalkeepers, referees, and the ball in every frame
- **Multi-Object Tracking** — ByteTrack (via Supervision) assigns persistent IDs across frames with occlusion handling
- **Team Classification** — K-Means clustering on jersey colors with cross-frame voting (`PlayerRegistry`); confirms team after 5 frames with ≥70% majority
- **Jersey Number OCR** — EasyOCR-based detection (0–99) with multi-ROI crops, CLAHE/binary preprocessing, and two-digit recovery heuristic; runs every Nth frame for performance
- **Ball Interpolation** — Forward-projects ball position during detection gaps (up to 10 frames) via linear extrapolation

### Pitch Calibration

| Mode | Description |
|------|-------------|
| **AI Only** | Fully automatic — YOLO-Pose detects 27 pitch keypoints |
| **Manual Fixed** | Interactive point selection + optical flow fallback |
| **Panorama** | Manual initialization + global camera motion tracking (Pan/Tilt/Zoom) |

Computes a 3×3 homography matrix to map pixel coordinates to real-world pitch coordinates (105×68m).

### Tactical Analysis

| Module | Description |
|--------|-------------|
| **Voronoi / Space Control** | Which team dominates which area of the pitch |
| **Heatmap** | Cumulative movement density per team |
| **Pass Network** | Passing lane frequency graph between teammates |
| **Pressure Index** | Defensive pressure (opponent count within configurable radius) |
| **Cover Shadow** | Blocked passing lanes visualization |
| **Team Compactness** | Convex hull of team shape |
| **Team Centroid** | Center of mass for each team |
| **Team Width & Length** | Spatial dimensions of team shape |
| **Velocity Vectors** | Player movement direction and speed |

### Phase Analysis

| Phase | Modules |
|-------|---------|
| **Attacking** | Shot Map, Pass Sonar (8-sector directional), Zone 14 Analysis, Build-up Sequence Tracker |
| **Defense** | Duel Heatmap (spatial distribution of 1v1 contests) |
| **Transition** | Attack transitions (recovery → shot, ≤15s) and defensive transitions (loss → recovery, ≤30s) |
| **Set Pieces** | Corner Kick Analyzer, Free Kick Analyzer (with wall detection and xG estimation) |

### Formation Detection

Detects team formations (4-3-3, 4-2-3-1, 3-5-2, etc.) via K-Means clustering on player positions; matched against 17+ canonical templates with sliding-window temporal smoothing.

### Event Detection

Automatically detects match events in real-time:

- **Possession Changes** — Ball ownership transfer with confirmation delay
- **Passes** — Intra-team ball transfers with origin, destination, distance, angle
- **Shots** — High ball velocity toward goal with outcome classification (goal/saved/blocked/off-target)
- **Duels** — 1v1 contests between opposing players with winner determination
- **Corners** — Corner kicks with side, taker, and outcome tracking
- **Free Kicks** — Defensive wall detection, xG estimation, outcome tracking

### Data Export

| Format | Output | Description |
|--------|--------|-------------|
| **FIFA EPTS STF** | `match_metadata.xml` + `match_tracking.dat` | Tracab-compatible tracking data, loadable via [kloppy](https://github.com/PySport/kloppy). Center-origin coordinates in centimeters |
| **PDF Report** | `match_report.pdf` | Multi-page post-match tactical report with stats and visualizations |
| **Cache** | Pickle file | Optional frame-level tracking cache for re-processing |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.12+ |
| Object Detection | Ultralytics YOLOv8 / YOLO26x |
| Tracking | ByteTrack via Supervision |
| Computer Vision | OpenCV (homography, optical flow, Voronoi, heatmaps) |
| Team Classification | Scikit-learn K-Means |
| Jersey OCR | EasyOCR (optional) |
| PDF Reports | ReportLab + Pillow |
| GPU | Apple Metal (`mps`), CUDA, or CPU |

---

## Getting Started

### Installation

```bash
# Python 3.12+ required
pip install -r requirements.txt

# Optional: Jersey number detection
pip install easyocr>=1.7.1
```

### Configuration

All settings are in `tactix/config.py`. Key options:

```python
# Input / Output
INPUT_VIDEO  = "assets/samples/test2.mp4"
OUTPUT_VIDEO = "assets/output/test2_Result.mp4"

# Device (GPU acceleration)
DEVICE = "mps"  # "cuda" or "cpu"

# Calibration
CALIBRATION_MODE = CalibrationMode.PANORAMA  # AI_ONLY | MANUAL_FIXED | PANORAMA
INTERACTIVE_MODE = False                      # True → manual point calibration UI

# Export toggles
EXPORT_STF  = False   # FIFA EPTS Standard Transfer Format
EXPORT_PDF  = False   # PDF tactical report

# FIFA STF metadata (used when EXPORT_STF = True)
STF_MATCH_ID        = "TACTIX-001"
STF_HOME_TEAM_NAME  = "Home"
STF_AWAY_TEAM_NAME  = "Away"
```

### Running

```bash
python run.py
```

**Interactive mode** (`INTERACTIVE_MODE = True`):
1. First frame appears — click on visible pitch landmarks (corner flags, penalty box corners, etc.)
2. Enter the corresponding keypoint ID shown in the console
3. Repeat for at least 4 points
4. Press `q` or `Esc` to start analysis

**Visualization menu** appears before processing:
- Toggle overlays (Voronoi, Heatmap, Pass Network, etc.) by entering their number
- Press `r` to run, `q` to quit

---

## Output

| File | Description |
|------|-------------|
| `assets/output/*_Result.mp4` | Annotated video with tactical overlays and minimap |

| `assets/output/stf/match_metadata.xml` | Tracab-compatible metadata (teams, rosters, jersey numbers) |
| `assets/output/stf/match_tracking.dat` | Tracab DAT raw tracking data (per-frame positions & speeds) |
| `assets/output/match_report.pdf` | Multi-page tactical analysis report |

#### Loading tracking data with kloppy

```python
from kloppy import tracab

dataset = tracab.load(
    meta_data="assets/output/stf/match_metadata.xml",
    raw_data="assets/output/stf/match_tracking.dat",
)
df = dataset.to_df()
print(df.head())
```

---

## Architecture

### Pipeline Stages

```
Frame → Calibration → Detection → Ball Interpolation → Tracking → 
Team Classification → Jersey OCR → Coordinate Mapping → 
Event Detection → Tactical Analysis → Visualization → Export
```

### Data Flow

All data flows through `FrameData` — one instance per frame, acting as the central data bus between pipeline stages. Side-channel state (`PlayerRegistry`, `BallStateTracker`, `EventDetector`) lives on the engine, not on the per-frame bus.

### Project Structure

```
Tactix/
├── run.py                          # Entry point
├── requirements.txt
├── tactix/
│   ├── config.py                   # Central configuration
│   ├── core/                       # Data types, keypoints, geometry, events, registry
│   ├── engine/                     # TactixEngine — main pipeline loop
│   ├── models/                     # Model interface abstraction
│   ├── vision/
│   │   ├── detector.py             # YOLO detection wrapper
│   │   ├── tracker.py              # ByteTrack tracking wrapper
│   │   ├── transformer.py          # Homography (pixel ↔ meters)
│   │   ├── camera.py               # Optical flow camera stabilization
│   │   └── calibration/            # AI, Manual, Panorama estimators
│   ├── semantics/
│   │   ├── team.py                 # K-Means team classification
│   │   └── jersey_ocr.py           # Jersey number detection
│   ├── analytics/
│   │   ├── base/                   # Heatmap, pass network, pressure index
│   │   ├── attacking/              # Shot map, pass sonar, zone 14, build-up
│   │   ├── defense/                # Duel heatmap
│   │   ├── transition/             # Attack/defense transition tracker
│   │   ├── set_pieces/             # Corner & free kick analyzers
│   │   ├── formation/              # Formation detector
│   │   └── events/                 # Event detector (passes, shots, duels, etc.)
│   ├── visualization/
│   │   ├── minimap.py              # 2D tactical minimap renderer
│   │   └── overlays/               # RGBA overlay renderers for each analysis module
│   ├── export/
│   │   ├── stf_exporter.py         # FIFA EPTS Standard Transfer Format
│   │   ├── pdf_exporter.py         # PDF tactical report
│   │   └── cache.py                # Pickle tracking cache
│   ├── ui/                         # Calibration UI, visualization menu
│   └── utils/                      # Video I/O, pitch generation
├── assets/
│   ├── weights/                    # YOLO model weights
│   ├── samples/                    # Input videos
│   └── output/                     # Generated output
├── datasets/                       # Training datasets
├── training/                       # Model training scripts
└── notebooks/                      # Experimentation
```

---

## License

See [LICENSE](LICENSE) for details.
