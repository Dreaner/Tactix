# ⚽ Tactix — Automated Football Tactical Analysis Engine

[![PyPI version](https://badge.fury.io/py/tactix.svg)](https://pypi.org/project/tactix/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

<p align="center">
  <strong>Turn broadcast footage into tactical intelligence.</strong><br>
  Computer vision pipeline that detects players, tracks movements, classifies teams,<br>
  and generates 2D tactical minimaps with rich analytical overlays.
</p>

---

## 📦 Installation

``` bash
pip install tactix
```

With jersey number OCR support (optional):

``` bash
pip install tactix[ocr]
```

### Requirements

- **Python 3.12+**
- **GPU** — Apple Silicon (`mps`), NVIDIA (`cuda`), or `cpu` fallback
- **Model weights** — see [Model Weights](#-model-weights) section below

---

## 🏋️ Model Weights

Tactix requires two YOLO model files. Download them from the [GitHub Releases](https://github.com/Dreaner/Tactix/releases) page:

| Model | Purpose |
|-------|---------|
| `ball_player_yolo26x.pt` | Player / Ball / Goalkeeper / Referee detection |
| `football_pitch.pt` | Pitch keypoint detection (27 points for homography) |

Place them anywhere you like — you'll point to them in the config.

---

## 🚀 Quick Start

### Option 1: Python Script (Recommended)

``` python
from tactix import TactixEngine, Config

cfg = Config()

# Set your video paths
cfg.INPUT_VIDEO  = "path/to/match.mp4"
cfg.OUTPUT_VIDEO = "path/to/output/result.mp4"

# Set model weight paths
cfg.PLAYER_MODEL_PATH = "path/to/ball_player_yolo26x.pt"
cfg.PITCH_MODEL_PATH  = "path/to/football_pitch.pt"

# Choose GPU device
cfg.DEVICE = "mps"   # "cuda" for NVIDIA, "cpu" for fallback

# Turn off interactive mode for scripted usage
cfg.INTERACTIVE_MODE = False

# Enable the overlays you want
cfg.SHOW_MINIMAP       = True
cfg.SHOW_VORONOI       = True
cfg.SHOW_HEATMAP       = True
cfg.SHOW_PASS_NETWORK  = True

# Run
engine = TactixEngine(cfg)
engine.run()
# → result.mp4 saved with tactical overlays
```

### Option 2: Command Line

``` bash
tactix
```

Launches interactive mode — calibration UI → visualization menu → processing.

### Option 3: Minimal Example

``` python
from tactix import TactixEngine, Config

cfg = Config()
cfg.INPUT_VIDEO  = "match.mp4"
cfg.OUTPUT_VIDEO = "result.mp4"
cfg.PLAYER_MODEL_PATH = "ball_player_yolo26x.pt"
cfg.PITCH_MODEL_PATH  = "football_pitch.pt"
cfg.INTERACTIVE_MODE = False
cfg.SHOW_MINIMAP = True

engine = TactixEngine(cfg)
engine.run()
```

---

## 📊 What It Produces

### 1. Annotated Video (`OUTPUT_VIDEO`)

The original broadcast with all enabled overlays composited — player ellipses with team colors, ball marker, and the 2D minimap in the corner.

### 2. FIFA EPTS Tracking Data (optional)

``` python
cfg.EXPORT_STF = True
cfg.OUTPUT_STF_DIR = "output/stf"
cfg.STF_HOME_TEAM_NAME = "Barcelona"
cfg.STF_AWAY_TEAM_NAME = "Real Madrid"
```

Produces Tracab-compatible files loadable with [kloppy](https://github.com/PySport/kloppy):

``` python
from kloppy import tracab

dataset = tracab.load(
    meta_data="output/stf/match_metadata.xml",
    raw_data="output/stf/match_tracking.dat",
)
df = dataset.to_df()
```

### 3. PDF Tactical Report (optional)

``` python
cfg.EXPORT_PDF = True
cfg.OUTPUT_PDF = "output/match_report.pdf"
```

Multi-page report with shot maps, pass sonars, formation analysis, and match statistics.

---

## ⚙️ Full Configuration Reference

### Paths

| Parameter | Default | Description |
|-----------|---------|-------------|
| `INPUT_VIDEO` | `"assets/samples/test2.mp4"` | Input video file |
| `OUTPUT_VIDEO` | `"assets/output/test2_Result.mp4"` | Output annotated video |
| `PLAYER_MODEL_PATH` | `"assets/weights/ball_player_yolo26x.pt"` | YOLO player/ball model |
| `PITCH_MODEL_PATH` | `"assets/weights/football_pitch.pt"` | YOLO pitch keypoint model |

### Device & Detection

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DEVICE` | `"mps"` | `"mps"` / `"cuda"` / `"cpu"` |
| `CONF_PLAYER` | `0.3` | Detection confidence threshold |

### Calibration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CALIBRATION_MODE` | `CalibrationMode.PANORAMA` | `AI_ONLY` / `MANUAL_FIXED` / `PANORAMA` |
| `INTERACTIVE_MODE` | `True` | Launch manual calibration UI on startup |

``` python
from tactix.config import CalibrationMode

cfg.CALIBRATION_MODE = CalibrationMode.AI_ONLY      # Fully automatic
cfg.CALIBRATION_MODE = CalibrationMode.MANUAL_FIXED  # Manual + optical flow
cfg.CALIBRATION_MODE = CalibrationMode.PANORAMA      # Manual + global tracking
```

### Visualization Toggles (all default `False`)

| Toggle | Description |
|--------|-------------|
| `SHOW_MINIMAP` | 2D tactical minimap |
| `SHOW_VORONOI` | Voronoi space control zones |
| `SHOW_HEATMAP` | Cumulative player heatmap |
| `SHOW_PASS_NETWORK` | Passing connections between teammates |
| `SHOW_VELOCITY` | Speed & direction arrows |
| `SHOW_PRESSURE` | Defensive pressing intensity |
| `SHOW_COMPACTNESS` | Team shape (convex hull) |
| `SHOW_COVER_SHADOW` | Blocked passing lanes |
| `SHOW_TEAM_CENTROID` | Team center of mass |
| `SHOW_TEAM_WIDTH_LENGTH` | Team width/length metrics |
| `SHOW_SHOT_MAP` | Shot locations and outcomes |
| `SHOW_PASS_SONAR` | 8-sector directional pass radar |
| `SHOW_ZONE_14` | Zone 14 activity heatmap |
| `SHOW_BUILDUP` | Build-up sequence visualization |
| `SHOW_TRANSITION` | Attack/defense transition tracking |
| `SHOW_DUEL_HEATMAP` | 1v1 duel spatial distribution |
| `SHOW_SET_PIECES` | Corner kick & free kick analysis |
| `SHOW_FORMATION` | Formation detection (4-3-3, 4-2-3-1, etc.) |

### Team Color Pre-scan

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ENABLE_COLOR_PRESCAN` | `True` | Sample frames across the video to learn jersey colors before processing |
| `PRESCAN_NUM_FRAMES` | `30` | Frames to sample |
| `PRESCAN_MIN_PLAYERS` | `4` | Min players per frame to include |

### Export

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EXPORT_STF` | `True` | Export FIFA EPTS tracking data |
| `EXPORT_PDF` | `True` | Export PDF tactical report |
| `ENABLE_CACHE` | `False` | Cache tracking data for re-processing |

---

## 🔧 Advanced Usage

### Using Individual Components

``` python
import cv2
from tactix.vision.detector import Detector

detector = Detector("ball_player_yolo26x.pt", device="mps")

frame = cv2.imread("frame.jpg")
frame_data = detector.detect(frame, frame_index=0)

for player in frame_data.players:
    print(f"Player at {player.rect}, conf={player.confidence:.2f}")

if frame_data.ball:
    print(f"Ball at {frame_data.ball.center}")
```

### Manual Calibration

For difficult camera angles:

``` python
cfg.INTERACTIVE_MODE = True
cfg.CALIBRATION_MODE = CalibrationMode.MANUAL_FIXED
```

An OpenCV window opens — click pitch landmarks on the video frame, enter the keypoint ID, repeat for ≥4 points, then press `q` to start.

---

## 🧩 Features

### Core Vision Pipeline

- **Player & Ball Detection** — YOLO26x detects players, goalkeepers, referees, and ball
- **Multi-Object Tracking** — ByteTrack assigns persistent IDs across frames
- **Team Classification** — K-Means on jersey colors with cross-frame voting; confirms after 5 frames with ≥70% majority
- **Jersey Number OCR** — EasyOCR-based detection with multi-ROI crops and two-digit recovery
- **Ball Interpolation** — Linear extrapolation during detection gaps (up to 10 frames)

### Tactical Analysis

| Module | Description |
|--------|-------------|
| Voronoi | Space control zones per team |
| Heatmap | Cumulative movement density |
| Pass Network | Passing frequency graph |
| Pressure Index | Opponent count within radius |
| Cover Shadow | Blocked passing lanes |
| Team Compactness | Convex hull of team shape |
| Shot Map | Shot locations with outcomes |
| Pass Sonar | 8-sector directional pass radar |
| Zone 14 | Activity in the key attacking zone |
| Build-up Tracker | Possession sequence tracking |
| Transition | Attack/defense transition analysis |
| Duel Heatmap | 1v1 contest spatial distribution |
| Set Pieces | Corner & free kick analysis with xG |
| Formation | Auto-detection (4-3-3, 4-2-3-1, 3-5-2, etc.) |

### Event Detection

Automatically detects: possession changes, passes, shots, duels, corners, free kicks.

---

## 🏗️ Architecture

### Pipeline

```
Frame → Pitch Calibration → Detection → Ball Interpolation → Tracking
  → Team Classification → Jersey OCR → Coordinate Mapping
  → Event Detection → Tactical Analysis → Visualization → Export
```

### Project Structure

```
tactix/
├── config.py                   # All paths, thresholds, toggles
├── cli.py                      # CLI entry point
├── core/                       # Types (Player, Ball, FrameData), registry, geometry
├── engine/system.py            # TactixEngine — main pipeline loop
├── vision/                     # Detection, tracking, calibration, homography
├── semantics/                  # Team classification, jersey OCR
├── analytics/                  # All tactical analysis modules
├── visualization/              # Minimap renderer + overlay layers
├── export/                     # STF, PDF, cache exporters
└── ui/                         # Calibration & visualization menu
```

All data flows through `FrameData` (one instance per frame) — the central data bus between pipeline stages.

---

## 🤝 Contributing

Contributions welcome! Key conventions:

- **Type hints mandatory** on all function signatures
- **Never hardcode** — use `Config` for all thresholds and paths
- New tactical modules → `tactix/analytics/`
- New overlays → `tactix/visualization/overlays/`
- Extend `FrameData` in `core/types.py` when new data needs to flow between stages

---

## 📄 License

GPL-3.0 — see [LICENSE](LICENSE).
