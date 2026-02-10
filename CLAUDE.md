# Tactix - CLAUDE.md

## Project Overview

Tactix is an automated football (soccer) tactical analysis engine that processes broadcast videos to extract tactical insights. It uses computer vision to detect and track players, classify teams by jersey color, calibrate the pitch perspective, and render 2D tactical minimaps with various overlay layers.

## Tech Stack

- **Language:** Python 3.12+
- **Object Detection:** Ultralytics YOLOv8 (player/ball detection + pitch keypoint detection)
- **Tracking:** ByteTrack via Supervision
- **Computer Vision:** OpenCV (homography, optical flow, Voronoi, heatmaps)
- **Team Classification:** Scikit-learn K-Means clustering
- **GPU:** Apple Metal (`mps`) by default, falls back to `cuda` or `cpu`

## Running the Project

```bash
pip install -r requirements.txt
python run.py
```

All configuration is in `tactix/config.py`. Edit it to change input/output paths, calibration mode, and visualization toggles before running.

## Architecture

### Pipeline Stages (in `tactix/engine/system.py`)

1. **Pitch Calibration** - Detects 27 pitch keypoints via YOLO-Pose or manual points; computes homography matrix
2. **Player Detection** - YOLO detects players, ball, referees, goalkeepers
3. **Tracking** - ByteTrack assigns persistent IDs across frames
4. **Team Classification** - K-Means on jersey colors; heuristic for goalkeeper detection
5. **Coordinate Mapping** - Pixel → real-world meters via homography; velocity computation
6. **Tactical Analysis** - Voronoi, heatmap, pressure index, pass network, compactness, etc.
7. **Visualization** - Renders 2D minimap with RGBA overlays composited onto video
8. **Export** - Saves JSON tracking data and MP4 output

### Key Data Flow

All data flows through `FrameData` (defined in `tactix/core/types.py`). Each pipeline stage reads from and writes to a single `FrameData` instance per frame — this is the central data bus.

### Calibration Modes (set in `config.py`)

| Mode | Description |
|------|-------------|
| `AI_ONLY` | Fully automatic YOLO-Pose keypoint detection |
| `MANUAL_FIXED` | User-specified fixed points + optical flow fallback |
| `PANORAMA` | Manual init + global motion tracking across frames |

## Key Files

| File | Purpose |
|------|---------|
| `run.py` | Entry point; orchestrates init, UI menus, engine startup |
| `tactix/config.py` | **Central config** — all paths, thresholds, toggles |
| `tactix/engine/system.py` | `TactixEngine` — main pipeline loop |
| `tactix/core/types.py` | Data contracts: `Player`, `Ball`, `FrameData`, `TeamID` |
| `tactix/core/registry.py` | **Persistent cross-frame state**: `PlayerRegistry` (team voting), `BallStateTracker` (interpolation) |
| `tactix/core/keypoints.py` | 27 standard pitch keypoint definitions |
| `tactix/core/geometry.py` | World coordinate system (105×68m pitch) |
| `tactix/vision/transformer.py` | Homography matrix management (pixel ↔ meters) |
| `tactix/vision/detector.py` | YOLO-based detection wrapper |
| `tactix/vision/tracker.py` | ByteTrack ID persistence wrapper |
| `tactix/vision/camera.py` | Optical flow camera stabilization |
| `tactix/semantics/team.py` | K-Means team classification |
| `tactix/visualization/minimap.py` | 2D tactical minimap renderer |
| `tactix/tactics/` | Tactical modules (pass network, Voronoi, heatmap, etc.) |
| `tactix/ui/calibration.py` | Interactive manual calibration UI |
| `tactix/export/json_exporter.py` | JSON tracking data export |
| `tactix/export/cache.py` | Pickle-based tracking cache (`TrackingCache`); enabled via `ENABLE_CACHE` |

## Conventions & Coding Rules

- **Type hints are mandatory** for all function signatures
- **Never hardcode** paths, thresholds, or parameters — always use `Config`
- All tactical parameters (distances, radii, angles) live in `tactix/config.py`
- Coordinate system: origin at TOP-LEFT, X rightward (0–105m), Y downward (0–68m)
- Team identities use `TeamID` enum: `A`, `B`, `REFEREE`, `GOALKEEPER`, `UNKNOWN`
- New tactical modules go in `tactix/tactics/` and should follow the single-responsibility pattern of existing modules
- New calibration methods go in `tactix/vision/calibration/` and must subclass `BaseEstimator`

## Configuration Reference

```python
# Key paths
INPUT_VIDEO  = "assets/samples/test2.mp4"
OUTPUT_VIDEO = "assets/output/test2_Result.mp4"
PITCH_MODEL_PATH  = "assets/weights/football_pitch.pt"
PLAYER_MODEL_PATH = "assets/weights/ball_player_yolo26x.pt"

# Device
DEVICE = "mps"  # "cuda" or "cpu"

# Visualization toggles (all default False)
SHOW_MINIMAP, SHOW_VORONOI, SHOW_HEATMAP, SHOW_COMPACTNESS,
SHOW_PASS_NETWORK, SHOW_VELOCITY, SHOW_PRESSURE,
SHOW_COVER_SHADOW, SHOW_TEAM_CENTROID, SHOW_TEAM_WIDTH_LENGTH

# Tracking & interpolation (added 2026-02-10)
BALL_INTERP_MAX_GAP = 10   # Frames to extrapolate when ball is missing

# Cache (added 2026-02-10, disabled by default)
ENABLE_CACHE = False
CACHE_DIR    = "assets/cache"
```

## Project Layout

```
Tactix/
├── run.py                   # Entry point
├── requirements.txt
├── tactix/
│   ├── config.py            # Central configuration
│   ├── core/                # Data types, keypoints, geometry
│   ├── engine/              # TactixEngine pipeline
│   ├── vision/              # Detection, tracking, calibration, camera
│   ├── semantics/           # Team classification
│   ├── tactics/             # Tactical analysis modules
│   ├── visualization/       # Minimap renderer
│   ├── ui/                  # Interactive UIs
│   └── export/              # Data exporters
├── assets/
│   ├── weights/             # YOLO model weights
│   ├── samples/             # Input test videos
│   ├── output/              # Generated video/JSON output
│   └── pitch_bg.png         # Pitch template image (1559×1010)
├── datasets/                # Training/test datasets
├── notebooks/               # Jupyter experimentation
├── training/                # Model training scripts
├── NOTE.md                  # Technical notes and future directions
├── TODO.md                  # Feature roadmap
└── AGENTS.md                # Development context
```

## No Formal Testing

There is no test suite currently. Experimentation is done via `notebooks/test.ipynb`. When writing new modules, validate against sample videos manually.

## Notes for Development

- Smoothing is a known priority (simple moving average or OneEuroFilter for homography stability)
- Optical flow acts as fallback when fewer than 4 pitch keypoints are detected by YOLO
- The minimap renders at a fixed canvas size; all overlay modules return RGBA numpy arrays that are composited together
- The `FrameData` bus is the contract between modules — extend it in `core/types.py` when new data needs to flow between stages

## Recent Changes (2026-02-10) — Tracking Stability Improvements

### What was changed and why

The original pipeline had three stability problems:
1. **Team assignment flickered**: `_stage_classification` re-ran K-Means every frame on every `UNKNOWN` player. If ByteTrack switched a player's ID (occlusion, fast movement), their team reset to grey.
2. **Velocity spikes on re-detection**: `tracker.update_velocity()` history was keyed by `tracker_id`. If an ID was lost and resumed after a gap >1 frame, the position jump between old and new history entries produced a false velocity spike.
3. **Ball detection gaps**: When the ball briefly left frame or was occluded, `frame_data.ball` went `None` and downstream modules lost all ball context.

### Changes made

| File | Change |
|------|--------|
| `tactix/core/registry.py` *(new)* | `PlayerRegistry`: accumulates per-ID team votes across frames; confirms after 5 frames with ≥70% majority. `BallStateTracker`: forward-projects ball position during detection gaps (up to 10 frames). |
| `tactix/engine/system.py` | Imports and instantiates `PlayerRegistry` + `BallStateTracker`. `_stage_classification` rewritten: confirmed players skip K-Means entirely; unconfirmed players vote. Ball interpolation applied immediately after `detector.detect()`. |
| `tactix/semantics/team.py` | Added `predict_one(color)` — single-color prediction used by the new voting loop. |
| `tactix/vision/tracker.py` | `update_velocity()`: added frame-gap guard — if history spans >10 frames, clear and restart to prevent false velocity spikes. |
| `tactix/export/cache.py` *(new)* | `TrackingCache`: pickle cache keyed by video filename + frame count. Disabled by default (`ENABLE_CACHE=False`). |
| `tactix/config.py` | Added `BALL_INTERP_MAX_GAP`, `ENABLE_CACHE`, `CACHE_DIR`. |

### Key design decision
`FrameData` (the per-frame bus) was **not** modified. The registry is a side-channel living on `TactixEngine`, consulted and updated each frame without touching the data contract.
