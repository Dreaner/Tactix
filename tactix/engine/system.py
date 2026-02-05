"""
Project: Tactix
File Created: 2026-02-04 16:07:03
Author: Xingnan Zhu
File Name: system.py
Description:
    The core engine of the Tactix system, acting as the central brain.
    It orchestrates the entire pipeline by integrating perception, logic, and
    visualization modules. Optimized to remove redundant drawing logic and
    includes matrix memory functionality for stable tracking.
"""

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm

# Import modules
from tactix.config import Config
from tactix.core.types import TeamID, Point
from tactix.semantics.team import TeamClassifier
from tactix.tactics.pass_network import PassNetwork
from tactix.tactics.space_control import SpaceControl
from tactix.tactics.heatmap import HeatmapGenerator
from tactix.vision.detector import Detector
from tactix.vision.pose import PitchEstimator, MockPitchEstimator
from tactix.vision.manuel_calibration import ManualPitchEstimator
from tactix.vision.tracker import Tracker
from tactix.vision.transformer import ViewTransformer
from tactix.visualization.minimap import MinimapRenderer
from tactix.vision.camera import CameraTracker


class TactixEngine:
    def __init__(self):
        self.cfg = Config()
        print("🚀 Initializing Tactix Engine...")

        # ==========================================
        # 1. Initialize Perception Modules
        # ==========================================
        # Decide whether to use real AI or Mock data based on config
        if self.cfg.USE_MOCK_PITCH:
            # Use manual calibration (with optical flow tracking)
            self.pitch_estimator = ManualPitchEstimator(self.cfg.MANUAL_KEYPOINTS)
        else:
            self.pitch_estimator = PitchEstimator(self.cfg.PITCH_MODEL_PATH, self.cfg.DEVICE)

        self.detector = Detector(self.cfg.PLAYER_MODEL_PATH, self.cfg.DEVICE, self.cfg.CONF_PLAYER)
        self.tracker = Tracker()
        
        # Initialize camera tracker (for smoothing jitter)
        self.camera_tracker = CameraTracker(smoothing_window=5)

        # ==========================================
        # 2. Initialize Logic Modules
        # ==========================================
        self.transformer = ViewTransformer()
        self.team_classifier = TeamClassifier(device='mps') # Use CPU for now, change to 'mps' or 'cuda' if GPU available
        self.pass_net = PassNetwork(self.cfg.MAX_PASS_DIST, self.cfg.BALL_OWNER_DIST)
        self.space_control = SpaceControl()
        self.heatmap_generator = HeatmapGenerator()

        # ==========================================
        # 3. Initialize Visualization Modules
        # ==========================================
        self.minimap_renderer = MinimapRenderer(self.cfg.PITCH_TEMPLATE)
        self._init_annotators()

        # State flags
        self.classifier_trained = False

    def _init_annotators(self):
        """Initialize Supervision annotators"""
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.4, text_padding=4)
        self.ball_annotator = sv.DotAnnotator(color=sv.Color.WHITE, radius=5)

        # Define color palette (corresponding to class_id 0-4)
        self.palette = sv.ColorPalette(colors=[
            sv.Color(230, 57, 70),   # 0: Team A (Red)
            sv.Color(69, 123, 157),  # 1: Team B (Blue)
            sv.Color(255, 255, 0),   # 2: Referee (Yellow)
            sv.Color(0, 0, 0),       # 3: Goalkeeper (Black)
            sv.Color(128, 128, 128)  # 4: Unknown (Grey)
        ])

    def run(self):
        """Main processing loop"""
        # Prepare video stream
        video_info = sv.VideoInfo.from_video_path(self.cfg.INPUT_VIDEO)
        frames = sv.get_video_frames_generator(self.cfg.INPUT_VIDEO)

        print(f"▶️ Processing: {self.cfg.INPUT_VIDEO}")
        print(f"   - Total Frames: {video_info.total_frames}")
        print(f"   - Resolution: {video_info.width}x{video_info.height}")

        with sv.VideoSink(self.cfg.OUTPUT_VIDEO, video_info) as sink:
            # Use tqdm for progress bar
            for i, frame in tqdm(enumerate(frames), total=video_info.total_frames):

                # ==========================================
                # === Stage 1: Pitch Calibration (World View) ===
                # ==========================================
                # 1. Try to detect keypoints using YOLO
                kpts_xy, kpts_conf = self.pitch_estimator.predict(frame)
                
                final_kpts = None
                
                # 2. Decision Logic: Trust YOLO or Optical Flow?
                # If YOLO detects enough points (e.g., > 3), consider it a "strong calibration"
                if kpts_xy is not None and len(kpts_xy) >= 4:
                    # Force reset the tracker to the current YOLO result
                    self.camera_tracker.reset(kpts_xy, frame)
                    final_kpts = kpts_xy
                else:
                    # If YOLO fails or detects too few points, try to "predict" using Optical Flow
                    tracked_kpts = self.camera_tracker.update(frame)
                    if tracked_kpts is not None:
                        final_kpts = tracked_kpts
                        # Fake confidence to make Transformer accept it
                        kpts_conf = np.ones(len(final_kpts)) 

                # Update matrix (returns True as long as a matrix is available, new or old)
                # Note: We pass final_kpts to the transformer
                has_matrix = False
                if final_kpts is not None:
                    has_matrix = self.transformer.update(final_kpts, kpts_conf, self.cfg.CONF_PITCH)

                # ==========================================
                # === Stage 2: Player Detection (Entities) ===
                # ==========================================
                frame_data = self.detector.detect(frame, i)

                # --- A. Tracking Module ---
                if len(frame_data.players) > 0:
                    xyxy = np.array([p.rect for p in frame_data.players])
                    class_ids = np.array([p.class_id for p in frame_data.players])
                    confidences = np.array([p.confidence for p in frame_data.players])
                    sv_dets = sv.Detections(xyxy=xyxy, class_id=class_ids, confidence=confidences)
                    self.tracker.update(sv_dets, frame_data)

                # --- B. Team Classification ---
                # Accumulate data in the first 30 frames to train the color classifier
                valid_players = [p for p in frame_data.players if p.team == TeamID.UNKNOWN]
                if not self.classifier_trained and len(valid_players) > 3 and i < 30:
                    self.team_classifier.fit(frame, frame_data.players)
                    if i > 15: self.classifier_trained = True

                # If trained, start predicting team for each player
                if self.classifier_trained:
                    self.team_classifier.predict(frame, frame_data)

                # ==========================================
                # === Stage 3: Coordinate Mapping (Projection) ===
                # ==========================================
                if has_matrix:
                    self.transformer.transform_players(frame_data.players)
                    
                    # 🔥 Calculate velocity vectors
                    # Must be called after transform_players because it needs pitch_position
                    self.tracker.update_velocity(frame_data)

                    if frame_data.ball:
                        # Transform ball separately
                        ball_pt = self.transformer.transform_point(frame_data.ball.center)
                        if ball_pt:
                             frame_data.ball.pitch_position = Point(x=ball_pt[0], y=ball_pt[1])

                # ==========================================
                # === Stage 4: Tactical Analysis (Tactics) ===
                # ==========================================
                # 4.1 Passing Network
                pass_lines = self.pass_net.analyze(frame_data)
                
                # 4.2 Space Control (Voronoi)
                voronoi_overlay = None
                if has_matrix:
                    voronoi_overlay = self.space_control.generate_voronoi(frame_data)
                    
                # 4.3 Heatmap
                heatmap_overlay = None
                if has_matrix:
                    self.heatmap_generator.update(frame_data)
                    # Generating every frame might be wasteful, can be done every N frames
                    # Here we generate every frame for demonstration
                    heatmap_overlay = self.heatmap_generator.generate_overlay()

                # ==========================================
                # === Stage 5: Visualization (Rendering) ===
                # ==========================================
                # Delegate all drawing logic to _draw_frame to avoid code duplication
                # Note: Passing final_kpts for debug display
                canvas = self._draw_frame(frame, frame_data, final_kpts, has_matrix, pass_lines, voronoi_overlay, heatmap_overlay)

                # Write to video
                sink.write_frame(canvas)

        print(f"✅ Done! Saved to {self.cfg.OUTPUT_VIDEO}")

    def _draw_frame(self, frame, frame_data, kpts_xy, has_matrix, pass_lines, voronoi_overlay, heatmap_overlay):
        """
        Handles all drawing logic for the current frame.
        Args:
            has_matrix: Whether a projection matrix is currently available (determines if minimap is drawn)
            pass_lines: List of passing lines from network analysis
            voronoi_overlay: Voronoi layer
            heatmap_overlay: Heatmap layer
        """
        annotated_frame = frame.copy()

        # 1. Draw Pitch Keypoints (Debug, can be commented out)
        if kpts_xy is not None:
            for x, y in kpts_xy:
                # Yellow dot
                cv2.circle(annotated_frame, (int(x), int(y)), 3, (0, 255, 255), -1)

        # 2. Draw Passing Network
        # Draw lines before players so they appear underneath
        for start, end, opacity in pass_lines:
            overlay = annotated_frame.copy()
            # Draw line
            cv2.line(overlay, start, end, (255, 255, 0), 2, cv2.LINE_AA)
            # Blend layer for transparency
            cv2.addWeighted(overlay, opacity, annotated_frame, 1 - opacity, 0, annotated_frame)
            # Draw end dot
            cv2.circle(annotated_frame, end, 4, (255, 255, 0), -1)

        # 3. Draw Players (Box + Label)
        if len(frame_data.players) > 0:
            xyxy = np.array([p.rect for p in frame_data.players])

            # --- Color Mapping Logic ---
            color_indices = []
            labels = []
            for p in frame_data.players:
                idx = 4 # Default Grey
                lbl = f"#{p.id}"

                if p.team == TeamID.A: idx = 0          # Red
                elif p.team == TeamID.B: idx = 1        # Blue
                elif p.team == TeamID.REFEREE: idx = 2; lbl = "Ref" # Yellow
                elif p.team == TeamID.GOALKEEPER: idx = 3; lbl = "GK" # Black

                color_indices.append(idx)
                labels.append(lbl)

            # Construct detections
            sv_dets = sv.Detections(xyxy=xyxy, class_id=np.array(color_indices))

            # Apply color palette
            self.box_annotator.color = self.palette
            self.label_annotator.color = self.palette

            # Draw boxes and labels
            annotated_frame = self.box_annotator.annotate(annotated_frame, sv_dets)
            annotated_frame = self.label_annotator.annotate(annotated_frame, sv_dets, labels=labels)

        # 4. Draw Ball
        if frame_data.ball:
            b_det = sv.Detections(xyxy=np.array([frame_data.ball.rect]), class_id=np.array([0]))
            annotated_frame = self.ball_annotator.annotate(annotated_frame, b_det)

        # 5. Draw Minimap (Overlay)
        if has_matrix:
            # Generate full-size minimap with Voronoi and Heatmap overlays
            minimap = self.minimap_renderer.draw(frame_data, voronoi_overlay, heatmap_overlay)

            # Calculate scaled dimensions (Fixed width 300px)
            h, w, _ = minimap.shape
            target_w = 300
            scale = target_w / w
            target_h = int(h * scale)

            # Resize
            minimap_small = cv2.resize(minimap, (target_w, target_h))

            # Safety check: Prevent minimap from being larger than video
            canvas_h, canvas_w, _ = annotated_frame.shape
            if 20 + target_h < canvas_h and 20 + target_w < canvas_w:
                # Paste (Top-Left, offset 20px)
                annotated_frame[20:20+target_h, 20:20+target_w] = minimap_small

                # Add a refined white border (Thickness=1)
                cv2.rectangle(annotated_frame, (20, 20), (20+target_w, 20+target_h), (255, 255, 255), 1)
        else:
            # If no matrix at all (System initializing)
            cv2.putText(annotated_frame, "Seeking Pitch...", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        return annotated_frame