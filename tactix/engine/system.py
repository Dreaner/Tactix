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
from tactix.config import Config, CalibrationMode, Colors
from tactix.core.types import TeamID, Point
from tactix.semantics.team import TeamClassifier
from tactix.tactics.pass_network import PassNetwork
from tactix.tactics.space_control import SpaceControl
from tactix.tactics.heatmap import HeatmapGenerator
from tactix.tactics.team_compactness import TeamCompactness
from tactix.tactics.pressure_index import PressureIndex # Import PressureIndex
from tactix.vision.detector import Detector
from tactix.vision.calibration.ai_estimator import AIPitchEstimator
from tactix.vision.calibration.manual_estimator import ManualPitchEstimator
from tactix.vision.calibration.panorama_estimator import PanoramaPitchEstimator
from tactix.vision.tracker import Tracker
from tactix.vision.transformer import ViewTransformer
from tactix.visualization.minimap import MinimapRenderer
from tactix.vision.camera import CameraTracker
from tactix.export.json_exporter import JsonExporter

class TactixEngine:
    def __init__(self, manual_keypoints=None):
        self.cfg = Config()
        print("🚀 Initializing Tactix Engine...")
        
        # If manual keypoints are provided (from interactive mode), override config
        if manual_keypoints:
            self.cfg.MANUAL_KEYPOINTS = manual_keypoints
            # Default to Panorama mode if manual points are provided, as it's more robust for moving cameras
            self.cfg.CALIBRATION_MODE = CalibrationMode.PANORAMA

        # ==========================================
        # 1. Initialize Perception Modules
        # ==========================================
        # Select Pitch Estimator based on Calibration Mode
        if self.cfg.CALIBRATION_MODE == CalibrationMode.MANUAL_FIXED:
            print("🔧 Mode: Manual Fixed Calibration")
            self.pitch_estimator = ManualPitchEstimator(self.cfg.MANUAL_KEYPOINTS)
        elif self.cfg.CALIBRATION_MODE == CalibrationMode.PANORAMA:
            print("🌐 Mode: Panorama Calibration")
            self.pitch_estimator = PanoramaPitchEstimator(self.cfg.MANUAL_KEYPOINTS)
        else:
            print("🤖 Mode: AI Auto Calibration")
            self.pitch_estimator = AIPitchEstimator(self.cfg.PITCH_MODEL_PATH, self.cfg.DEVICE)

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
        self.team_compactness = TeamCompactness()
        self.pressure_index = PressureIndex(self.cfg.PRESSURE_RADIUS) # Initialize PressureIndex

        # ==========================================
        # 3. Initialize Visualization Modules
        # ==========================================
        self.minimap_renderer = MinimapRenderer(self.cfg.PITCH_TEMPLATE)
        self._init_annotators()
        
        # ==========================================
        # 4. Initialize Export Module
        # ==========================================
        self.exporter = None
        if self.cfg.EXPORT_DATA:
            print(f"💾 Data Export Enabled: {self.cfg.OUTPUT_JSON}")
            self.exporter = JsonExporter(self.cfg.OUTPUT_JSON)

        # State flags
        self.classifier_trained = False

    def _init_annotators(self):
        """Initialize Supervision annotators"""
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.4, text_padding=4)
        self.ball_annotator = sv.DotAnnotator(color=Colors.to_sv(Colors.BALL), radius=5)

        # Define color palette (corresponding to class_id 0-4)
        # Order: Team A, Team B, Referee, Goalkeeper, Unknown
        self.palette = sv.ColorPalette(colors=[
            Colors.to_sv(Colors.TEAM_A),
            Colors.to_sv(Colors.TEAM_B),
            Colors.to_sv(Colors.REFEREE),
            Colors.to_sv(Colors.GOALKEEPER),
            Colors.to_sv(Colors.UNKNOWN)
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
                # 1. Predict keypoints (AI, Manual, or Panorama)
                pitch_keypoints, keypoint_confidences = self.pitch_estimator.predict(frame)
                
                active_keypoints = None
                
                # 2. Refine Keypoints (Smoothing / Fallback)
                
                if self.cfg.CALIBRATION_MODE == CalibrationMode.AI_ONLY:
                    # AI Mode Logic: Trust AI if good, else use Optical Flow fallback
                    if pitch_keypoints is not None and len(pitch_keypoints) >= 4:
                        self.camera_tracker.reset(pitch_keypoints, frame)
                        active_keypoints = pitch_keypoints
                    else:
                        tracked_keypoints = self.camera_tracker.update(frame)
                        if tracked_keypoints is not None:
                            active_keypoints = tracked_keypoints
                            keypoint_confidences = np.ones(len(active_keypoints))
                else:
                    # Manual/Panorama Mode Logic: Trust the estimator directly
                    active_keypoints = pitch_keypoints

                # Update matrix (returns True as long as a matrix is available, new or old)
                has_matrix = False
                if active_keypoints is not None:
                    has_matrix = self.transformer.update(active_keypoints, keypoint_confidences, self.cfg.CONF_PITCH)

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
                pass_lines = []
                if self.cfg.SHOW_PASS_NETWORK:
                    pass_lines = self.pass_net.analyze(frame_data)
                
                # 4.2 Space Control (Voronoi)
                voronoi_overlay = None
                if has_matrix and self.cfg.SHOW_VORONOI:
                    voronoi_overlay = self.space_control.generate_voronoi(frame_data)
                    
                # 4.3 Heatmap
                heatmap_overlay = None
                if has_matrix:
                    self.heatmap_generator.update(frame_data)
                    if self.cfg.SHOW_HEATMAP:
                        heatmap_overlay = self.heatmap_generator.generate_overlay()
                    
                # 4.4 Team Compactness (Convex Hull)
                compactness_overlay = None
                if has_matrix and self.cfg.SHOW_COMPACTNESS:
                    compactness_overlay = self.team_compactness.generate_overlay(frame_data)
                    
                # 4.5 Pressure Index
                if self.cfg.SHOW_PRESSURE:
                    self.pressure_index.calculate(frame_data)

                # ==========================================
                # === Stage 5: Visualization (Rendering) ===
                # ==========================================
                canvas = self._draw_frame(frame, frame_data, active_keypoints, has_matrix, pass_lines, voronoi_overlay, heatmap_overlay, compactness_overlay)

                # Write to video
                sink.write_frame(canvas)
                
                # ==========================================
                # === Stage 6: Data Export ===
                # ==========================================
                if self.exporter:
                    self.exporter.add_frame(frame_data)

        # Save exported data
        if self.exporter:
            self.exporter.save()

        print(f"✅ Done! Saved to {self.cfg.OUTPUT_VIDEO}")

    def _draw_frame(self, frame, frame_data, pitch_keypoints, has_matrix, pass_lines, voronoi_overlay, heatmap_overlay, compactness_overlay):
        """
        Handles all drawing logic for the current frame.
        """
        annotated_frame = frame.copy()

        # 1. Draw Pitch Keypoints (Debug)
        if self.cfg.SHOW_DEBUG_KEYPOINTS and pitch_keypoints is not None:
            for x, y in pitch_keypoints:
                cv2.circle(annotated_frame, (int(x), int(y)), 3, Colors.to_bgr(Colors.KEYPOINT), -1)

        # 2. Draw Passing Network
        if self.cfg.SHOW_PASS_NETWORK:
            for start, end, opacity in pass_lines:
                overlay = annotated_frame.copy()
                # Draw line
                cv2.line(overlay, start, end, Colors.to_bgr(Colors.KEYPOINT), 2, cv2.LINE_AA)
                # Blend layer for transparency
                cv2.addWeighted(overlay, opacity, annotated_frame, 1 - opacity, 0, annotated_frame)
                # Draw end dot
                cv2.circle(annotated_frame, end, 4, Colors.to_bgr(Colors.KEYPOINT), -1)

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
            # Generate full-size minimap with Voronoi, Heatmap, and Compactness overlays
            # Pass config flags to renderer if needed, but here we control overlays via arguments
            minimap = self.minimap_renderer.draw(
                frame_data, 
                voronoi_overlay, 
                heatmap_overlay, 
                compactness_overlay,
                show_velocity=self.cfg.SHOW_VELOCITY, # Pass velocity flag
                show_pressure=self.cfg.SHOW_PRESSURE # Pass pressure flag
            )

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
                cv2.rectangle(annotated_frame, (20, 20), (20+target_w, 20+target_h), Colors.to_bgr(Colors.TEXT), 1)
        else:
            # If no matrix at all (System initializing)
            cv2.putText(annotated_frame, "Seeking Pitch...", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, Colors.to_bgr(Colors.KEYPOINT), 2)

        return annotated_frame