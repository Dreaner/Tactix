"""
Project: Tactix
File Created: 2026-02-04 16:07:03
Author: Xingnan Zhu
File Name: system.py
Description:
    The core engine of the Tactix system. Orchestrates the full processing
    pipeline across six focused stage methods, keeping the main loop clean
    and each stage independently readable.
"""

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm

from tactix.config import Config, CalibrationMode, Colors
from tactix.core.registry import PlayerRegistry, BallStateTracker
from tactix.core.types import TeamID, Point, FrameData, TacticalOverlays
from tactix.semantics.team import TeamClassifier
from tactix.tactics.base.pass_network import PassNetwork
from tactix.tactics.base.space_control import SpaceControl
from tactix.tactics.base.heatmap import HeatmapGenerator
from tactix.tactics.base.team_compactness import TeamCompactness
from tactix.tactics.base.pressure_index import PressureIndex
from tactix.tactics.base.cover_shadow import CoverShadow
from tactix.tactics.base.team_centroid import TeamCentroid
from tactix.tactics.base.team_width_length import TeamWidthLength
from tactix.vision.detector import Detector
from tactix.vision.calibration.ai_estimator import AIPitchEstimator
from tactix.vision.calibration.manual_estimator import ManualPitchEstimator
from tactix.vision.calibration.panorama_estimator import PanoramaPitchEstimator
from tactix.vision.tracker import Tracker
from tactix.vision.transformer import ViewTransformer
from tactix.visualization.minimap import MinimapRenderer
from tactix.vision.camera import CameraTracker
from tactix.export.json_exporter import JsonExporter
from tactix.analytics.events.event_detector import EventDetector
from tactix.analytics.attacking.shot_map import ShotMap
from tactix.analytics.attacking.zone_analyzer import ZoneAnalyzer
from tactix.analytics.attacking.pass_sonar import PassSonar
from tactix.analytics.attacking.buildup_tracker import BuildupTracker
from tactix.visualization.overlays.attacking.shot_map import ShotMapOverlay
from tactix.visualization.overlays.attacking.zone_14 import Zone14Overlay
from tactix.visualization.overlays.attacking.pass_sonar import PassSonarOverlay
from tactix.visualization.overlays.attacking.buildup import BuildupOverlay


class TactixEngine:
    def __init__(self, cfg: Config = None, manual_keypoints=None):
        self.cfg = cfg or Config()
        print("🚀 Initializing Tactix Engine...")

        # Apply manual keypoints override after config is set
        if manual_keypoints:
            self.cfg.CALIBRATION_MODE = CalibrationMode.PANORAMA

        # ==========================================
        # 1. Perception Modules
        # ==========================================
        if self.cfg.GEOMETRY_ENABLED:
            if self.cfg.CALIBRATION_MODE == CalibrationMode.MANUAL_FIXED:
                print("🔧 Mode: Manual Fixed Calibration")
                self.pitch_estimator = ManualPitchEstimator(manual_keypoints)
            elif self.cfg.CALIBRATION_MODE == CalibrationMode.PANORAMA:
                print("🌐 Mode: Panorama Calibration")
                self.pitch_estimator = PanoramaPitchEstimator(manual_keypoints)
            else:
                print("🤖 Mode: AI Auto Calibration")
                self.pitch_estimator = AIPitchEstimator(self.cfg.PITCH_MODEL_PATH, self.cfg.DEVICE)
        else:
            self.pitch_estimator = None

        self.detector = Detector(self.cfg.PLAYER_MODEL_PATH, self.cfg.DEVICE, self.cfg.CONF_PLAYER)
        self.tracker = Tracker()
        self.camera_tracker = CameraTracker(smoothing_window=5)

        # ==========================================
        # 2. Logic Modules
        # ==========================================
        self.transformer = ViewTransformer()
        self.team_classifier = TeamClassifier(device=self.cfg.DEVICE)
        self.pass_net = PassNetwork(self.cfg.MAX_PASS_DIST, self.cfg.BALL_OWNER_DIST)
        self.space_control = SpaceControl()
        self.heatmap_generator = HeatmapGenerator()
        self.team_compactness = TeamCompactness()
        self.pressure_index = PressureIndex(self.cfg.PRESSURE_RADIUS)
        self.cover_shadow = CoverShadow(self.cfg.SHADOW_LENGTH, self.cfg.SHADOW_ANGLE)
        self.team_centroid = TeamCentroid()
        self.team_width_length = TeamWidthLength()

        # ==========================================
        # 3. Visualization Modules
        # ==========================================
        self.minimap_renderer = MinimapRenderer(self.cfg.PITCH_TEMPLATE)
        self._init_annotators()

        # ==========================================
        # 4. Export Module
        # ==========================================
        self.exporter = None
        if self.cfg.EXPORT_DATA:
            print(f"💾 Data Export Enabled: {self.cfg.OUTPUT_JSON}")
            self.exporter = JsonExporter(self.cfg.OUTPUT_JSON)

        # State
        self.classifier_trained = False
        self.player_registry = PlayerRegistry()
        self.ball_state_tracker = BallStateTracker()
        self.event_detector = EventDetector(self.cfg)

        # ==========================================
        # 5. M1 Attacking Phase Modules
        # ==========================================
        self.shot_map = ShotMap()
        self.zone_analyzer = ZoneAnalyzer(self.cfg)
        self.pass_sonar = PassSonar()
        self.buildup_tracker = BuildupTracker(self.cfg)

    def _init_annotators(self):
        """Initialize Supervision annotators and color palette."""
        # EllipseAnnotator draws a colored arc at the base of each detection bbox (feet area).
        # ~280° arc with a gap at the bottom, matching the football_analysis visual style.
        self.ellipse_annotator = sv.EllipseAnnotator(thickness=2, start_angle=-45, end_angle=235)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.4, text_padding=4)

        self.palette = sv.ColorPalette(colors=[
            Colors.to_sv(Colors.TEAM_A),
            Colors.to_sv(Colors.TEAM_B),
            Colors.to_sv(Colors.REFEREE),
            Colors.to_sv(Colors.GOALKEEPER),
            Colors.to_sv(Colors.UNKNOWN),
        ])

    # ==========================================
    # Main Loop
    # ==========================================

    def run(self):
        """Main processing loop — delegates each frame to focused stage methods."""
        video_info = sv.VideoInfo.from_video_path(self.cfg.INPUT_VIDEO)
        frames = sv.get_video_frames_generator(self.cfg.INPUT_VIDEO)

        print(f"▶️ Processing: {self.cfg.INPUT_VIDEO}")
        print(f"   - Total Frames: {video_info.total_frames}")
        print(f"   - Resolution: {video_info.width}x{video_info.height}")

        with sv.VideoSink(self.cfg.OUTPUT_VIDEO, video_info) as sink:
            for i, frame in tqdm(enumerate(frames), total=video_info.total_frames):
                active_kps, has_matrix = self._stage_calibration(frame)
                frame_data = self.detector.detect(frame, i)
                frame_data.ball = self.ball_state_tracker.update(i, frame_data.ball)
                self._stage_tracking(frame_data)
                self._stage_classification(frame, frame_data, i, has_matrix)

                overlays = TacticalOverlays()
                if has_matrix:
                    self._stage_coordinate_mapping(frame_data)
                    overlays = self._stage_tactical_analysis(frame_data)

                canvas = self._stage_visualization(frame, frame_data, active_kps, has_matrix, overlays)
                sink.write_frame(canvas)

                if self.exporter:
                    self.exporter.add_frame(frame_data)

        if self.exporter:
            self.exporter.save()

        print(f"✅ Done! Saved to {self.cfg.OUTPUT_VIDEO}")

    # ==========================================
    # Stage Methods
    # ==========================================

    def _stage_calibration(self, frame: np.ndarray) -> tuple[np.ndarray | None, bool]:
        """Stage 1: Estimate pitch keypoints and update the homography matrix."""
        if not self.cfg.GEOMETRY_ENABLED or not self.pitch_estimator:
            return None, False

        pitch_keypoints, keypoint_confidences = self.pitch_estimator.predict(frame)
        active_keypoints = None

        if self.cfg.CALIBRATION_MODE == CalibrationMode.AI_ONLY:
            if pitch_keypoints is not None and len(pitch_keypoints) >= 4:
                self.camera_tracker.reset(pitch_keypoints, frame)
                active_keypoints = pitch_keypoints
            else:
                tracked = self.camera_tracker.update(frame)
                if tracked is not None:
                    active_keypoints = tracked
                    keypoint_confidences = np.ones(len(active_keypoints))
        else:
            active_keypoints = pitch_keypoints

        if active_keypoints is None:
            return None, False

        has_matrix = self.transformer.update(active_keypoints, keypoint_confidences, self.cfg.CONF_PITCH)
        return active_keypoints, has_matrix

    def _stage_tracking(self, frame_data: FrameData) -> None:
        """Stage 2: Assign persistent track IDs via ByteTrack."""
        if not frame_data.players:
            return
        xyxy = np.array([p.rect for p in frame_data.players])
        class_ids = np.array([p.class_id for p in frame_data.players])
        confidences = np.array([p.confidence for p in frame_data.players])
        sv_dets = sv.Detections(xyxy=xyxy, class_id=class_ids, confidence=confidences)
        self.tracker.update(sv_dets, frame_data)

    def _stage_classification(
        self, frame: np.ndarray, frame_data: FrameData, frame_index: int, has_matrix: bool
    ) -> None:
        """Stage 3: Train team classifier then assign teams via persistent registry voting."""
        # --- Training window (unchanged logic) ---
        training_candidates = [
            p for p in frame_data.players if p.class_id == 2 and p.team == TeamID.UNKNOWN
        ]
        if not self.classifier_trained and len(training_candidates) > 3 and frame_index < 30:
            self.team_classifier.fit(frame, training_candidates)
            if frame_index > 15:
                self.classifier_trained = True

        if not self.classifier_trained:
            return

        # --- Per-player classification with persistent registry ---
        for p in frame_data.players:
            pid = p.id
            if pid == -1:
                continue  # Untracked; no stable ID to key the registry on

            # YOLO-trusted roles: hard-set and freeze in registry
            if p.class_id == 1:
                p.team = TeamID.GOALKEEPER
                self.player_registry.override_team(pid, TeamID.GOALKEEPER)
                continue
            if p.class_id == 3:
                p.team = TeamID.REFEREE
                self.player_registry.override_team(pid, TeamID.REFEREE)
                continue

            # Outfield players only from here
            if p.class_id != 2:
                continue

            if self.player_registry.is_confirmed(pid):
                # Stable assignment already locked in — no need to re-run K-Means
                p.team = self.player_registry.get_team(pid)
            else:
                # Still accumulating evidence: extract color, vote, apply best guess
                color = self.team_classifier._extract_shirt_color(frame, p.rect)
                if color is not None:
                    self.player_registry.record_color_sample(pid, color)
                    predicted = self.team_classifier.predict_one(color)
                    self.player_registry.record_team_vote(pid, predicted)
                p.team = self.player_registry.get_team(pid)

        # Apply heuristic GK correction only when we have pitch geometry
        if has_matrix:
            self._apply_goalkeeper_heuristic(frame_data)

    def _apply_goalkeeper_heuristic(self, frame_data: FrameData) -> None:
        """
        Heuristic fallback: if YOLO misses a goalkeeper, promote the deepest
        outfield player inside each penalty box.
        YOLO classes: 0=ball, 1=goalkeeper, 2=player, 3=referee
        """
        # Team A defends the left goal (x ≈ 0)
        gk_a = any(
            p.team == TeamID.GOALKEEPER and p.pitch_position and p.pitch_position.x < 52.5
            for p in frame_data.players
        )
        if not gk_a:
            candidates_a = [p for p in frame_data.players if p.team == TeamID.A and p.pitch_position]
            if candidates_a:
                closest = min(candidates_a, key=lambda p: p.pitch_position.x)
                if closest.pitch_position.x < 16.5:
                    closest.team = TeamID.GOALKEEPER

        # Team B defends the right goal (x ≈ 105)
        gk_b = any(
            p.team == TeamID.GOALKEEPER and p.pitch_position and p.pitch_position.x > 52.5
            for p in frame_data.players
        )
        if not gk_b:
            candidates_b = [p for p in frame_data.players if p.team == TeamID.B and p.pitch_position]
            if candidates_b:
                closest = max(candidates_b, key=lambda p: p.pitch_position.x)
                if closest.pitch_position.x > (105 - 16.5):
                    closest.team = TeamID.GOALKEEPER

    def _stage_coordinate_mapping(self, frame_data: FrameData) -> None:
        """Stage 4: Project pixel coordinates to real-world pitch coordinates (meters)."""
        self.transformer.transform_players(frame_data.players)
        self.tracker.update_velocity(frame_data)

        if frame_data.ball:
            ball_pt = self.transformer.transform_point(frame_data.ball.center)
            if ball_pt:
                frame_data.ball.pitch_position = Point(x=ball_pt[0], y=ball_pt[1])

    def _stage_tactical_analysis(self, frame_data: FrameData) -> TacticalOverlays:
        """Stage 5: Run enabled tactical modules and return their overlay arrays."""
        frame_data.events = self.event_detector.detect(frame_data)
        overlays = TacticalOverlays()

        if self.cfg.SHOW_PASS_NETWORK:
            overlays.pass_lines = self.pass_net.analyze(frame_data)

        if self.cfg.SHOW_VORONOI:
            overlays.voronoi = self.space_control.generate_voronoi(frame_data)

        self.heatmap_generator.update(frame_data)
        if self.cfg.SHOW_HEATMAP:
            overlays.heatmap = self.heatmap_generator.generate_overlay()

        if self.cfg.SHOW_COMPACTNESS:
            overlays.compactness = self.team_compactness.generate_overlay(frame_data)

        if self.cfg.SHOW_PRESSURE:
            self.pressure_index.calculate(frame_data)

        if self.cfg.SHOW_COVER_SHADOW:
            overlays.shadow = self.cover_shadow.generate_overlay(frame_data)

        if self.cfg.SHOW_TEAM_CENTROID:
            overlays.centroid = self.team_centroid.generate_overlay(frame_data)

        if self.cfg.SHOW_TEAM_WIDTH_LENGTH:
            overlays.width_length = self.team_width_length.generate_overlay(frame_data)

        # ==========================================
        # M1 — Attacking Phase
        # ==========================================
        events = frame_data.events
        if events is not None:
            # Always accumulate (state updates are cheap)
            for shot in events.shots:
                self.shot_map.record(shot)
            for pass_evt in events.passes:
                self.zone_analyzer.record(pass_evt)
                self.pass_sonar.record(pass_evt)
            self.buildup_tracker.update(frame_data.frame_index, events)

        # Always update pass sonar positions (cheap, needed for rendering)
        self.pass_sonar.update_positions(frame_data)

        # Generate overlays via dedicated renderer classes
        cw = self.minimap_renderer.canvas_w
        ch = self.minimap_renderer.canvas_h
        if self.cfg.SHOW_SHOT_MAP:
            overlays.shot_map = ShotMapOverlay.render(self.shot_map, cw, ch)
        if self.cfg.SHOW_ZONE_14:
            overlays.zone_14 = Zone14Overlay.render(self.zone_analyzer, cw, ch)
        if self.cfg.SHOW_PASS_SONAR:
            overlays.pass_sonar = PassSonarOverlay.render(self.pass_sonar, cw, ch)
        if self.cfg.SHOW_BUILDUP:
            overlays.buildup = BuildupOverlay.render(self.buildup_tracker, cw, ch)

        return overlays

    def _stage_visualization(
        self,
        frame: np.ndarray,
        frame_data: FrameData,
        active_kps: np.ndarray | None,
        has_matrix: bool,
        overlays: TacticalOverlays,
    ) -> np.ndarray:
        """Stage 6: Annotate the frame and composite the minimap."""
        annotated = frame.copy()

        # Debug keypoints
        if self.cfg.GEOMETRY_ENABLED and self.cfg.SHOW_DEBUG_KEYPOINTS and active_kps is not None:
            for x, y in active_kps:
                cv2.circle(annotated, (int(x), int(y)), 3, Colors.to_bgr(Colors.KEYPOINT), -1)

        # Pass network lines (drawn on the main view)
        if self.cfg.GEOMETRY_ENABLED and self.cfg.SHOW_PASS_NETWORK:
            for start, end, opacity in overlays.pass_lines:
                layer = annotated.copy()
                cv2.line(layer, start, end, Colors.to_bgr(Colors.KEYPOINT), 2, cv2.LINE_AA)
                cv2.addWeighted(layer, opacity, annotated, 1 - opacity, 0, annotated)
                cv2.circle(annotated, end, 4, Colors.to_bgr(Colors.KEYPOINT), -1)

        # Player boxes and labels
        if frame_data.players:
            xyxy = np.array([p.rect for p in frame_data.players])
            color_indices, labels = [], []
            for p in frame_data.players:
                kmh = p.speed * 3.6
                speed_str = f" {kmh:.1f}" if kmh > 0.5 else ""
                if p.team == TeamID.A:
                    color_indices.append(0); labels.append(f"#{p.id}{speed_str}")
                elif p.team == TeamID.B:
                    color_indices.append(1); labels.append(f"#{p.id}{speed_str}")
                elif p.team == TeamID.REFEREE:
                    color_indices.append(2); labels.append("Ref")
                elif p.team == TeamID.GOALKEEPER:
                    color_indices.append(3); labels.append(f"GK{speed_str}")
                else:
                    color_indices.append(4); labels.append(f"#{p.id}{speed_str}")

            sv_dets = sv.Detections(xyxy=xyxy, class_id=np.array(color_indices))
            self.ellipse_annotator.color = self.palette
            self.label_annotator.color = self.palette
            annotated = self.ellipse_annotator.annotate(annotated, sv_dets)
            annotated = self.label_annotator.annotate(annotated, sv_dets, labels=labels)

        # Ball: downward-pointing triangle above the ball center
        if frame_data.ball:
            cx, cy = frame_data.ball.center
            size = 12       # half-width of the triangle base (px)
            tip_gap = 8     # gap between triangle tip and ball center (px)
            pts = np.array([
                [cx - size, cy - tip_gap - size],   # top-left
                [cx + size, cy - tip_gap - size],   # top-right
                [cx,        cy - tip_gap],           # tip (points down toward ball)
            ], dtype=np.int32)
            cv2.fillPoly(annotated, [pts], Colors.to_bgr(Colors.BALL))
            cv2.polylines(annotated, [pts], isClosed=True, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        # Minimap overlay (top-left corner)
        if has_matrix and self.cfg.SHOW_MINIMAP:
            minimap = self.minimap_renderer.draw(
                frame_data,
                overlays.voronoi,
                overlays.heatmap,
                overlays.compactness,
                overlays.shadow,
                overlays.centroid,
                overlays.width_length,
                show_velocity=self.cfg.SHOW_VELOCITY,
                show_pressure=self.cfg.SHOW_PRESSURE,
                shot_map_overlay=overlays.shot_map,
                zone_14_overlay=overlays.zone_14,
                pass_sonar_overlay=overlays.pass_sonar,
                buildup_overlay=overlays.buildup,
            )
            h, w, _ = minimap.shape
            target_w = 300
            target_h = int(h * target_w / w)
            minimap_small = cv2.resize(minimap, (target_w, target_h))

            canvas_h, canvas_w, _ = annotated.shape
            if 20 + target_h < canvas_h and 20 + target_w < canvas_w:
                annotated[20:20 + target_h, 20:20 + target_w] = minimap_small
                cv2.rectangle(annotated, (20, 20), (20 + target_w, 20 + target_h), Colors.to_bgr(Colors.TEXT), 1)

        elif not self.cfg.GEOMETRY_ENABLED:
            cv2.putText(annotated, "GEOMETRY DISABLED", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, Colors.to_bgr(Colors.KEYPOINT), 2)

        return annotated
