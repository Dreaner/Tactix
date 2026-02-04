"""
Project: Tactix
File Created: 2026-02-04 16:07:03
Author: Xingnan Zhu
File Name: system.py
Description: 把所有模块串联成一条流水线。
"""


import cv2
import numpy as np
from tqdm import tqdm
import supervision as sv

# 引入各模块
from tactix.config import Config
from tactix.vision.detector import Detector
from tactix.vision.pose import PitchEstimator
from tactix.vision.tracker import Tracker
from tactix.vision.transformer import ViewTransformer
from tactix.semantics.team import TeamClassifier
from tactix.tactics.pass_network import PassNetwork
from tactix.visualization.minimap import MinimapRenderer
from tactix.core.types import TeamID

class TactixEngine:
    def __init__(self):
        self.cfg = Config()
        print("🚀 Initializing Tactix Engine...")

        # 1. 初始化感知模块
        self.pitch_estimator = PitchEstimator(self.cfg.PITCH_MODEL_PATH, self.cfg.DEVICE)
        self.detector = Detector(self.cfg.PLAYER_MODEL_PATH, self.cfg.DEVICE, self.cfg.CONF_PLAYER)
        self.tracker = Tracker()
        
        # 2. 初始化逻辑模块
        self.transformer = ViewTransformer()
        self.team_classifier = TeamClassifier(device='cpu')
        self.pass_net = PassNetwork(self.cfg.MAX_PASS_DIST, self.cfg.BALL_OWNER_DIST)
        
        # 3. 初始化渲染模块
        self.minimap_renderer = MinimapRenderer(self.cfg.PITCH_TEMPLATE)
        self._init_annotators()

        # 状态
        self.classifier_trained = False

    def _init_annotators(self):
        """初始化 Supervision 画图工具"""
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.4)
        self.ball_annotator = sv.DotAnnotator(color=sv.Color.WHITE, radius=5)
        # 颜色板
        self.palette = sv.ColorPalette(colors=[
            sv.Color(230, 57, 70),   # A队: 红
            sv.Color(69, 123, 157),  # B队: 蓝
            sv.Color(255, 255, 0),   # 裁判: 黄
            sv.Color(0, 0, 0),       # 门将: 黑
            sv.Color(128, 128, 128)  # 未知: 灰
        ])

    def run(self):
        """主运行循环"""
        video_info = sv.VideoInfo.from_video_path(self.cfg.INPUT_VIDEO)
        frames = sv.get_video_frames_generator(self.cfg.INPUT_VIDEO)
        
        print(f"▶️ Processing: {self.cfg.INPUT_VIDEO}")

        with sv.VideoSink(self.cfg.OUTPUT_VIDEO, video_info) as sink:
            for i, frame in tqdm(enumerate(frames), total=video_info.total_frames):
                
                # === Stage 1: Pitch Calibration (世界观) ===
                kpts_xy, kpts_conf = self.pitch_estimator.predict(frame)
                is_calibrated = self.transformer.update(kpts_xy, kpts_conf, self.cfg.CONF_PITCH)

                # === Stage 2: Player Detection (实体) ===
                frame_data = self.detector.detect(frame, i)
                
                # 跟踪 (Tracking)
                if len(frame_data.players) > 0:
                    # 构造 tracker 需要的 sv.Detections
                    xyxy = np.array([p.rect for p in frame_data.players])
                    class_ids = np.array([p.class_id for p in frame_data.players])
                    sv_dets = sv.Detections(xyxy=xyxy, class_id=class_ids)
                    self.tracker.update(sv_dets, frame_data)

                # 球队分类 (Team Classification)
                valid_players = [p for p in frame_data.players if p.team == TeamID.UNKNOWN]
                if not self.classifier_trained and len(valid_players) > 3 and i < 30:
                    self.team_classifier.fit(frame, frame_data.players)
                    if i > 15: self.classifier_trained = True
                
                if self.classifier_trained:
                    self.team_classifier.predict(frame, frame_data)

                # === Stage 3: Coordinate Mapping (映射) ===
                if is_calibrated:
                    self.transformer.transform_players(frame_data.players)
                    if frame_data.ball:
                        # 单独转换球
                        ball_pt = self.transformer.transform_point(frame_data.ball.center)
                        if ball_pt:
                             # 临时存入 pitch_position (这里假设 Ball 类也有这个字段)
                             from tactix.core.types import Point
                             frame_data.ball.pitch_position = Point(x=ball_pt[0], y=ball_pt[1])

                # === Stage 4: Visualization (渲染) ===
                canvas = self._draw_frame(frame, frame_data, kpts_xy, is_calibrated)
                sink.write_frame(canvas)

        print(f"✅ Done! Saved to {self.cfg.OUTPUT_VIDEO}")

    def _draw_frame(self, frame, frame_data, kpts_xy, is_calibrated):
        """负责所有绘图逻辑，保持主循环干净"""
        annotated_frame = frame.copy()

        # 1. 画球场关键点 (Debug)
        if kpts_xy is not None:
            for x, y in kpts_xy:
                cv2.circle(annotated_frame, (int(x), int(y)), 3, (0, 255, 255), -1)

        # 2. 画球员
        if len(frame_data.players) > 0:
            xyxy = np.array([p.rect for p in frame_data.players])
            # 简单的颜色映射逻辑... (此处省略，复用之前的 color_indices 逻辑)
            # 为了简洁，这里暂时全部用红色，实际请把之前的 color_indices 逻辑搬过来
            class_ids = np.zeros(len(xyxy), dtype=int) 
            # ...你需要在这里实现把 TeamID 转成 color index (0-4)
            
            # 使用之前的逻辑填充 class_ids
            color_indices = []
            labels = []
            for p in frame_data.players:
                idx = 4
                lbl = f"#{p.id}"
                if p.team == TeamID.A: idx = 0
                elif p.team == TeamID.B: idx = 1
                elif p.team == TeamID.REFEREE: idx = 2; lbl = "Ref"
                elif p.team == TeamID.GOALKEEPER: idx = 3; lbl = "GK"
                color_indices.append(idx)
                labels.append(lbl)
            
            sv_dets = sv.Detections(xyxy=xyxy, class_id=np.array(color_indices))
            self.box_annotator.color = self.palette
            self.label_annotator.color = self.palette
            
            annotated_frame = self.box_annotator.annotate(annotated_frame, sv_dets)
            annotated_frame = self.label_annotator.annotate(annotated_frame, sv_dets, labels=labels)

        # 3. 画球
        if frame_data.ball:
            b_det = sv.Detections(xyxy=np.array([frame_data.ball.rect]), class_id=np.array([0]))
            annotated_frame = self.ball_annotator.annotate(annotated_frame, b_det)

        # 4. 画小地图 (如果校准成功)
        if is_calibrated:
            minimap = self.minimap_renderer.draw(frame_data)
            # 贴图逻辑
            h, w, _ = minimap.shape
            target_w = 300
            scale = target_w / w
            target_h = int(h * scale)
            minimap_small = cv2.resize(minimap, (target_w, target_h))
            
            # 贴到左上角
            annotated_frame[20:20+target_h, 20:20+target_w] = minimap_small
            # 绿框
            cv2.rectangle(annotated_frame, (20, 20), (20+target_w, 20+target_h), (0, 255, 0), 2)
        else:
            cv2.putText(annotated_frame, "Seeking Pitch...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        return annotated_frame