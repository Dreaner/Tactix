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

# 引入各模块
from tactix.config import Config
from tactix.core.types import TeamID, Point
from tactix.semantics.team import TeamClassifier
from tactix.tactics.pass_network import PassNetwork
from tactix.vision.detector import Detector
from tactix.vision.pose import PitchEstimator, MockPitchEstimator
from tactix.vision.tracker import Tracker
from tactix.vision.transformer import ViewTransformer
from tactix.visualization.minimap import MinimapRenderer


class TactixEngine:
    def __init__(self):
        self.cfg = Config()
        print("🚀 Initializing Tactix Engine...")

        # ==========================================
        # 1. 初始化感知模块 (Perception)
        # ==========================================
        # 根据配置决定使用真 AI 还是 Mock 数据
        if self.cfg.USE_MOCK_PITCH:
            self.pitch_estimator = MockPitchEstimator(self.cfg.MOCK_KEYPOINTS)
        else:
            self.pitch_estimator = PitchEstimator(self.cfg.PITCH_MODEL_PATH, self.cfg.DEVICE)

        self.detector = Detector(self.cfg.PLAYER_MODEL_PATH, self.cfg.DEVICE, self.cfg.CONF_PLAYER)
        self.tracker = Tracker()

        # ==========================================
        # 2. 初始化逻辑模块 (Logic)
        # ==========================================
        self.transformer = ViewTransformer()
        self.team_classifier = TeamClassifier(device='cpu') # 暂用 CPU，如果有 GPU 可改 'mps' 或 'cuda'
        self.pass_net = PassNetwork(self.cfg.MAX_PASS_DIST, self.cfg.BALL_OWNER_DIST)

        # ==========================================
        # 3. 初始化渲染模块 (Visualization)
        # ==========================================
        self.minimap_renderer = MinimapRenderer(self.cfg.PITCH_TEMPLATE)
        self._init_annotators()

        # 状态标记
        self.classifier_trained = False

    def _init_annotators(self):
        """初始化 Supervision 画图工具"""
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.4, text_padding=4)
        self.ball_annotator = sv.DotAnnotator(color=sv.Color.WHITE, radius=5)

        # 定义颜色板 (对应 class_id 0-4)
        self.palette = sv.ColorPalette(colors=[
            sv.Color(230, 57, 70),   # 0: A队 (红)
            sv.Color(69, 123, 157),  # 1: B队 (蓝)
            sv.Color(255, 255, 0),   # 2: 裁判 (黄)
            sv.Color(0, 0, 0),       # 3: 门将 (黑)
            sv.Color(128, 128, 128)  # 4: 未知 (灰)
        ])

    def run(self):
        """主运行循环"""
        # 准备视频流
        video_info = sv.VideoInfo.from_video_path(self.cfg.INPUT_VIDEO)
        frames = sv.get_video_frames_generator(self.cfg.INPUT_VIDEO)

        print(f"▶️ Processing: {self.cfg.INPUT_VIDEO}")
        print(f"   - Total Frames: {video_info.total_frames}")
        print(f"   - Resolution: {video_info.width}x{video_info.height}")

        with sv.VideoSink(self.cfg.OUTPUT_VIDEO, video_info) as sink:
            # 使用 tqdm 显示进度条
            for i, frame in tqdm(enumerate(frames), total=video_info.total_frames):

                # ==========================================
                # === Stage 1: Pitch Calibration (世界观) ===
                # ==========================================
                kpts_xy, kpts_conf = self.pitch_estimator.predict(frame)

                # 更新矩阵 (只要有矩阵可用，无论是新的还是旧的，都返回 True)
                has_matrix = self.transformer.update(kpts_xy, kpts_conf, self.cfg.CONF_PITCH)

                # ==========================================
                # === Stage 2: Player Detection (实体) ===
                # ==========================================
                frame_data = self.detector.detect(frame, i)

                # --- A. 跟踪模块 (Tracking) ---
                if len(frame_data.players) > 0:
                    xyxy = np.array([p.rect for p in frame_data.players])
                    class_ids = np.array([p.class_id for p in frame_data.players])
                    confidences = np.array([p.confidence for p in frame_data.players])
                    sv_dets = sv.Detections(xyxy=xyxy, class_id=class_ids, confidence=confidences)
                    self.tracker.update(sv_dets, frame_data)

                # --- B. 球队分类 (Team Classification) ---
                # 在前 30 帧积累数据，训练颜色分类器
                valid_players = [p for p in frame_data.players if p.team == TeamID.UNKNOWN]
                if not self.classifier_trained and len(valid_players) > 3 and i < 30:
                    self.team_classifier.fit(frame, frame_data.players)
                    if i > 15: self.classifier_trained = True

                # 如果训练好了，就开始预测每人的队伍
                if self.classifier_trained:
                    self.team_classifier.predict(frame, frame_data)

                # ==========================================
                # === Stage 3: Coordinate Mapping (映射) ===
                # ==========================================
                if has_matrix:
                    self.transformer.transform_players(frame_data.players)

                    if frame_data.ball:
                        # 单独转换球
                        ball_pt = self.transformer.transform_point(frame_data.ball.center)
                        if ball_pt:
                             frame_data.ball.pitch_position = Point(x=ball_pt[0], y=ball_pt[1])

                # ==========================================
                # === Stage 4: Visualization (渲染) ===
                # ==========================================
                # 将所有绘图逻辑委托给 _draw_frame，避免重复代码
                canvas = self._draw_frame(frame, frame_data, kpts_xy, has_matrix)

                # 写入视频
                sink.write_frame(canvas)

        print(f"✅ Done! Saved to {self.cfg.OUTPUT_VIDEO}")

    def _draw_frame(self, frame, frame_data, kpts_xy, has_matrix):
        """
        负责这一帧所有的绘图逻辑。
        Args:
            has_matrix: 当前是否有可用的投影矩阵（决定是否画小地图）
        """
        annotated_frame = frame.copy()

        # 1. 画球场关键点 (Debug用，可以注释掉)
        if kpts_xy is not None:
            for x, y in kpts_xy:
                # 黄色小点
                cv2.circle(annotated_frame, (int(x), int(y)), 3, (0, 255, 255), -1)

        # 2. 画球员 (Box + Label)
        if len(frame_data.players) > 0:
            xyxy = np.array([p.rect for p in frame_data.players])

            # --- 颜色映射逻辑 ---
            color_indices = []
            labels = []
            for p in frame_data.players:
                idx = 4 # 默认灰色
                lbl = f"#{p.id}"

                if p.team == TeamID.A: idx = 0          # 红
                elif p.team == TeamID.B: idx = 1        # 蓝
                elif p.team == TeamID.REFEREE: idx = 2; lbl = "Ref" # 黄
                elif p.team == TeamID.GOALKEEPER: idx = 3; lbl = "GK" # 黑

                color_indices.append(idx)
                labels.append(lbl)

            # 构造 detections
            sv_dets = sv.Detections(xyxy=xyxy, class_id=np.array(color_indices))

            # 应用颜色板
            self.box_annotator.color = self.palette
            self.label_annotator.color = self.palette

            # 画框和标签
            annotated_frame = self.box_annotator.annotate(annotated_frame, sv_dets)
            annotated_frame = self.label_annotator.annotate(annotated_frame, sv_dets, labels=labels)

        # 3. 画球
        if frame_data.ball:
            b_det = sv.Detections(xyxy=np.array([frame_data.ball.rect]), class_id=np.array([0]))
            annotated_frame = self.ball_annotator.annotate(annotated_frame, b_det)

        # 4. 画小地图 (Overlay Minimap)
        if has_matrix:
            # 生成全尺寸小地图
            minimap = self.minimap_renderer.draw(frame_data)

            # 计算缩放尺寸 (固定宽度 300px)
            h, w, _ = minimap.shape
            target_w = 300
            scale = target_w / w
            target_h = int(h * scale)

            # 缩放
            minimap_small = cv2.resize(minimap, (target_w, target_h))

            # 安全检查：防止小地图比视频还大
            canvas_h, canvas_w, _ = annotated_frame.shape
            if 20 + target_h < canvas_h and 20 + target_w < canvas_w:
                # 贴图 (左上角，偏移20px)
                annotated_frame[20:20+target_h, 20:20+target_w] = minimap_small

                # 加一个精致的白色细边框 (Thickness=1)
                cv2.rectangle(annotated_frame, (20, 20), (20+target_w, 20+target_h), (255, 255, 255), 1)
        else:
            # 如果完全没有矩阵 (系统初始化中)
            cv2.putText(annotated_frame, "Seeking Pitch...", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        return annotated_frame