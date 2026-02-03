"""
Project: Tactix
File Created: 2026-02-02 11:55:51
Author: Xingnan Zhu
File Name: main.py
Description: xxx...
"""


"""
Project: Tactix
File Name: main.py
"""

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm

# === 核心模块引入 ===
from tactix.vision.detector import Detector
from tactix.vision.tracker import Tracker
from tactix.vision.camera import CameraTracker
from tactix.vision.transformer import ViewTransformer
from tactix.semantics.team import TeamClassifier
from tactix.tactics.pass_network import PassNetwork
from tactix.visualization.minimap import MinimapRenderer
from tactix.core.keypoints import get_target_points
from tactix.core.types import TeamID, Point

def main():
    # ==========================================
    # 1. 配置 (Config)
    # ==========================================
    MODEL_PATH = "assets/weights/football_v1.pt" 
    SOURCE_VIDEO_PATH = "assets/samples/InterGoalClip.mp4"
    TARGET_VIDEO_PATH = "assets/output/Final_Result.mp4"
    PITCH_IMAGE_PATH = "assets/pitch_bg.png"

    # 校准数据 (第0帧)
    CALIBRATION_SOURCE = np.array([(137, 89), (1126, 87), (1045, 398), (138, 222)])
    CALIBRATION_TARGETS = ['L_PA_TOP_LINE', 'MID_TOP', 'CIRCLE_BOTTOM', 'L_PENALTY_SPOT']

    # ==========================================
    # 2. 初始化 (Init)
    # ==========================================
    print(f"🚀 初始化 Tactix 模块...")

    # A. 视觉感知
    detector = Detector(model_weights=MODEL_PATH, device='mps', conf_threshold=0.1)
    tracker = Tracker()
    camera_tracker = CameraTracker(initial_keypoints=CALIBRATION_SOURCE) # 🎥 专门负责跟镜头

    # B. 语义与几何
    team_classifier = TeamClassifier(device='cpu')
    classifier_trained = False
    
    # 战术板目标点是固定的，只需要取一次
    target_points = get_target_points(CALIBRATION_TARGETS)

    # C. 战术分析
    pass_net = PassNetwork(max_pass_dist=400, ball_owner_dist=60)

    # D. 渲染器
    minimap_renderer = MinimapRenderer(bg_image_path=PITCH_IMAGE_PATH) # 🗺️ 专门负责画图
    
    # Supervision Annotators
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.4)
    ball_annotator = sv.DotAnnotator(color=sv.Color.WHITE, radius=5)

    # 视频流
    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

    # ==========================================
    # 3. 主循环 (Main Loop)
    # ==========================================
    print(f"▶️ 开始处理...")
    
    with sv.VideoSink(TARGET_VIDEO_PATH, video_info=video_info) as sink:
        for i, frame in tqdm(enumerate(frame_generator), total=video_info.total_frames):
            
            # --- [Step 0] 动态校准 (Camera Update) ---
            # 1. 让 camera_tracker 算出这一帧的那 4 个点跑哪去了
            current_points = camera_tracker.update(frame)
            
            # 2. 用新点重新生成变换矩阵
            view_transformer = ViewTransformer(
                source_points=current_points,
                target_points=target_points
            )

            # --- [Step 1] 检测与跟踪 ---
            frame_data = detector.detect(frame, frame_index=i)
            
            if len(frame_data.players) > 0:
                # 构造 tracker 需要的数据
                xyxy = np.array([p.rect for p in frame_data.players])
                class_ids = np.array([p.class_id for p in frame_data.players])
                confidences = np.array([0.8] * len(frame_data.players))
                
                detections_sv = sv.Detections(xyxy=xyxy, confidence=confidences, class_id=class_ids)
                tracker.update(detections_sv, frame_data)

            # --- [Step 2] 球队分类 ---
            valid_players = [p for p in frame_data.players if p.team == TeamID.UNKNOWN]
            if not classifier_trained and len(valid_players) > 5:
                team_classifier.fit(frame, frame_data.players)
                classifier_trained = True
            if classifier_trained:
                team_classifier.predict(frame, frame_data)

            # --- [Step 3] 坐标映射 (2D Mapping) ---
            view_transformer.transform_players(frame_data.players)
            if frame_data.ball:
                ball_pos = view_transformer.transform_point(frame_data.ball.center)
                if ball_pos:
                    frame_data.ball.pitch_position = Point(x=ball_pos[0], y=ball_pos[1])

            # --- [Step 4] 战术分析 ---
            pass_lines = pass_net.analyze(frame_data)

            # --- [Step 5] 渲染合成 (Rendering) ---
            annotated_frame = frame.copy()

            # 5.1 画传球线
            for start, end, opacity in pass_lines:
                overlay = annotated_frame.copy()
                cv2.line(overlay, start, end, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.addWeighted(overlay, opacity, annotated_frame, 1 - opacity, 0, annotated_frame)
                cv2.circle(annotated_frame, end, 4, (255, 255, 0), -1)

            # 5.2 画球员框 (Supervision)
            if len(frame_data.players) > 0:
                # 映射 TeamID 到颜色索引 (0-4)
                xyxy = np.array([p.rect for p in frame_data.players])
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
                
                det_viz = sv.Detections(xyxy=xyxy, class_id=np.array(color_indices))
                
                # 定义颜色板
                palette = sv.ColorPalette(colors=[
                    sv.Color(255, 0, 0), sv.Color(0, 0, 255), 
                    sv.Color(255, 255, 0), sv.Color(255, 165, 0), sv.Color(128, 128, 128)
                ])
                box_annotator.color = palette
                label_annotator.color = palette
                
                annotated_frame = box_annotator.annotate(annotated_frame, det_viz)
                annotated_frame = label_annotator.annotate(annotated_frame, det_viz, labels=labels)

            # 5.3 画球
            if frame_data.ball:
                ball_det = sv.Detections(xyxy=np.array([frame_data.ball.rect]), class_id=np.array([0]))
                annotated_frame = ball_annotator.annotate(annotated_frame, ball_det)

            # 5.4 调试：画出光流跟踪点 (绿点)
            for pt in current_points:
                cv2.circle(annotated_frame, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)

            # 5.5 画中画：小地图
            # 这一步直接调用我们封装好的 renderer
            minimap_img = minimap_renderer.draw(frame_data)
            
            # 缩放并贴图
            target_width = 350
            scale = target_width / minimap_img.shape[1]
            target_height = int(minimap_img.shape[0] * scale)
            minimap_small = cv2.resize(minimap_img, (target_width, target_height))
            
            x_off, y_off = 20, 20
            if y_off + target_height < annotated_frame.shape[0]:
                annotated_frame[y_off:y_off+target_height, x_off:x_off+target_width] = minimap_small
                cv2.rectangle(annotated_frame, (x_off, y_off), (x_off+target_width, y_off+target_height), (255, 255, 255), 2)

            sink.write_frame(annotated_frame)

    print(f"\n✅ 完成! 视频保存至: {TARGET_VIDEO_PATH}")

if __name__ == "__main__":
    main()