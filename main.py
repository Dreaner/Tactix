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
Description: V4 Automatic Calibration Pipeline
"""

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm

# === 核心模块引入 ===
from tactix.vision.detector import Detector        # Model B: 找人
from tactix.vision.pose import PitchEstimator      # Model A: 找球场 (新增!)
from tactix.vision.tracker import Tracker
from tactix.vision.transformer import ViewTransformer
from tactix.semantics.team import TeamClassifier
from tactix.tactics.pass_network import PassNetwork
from tactix.visualization.minimap import MinimapRenderer
from tactix.core.types import TeamID, Point

def main():
    # ==========================================
    # 1. 配置 (Config)
    # ==========================================
    # 球员检测模型 (Model B)
    PLAYER_MODEL_PATH = "assets/weights/yolov8m.pt" 
    # 球场关键点模型 (Model A - 表弟训练的那个 V4 模型)
    PITCH_MODEL_PATH  = "assets/weights/best.pt"   
    
    SOURCE_VIDEO_PATH = "assets/samples/InterGoalClip.mp4"
    TARGET_VIDEO_PATH = "assets/output/Final_Result_V4.mp4"
    PITCH_IMAGE_PATH  = "assets/pitch_bg.png"

    # ==========================================
    # 2. 初始化 (Init)
    # ==========================================
    print(f"🚀 初始化 Tactix V4 全自动引擎...")

    # A. 视觉感知 (双模型驱动)
    # 找人模型
    detector = Detector(model_weights=PLAYER_MODEL_PATH, device='mps', conf_threshold=0.3)
    # 找场模型 (新增)
    pitch_estimator = PitchEstimator(model_path=PITCH_MODEL_PATH, device='mps')
    
    tracker = Tracker()

    # B. 几何引擎
    # V4 不需要初始化点，它会等待第一帧的预测结果
    view_transformer = ViewTransformer() 

    # C. 语义与战术
    team_classifier = TeamClassifier(device='cpu')
    classifier_trained = False
    pass_net = PassNetwork(max_pass_dist=400, ball_owner_dist=60)

    # D. 渲染器
    minimap_renderer = MinimapRenderer(bg_image_path=PITCH_IMAGE_PATH)
    
    # 绘图工具
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.4)
    ball_annotator = sv.DotAnnotator(color=sv.Color.WHITE, radius=5)

    # 视频流设置
    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

    # ==========================================
    # 3. 主循环 (Main Loop)
    # ==========================================
    print(f"▶️ 开始 V4 推理处理...")
    
    with sv.VideoSink(TARGET_VIDEO_PATH, video_info=video_info) as sink:
        for i, frame in tqdm(enumerate(frame_generator), total=video_info.total_frames):
            
            # ------------------------------------------------------
            # [Step 1] 🌍 球场感知与校准 (The World)
            # ------------------------------------------------------
            # 1.1 运行 Pitch Model，找 27 个关键点
            kpts_xy, kpts_conf = pitch_estimator.predict(frame)
            
            calibration_success = False
            if kpts_xy is not None:
                # 1.2 自动更新单应性矩阵 (RANSAC)
                # 只要这里返回 True，说明矩阵算出来了
                calibration_success = view_transformer.update_from_model(kpts_xy, kpts_conf)

            # ------------------------------------------------------
            # [Step 2] 👥 球员检测与跟踪 (The Entities)
            # ------------------------------------------------------
            frame_data = detector.detect(frame, frame_index=i)
            
            if len(frame_data.players) > 0:
                xyxy = np.array([p.rect for p in frame_data.players])
                class_ids = np.array([p.class_id for p in frame_data.players])
                # 构造 supervision 对象
                detections_sv = sv.Detections(
                    xyxy=xyxy, 
                    confidence=np.array([0.8]*len(xyxy)), 
                    class_id=class_ids
                )
                tracker.update(detections_sv, frame_data)

            # ------------------------------------------------------
            # [Step 3] 👕 球队分类 (Team Color)
            # ------------------------------------------------------
            valid_players = [p for p in frame_data.players if p.team == TeamID.UNKNOWN]
            
            # 前几帧积累数据训练
            if not classifier_trained and len(valid_players) > 3 and i < 30:
                team_classifier.fit(frame, frame_data.players)
                if i > 10: classifier_trained = True # 简单粗暴，10帧后就当训练好了
            
            # 预测
            if classifier_trained:
                team_classifier.predict(frame, frame_data)

            # ------------------------------------------------------
            # [Step 4] 📍 坐标映射 (Pixel -> Meter -> Tactic Board)
            # ------------------------------------------------------
            # 只有当 Pitch 校准成功时，才进行映射
            if calibration_success:
                view_transformer.transform_players(frame_data.players)
                if frame_data.ball:
                    ball_pos = view_transformer.transform_point(frame_data.ball.center)
                    if ball_pos:
                        frame_data.ball.pitch_position = Point(x=ball_pos[0], y=ball_pos[1])

            # ------------------------------------------------------
            # [Step 5] 🧠 战术分析
            # ------------------------------------------------------
            pass_lines = pass_net.analyze(frame_data)

            # ------------------------------------------------------
            # [Step 6] 🎨 渲染合成 (Rendering)
            # ------------------------------------------------------
            annotated_frame = frame.copy()

            # 6.1 [调试] 画出球场关键点 (证明 V4 模型在工作)
            if kpts_xy is not None:
                for idx, (x, y) in enumerate(kpts_xy):
                    conf = kpts_conf[idx]
                    if conf > 0.5: # 只画可信的点
                        # 画个青色小圆点
                        cv2.circle(annotated_frame, (int(x), int(y)), 4, (255, 255, 0), -1)
                        # (可选) 画 ID 看看是哪个点
                        # cv2.putText(annotated_frame, str(idx), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

            # 6.2 画传球线
            for start, end, opacity in pass_lines:
                overlay = annotated_frame.copy()
                cv2.line(overlay, start, end, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.addWeighted(overlay, opacity, annotated_frame, 1 - opacity, 0, annotated_frame)

            # 6.3 画球员框
            if len(frame_data.players) > 0:
                # 颜色逻辑
                xyxy = np.array([p.rect for p in frame_data.players])
                color_indices = []
                labels = []
                for p in frame_data.players:
                    idx = 4 # 灰色(未知)
                    lbl = f"#{p.id}"
                    if p.team == TeamID.A: idx = 0       # 红
                    elif p.team == TeamID.B: idx = 1     # 蓝
                    elif p.team == TeamID.REFEREE: idx = 2; lbl = "Ref"
                    elif p.team == TeamID.GOALKEEPER: idx = 3; lbl = "GK"
                    color_indices.append(idx)
                    labels.append(lbl)
                
                det_viz = sv.Detections(xyxy=xyxy, class_id=np.array(color_indices))
                
                palette = sv.ColorPalette(colors=[
                    sv.Color(230, 57, 70),   # A队: 红
                    sv.Color(69, 123, 157),  # B队: 蓝
                    sv.Color(255, 255, 0),   # 裁判: 黄
                    sv.Color(0, 0, 0),       # 门将: 黑
                    sv.Color(128, 128, 128)  # 未知: 灰
                ])
                box_annotator.color = palette
                label_annotator.color = palette
                
                annotated_frame = box_annotator.annotate(annotated_frame, det_viz)
                annotated_frame = label_annotator.annotate(annotated_frame, det_viz, labels=labels)

            # 6.4 画球
            if frame_data.ball:
                ball_det = sv.Detections(xyxy=np.array([frame_data.ball.rect]), class_id=np.array([0]))
                annotated_frame = ball_annotator.annotate(annotated_frame, ball_det)

            # 6.5 画小地图 (如果校准成功)
            if calibration_success:
                minimap_img = minimap_renderer.draw(frame_data)
                
                # 贴图逻辑
                target_w = 320
                scale = target_w / minimap_img.shape[1]
                target_h = int(minimap_img.shape[0] * scale)
                minimap_small = cv2.resize(minimap_img, (target_w, target_h))
                
                x_off, y_off = 30, 30
                # 边界检查
                if y_off + target_h < annotated_frame.shape[0] and x_off + target_w < annotated_frame.shape[1]:
                    # 加个白边框
                    annotated_frame[y_off-2:y_off+target_h+2, x_off-2:x_off+target_w+2] = (255,255,255)
                    annotated_frame[y_off:y_off+target_h, x_off:x_off+target_w] = minimap_small
            else:
                # 如果这一帧没算出来矩阵，在左上角写个警告
                cv2.putText(annotated_frame, "Searching for Pitch...", (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            sink.write_frame(annotated_frame)

    print(f"\n✅ V4 处理完成! 结果保存在: {TARGET_VIDEO_PATH}")

if __name__ == "__main__":
    main()