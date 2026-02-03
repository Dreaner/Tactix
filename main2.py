"""
Project: Tactix
File Created: 2026-02-02 11:55:51
Author: Xingnan Zhu (Modified by Assistant)
File Name: main.py
Description: Tactix Main Engine with 2D Minimap Integration
"""

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm

# 引入模块
from tactix.vision.detector import Detector
from tactix.vision.tracker import Tracker
from tactix.semantics.team import TeamClassifier
from tactix.tactics.pass_network import PassNetwork
from tactix.core.types import TeamID, Point
# 新增引入
from tactix.vision.transformer import ViewTransformer
from tactix.core.keypoints import get_target_points

# ==========================================
# 辅助函数: 绘制小地图 (不使用额外帮助函数)
# ==========================================
def draw_minimap(bg_image, players, ball):
    # 拷贝背景
    minimap = bg_image.copy()
    
    # 1. 画球员
    for p in players:
        if p.pitch_position:
            mx, my = int(p.pitch_position.x), int(p.pitch_position.y)
            
            # 直接在这里定义颜色，不依赖外部函数
            color = (200, 200, 200) # 默认灰
            if p.team == TeamID.A:      color = (0, 0, 255)   # Team A -> 红色 (BGR)
            elif p.team == TeamID.B:    color = (255, 0, 0)   # Team B -> 蓝色 (BGR)
            elif p.team == TeamID.GOALKEEPER: color = (0, 255, 255) # 门将 -> 黄色
            elif p.team == TeamID.REFEREE:    color = (0, 0, 0)     # 裁判 -> 黑色
            
            # 画实心点 + 白色描边
            cv2.circle(minimap, (mx, my), 15, color, -1)
            cv2.circle(minimap, (mx, my), 15, (255, 255, 255), 2)

    # 2. 画球
    if ball and ball.pitch_position:
        bx, by = int(ball.pitch_position.x), int(ball.pitch_position.y)
        cv2.circle(minimap, (bx, by), 12, (0, 0, 0), -1)      # 黑色轮廓
        cv2.circle(minimap, (bx, by), 8, (255, 255, 255), -1) # 白色内核

    return minimap

def main():
    # ==========================================
    # 1. 配置路径 (Configuration)
    # ==========================================
    MODEL_PATH = "assets/weights/football_v1.pt" 
    SOURCE_VIDEO_PATH = "assets/samples/InterGoalClip.mp4"
    TARGET_VIDEO_PATH = "assets/output/InterGoalClip_out.mp4"
    PITCH_IMAGE_PATH = "assets/pitch_bg.png" # 战术板背景

    # --- 战术板校准数据 (你之前运行 calibrate.py 得到的数据) ---
    CALIBRATION_SOURCE = np.array([(137, 89), (1126, 87), (1045, 398), (138, 222)])
    CALIBRATION_TARGETS = ['L_PA_TOP_LINE', 'MID_TOP', 'CIRCLE_BOTTOM', 'L_PENALTY_SPOT']

    # ==========================================
    # 2. 初始化核心引擎 (Initialization)
    # ==========================================
    print(f"🚀 初始化 Tactix 引擎 (M3 Pro/MPS)...")
    
    # A. 视觉层
    detector = Detector(model_weights=MODEL_PATH, device='mps', conf_threshold=0.1)
    tracker = Tracker()

    # B. 语义层
    team_classifier = TeamClassifier(device='cpu')
    classifier_trained = False

    # C. 战术层
    pass_net = PassNetwork(max_pass_dist=400, ball_owner_dist=60)

    # D. 2D 映射层 (新增) 
    print("📐 初始化 2D 映射系统...")
    target_points = get_target_points(CALIBRATION_TARGETS)
    view_transformer = ViewTransformer(
        source_points=CALIBRATION_SOURCE, 
        target_points=target_points
    )
    
    # 加载小地图背景
    minimap_bg = cv2.imread(PITCH_IMAGE_PATH)
    if minimap_bg is None:
        print("⚠️ 警告: 找不到战术板图片，使用黑色背景代替")
        minimap_bg = np.zeros((1010, 1559, 3), dtype=np.uint8)

    # E. 可视化工具
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.4, text_padding=3)
    ball_annotator = sv.DotAnnotator(color=sv.Color.WHITE, radius=5)

    # 视频流设置
    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

    # ==========================================
    # 3. 处理循环 (Processing Loop)
    # ==========================================
    print(f"🎥 开始处理视频: {SOURCE_VIDEO_PATH} -> {TARGET_VIDEO_PATH}")

    with sv.VideoSink(TARGET_VIDEO_PATH, video_info=video_info) as sink:
        for i, frame in tqdm(enumerate(frame_generator), total=video_info.total_frames):
            
            # --- [Step 1] 感知: 检测 ---
            frame_data = detector.detect(frame, frame_index=i)

            # --- [Step 2] 感知: 跟踪 ---
            if len(frame_data.players) > 0:
                xyxy = np.array([p.rect for p in frame_data.players])
                class_ids = np.array([p.class_id for p in frame_data.players])
                confidences = np.array([0.8] * len(frame_data.players))

                detections_for_tracker = sv.Detections(
                    xyxy=xyxy,
                    confidence=confidences,
                    class_id=class_ids
                )
                tracker.update(detections_for_tracker, frame_data)

            # --- [Step 3] 语义: 球队分类 ---
            valid_players = [p for p in frame_data.players if p.team == TeamID.UNKNOWN]
            if not classifier_trained and len(valid_players) > 5:
                team_classifier.fit(frame, frame_data.players)
                classifier_trained = True
            
            if classifier_trained:
                team_classifier.predict(frame, frame_data)

            # --- [Step 4] 空间: 透视变换 (新增) ---
            # 计算所有球员在战术板上的位置
            view_transformer.transform_players(frame_data.players)
            
            # 计算球的位置
            if frame_data.ball:
                ball_pos_map = view_transformer.transform_point(frame_data.ball.center)
                if ball_pos_map:
                    frame_data.ball.pitch_position = Point(x=ball_pos_map[0], y=ball_pos_map[1])

            # --- [Step 5] 战术: 传球网络 ---
            pass_lines = pass_net.analyze(frame_data)

            # --- [Step 6] 可视化 ---
            annotated_frame = frame.copy()

            # Layer A: 传球连线
            for start_pt, end_pt, opacity in pass_lines:
                line_color = (255, 255, 0)
                overlay = annotated_frame.copy()
                cv2.line(overlay, start_pt, end_pt, line_color, 2, lineType=cv2.LINE_AA)
                cv2.addWeighted(overlay, opacity, annotated_frame, 1 - opacity, 0, annotated_frame)
                cv2.circle(annotated_frame, end_pt, 4, line_color, -1)

            # Layer B: 持球人高亮
            if frame_data.ball and frame_data.ball.owner_id is not None:
                owner = frame_data.get_player_by_id(frame_data.ball.owner_id)
                if owner:
                    cv2.ellipse(annotated_frame, owner.anchor, (25, 10), 0, 0, 360, (0, 255, 255), 2)

            # Layer C: 球员框和标签 (Supervision)
            if len(frame_data.players) > 0:
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

                detections_viz = sv.Detections(
                    xyxy=xyxy,
                    class_id=np.array(color_indices)
                )
                
                # 定义颜色板 (0:红, 1:蓝, 2:黄, 3:橙, 4:灰)
                # 注意：BoxAnnotator 不需要显式传 custom_color_lookup，它会自动根据 class_id 找 palette
                # 但我们需要确保 custom_color_lookup 是 ColorPalette 对象或者用 class_id 映射
                # 这里为了稳妥，我们使用 sv.BoxAnnotator 默认的颜色映射逻辑
                
                # 重新定义颜色板以确保一致
                colors = sv.ColorPalette(colors=[
                    sv.Color(255, 0, 0),     # 0: Red
                    sv.Color(0, 0, 255),     # 1: Blue
                    sv.Color(255, 255, 0),   # 2: Yellow
                    sv.Color(255, 165, 0),   # 3: Orange
                    sv.Color(128, 128, 128)  # 4: Gray
                ])
                
                # 更新 annotator 的 palette
                box_annotator.color = colors
                label_annotator.color = colors

                annotated_frame = box_annotator.annotate(
                    scene=annotated_frame,
                    detections=detections_viz
                )
                annotated_frame = label_annotator.annotate(
                    scene=annotated_frame,
                    detections=detections_viz,
                    labels=labels
                )

            # Layer D: 足球
            if frame_data.ball:
                ball_xyxy = np.array([frame_data.ball.rect])
                ball_detections = sv.Detections(xyxy=ball_xyxy, class_id=np.array([0]))
                annotated_frame = ball_annotator.annotate(
                    scene=annotated_frame,
                    detections=ball_detections
                )

            # Layer E: 绘制并贴上小地图 (新增功能)
            # 1. 生成完整的小地图
            minimap_img = draw_minimap(minimap_bg, frame_data.players, frame_data.ball)
            
            # 2. 缩放小地图 (比如宽度固定为 350像素)
            target_width = 350
            scale = target_width / minimap_img.shape[1]
            target_height = int(minimap_img.shape[0] * scale)
            minimap_small = cv2.resize(minimap_img, (target_width, target_height))
            
            # 3. 贴到左上角 (带一点半透明背景让它看清楚)
            # 定义位置 (padding 20)
            x_offset, y_offset = 20, 20
            
            # 边界检查
            if y_offset + target_height < annotated_frame.shape[0] and x_offset + target_width < annotated_frame.shape[1]:
                annotated_frame[y_offset:y_offset+target_height, x_offset:x_offset+target_width] = minimap_small
                
                # (可选) 画个边框
                cv2.rectangle(annotated_frame, (x_offset, y_offset), 
                              (x_offset+target_width, y_offset+target_height), (255, 255, 255), 2)

            # 写入保存
            sink.write_frame(annotated_frame)

    print(f"\n✅ 处理完成！结果已保存: {TARGET_VIDEO_PATH}")

if __name__ == "__main__":
    main()