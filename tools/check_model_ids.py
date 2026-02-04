# check_model_ids.py
import cv2
from ultralytics import YOLO
import supervision as sv

# 1. 加载模型
model = YOLO("assets/weights/football_pitch.pt")

# 2. 读取视频一帧
video_path = "assets/samples/InterGoalClip.mp4"
generator = sv.get_video_frames_generator(video_path)
frame = next(generator) # 拿第一帧

# 3. 预测
results = model(frame)[0]

if results.keypoints is not None and len(results.keypoints.data) > 0:
    # 取置信度最高的那个球场
    kpts = results.keypoints.data[0].cpu().numpy()
    xy = kpts[:, :2]
    conf = kpts[:, 2]

    # 4. 画图
    for i, (x, y) in enumerate(xy):
        if conf[i] < 0.5: continue # 过滤掉不准的
        
        # 画圈
        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
        # 🔥 写上 ID 数字 (关键步骤!)
        cv2.putText(frame, str(i), (int(x), int(y)-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imwrite("check_ids_result.jpg", frame)
    print("📸 保存成功！请打开 check_ids_result.jpg 查看 ID。")
else:
    print("❌ 这一帧没检测到球场，请换个视频或跳过几帧试试。")