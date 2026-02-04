# diagnostic.py
import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np

# 1. 加载模型
print("正在加载模型...")
model = YOLO("assets/weights/football_pitch.pt")

# 2. 读取第一帧
video_path = "assets/samples/InterGoalClip.mp4"


# generator = sv.get_video_frames_generator(video_path)
# frame = next(generator) 

# 改成这样，跳过前 150 帧
iterator = sv.get_video_frames_generator(video_path)
for _ in range(150): 
    next(iterator)
frame = next(iterator)

# 3. 预测 (不设任何阈值)
print("正在推理...")
results = model(frame, verbose=False)[0]

# 4. 深度分析
if results.keypoints is not None and len(results.keypoints.data) > 0:
    # 找最自信的那个框
    best_idx = results.boxes.conf.argmax().item()
    kpts = results.keypoints.data[best_idx].cpu().numpy()
    # kpts 结构: [27, 3] -> (x, y, conf)

    print(f"\n====== 🩺 诊断报告 (Frame 0) ======")
    print(f"{'ID':<4} | {'Conf':<6} | {'X':<6} | {'Y':<6} | {'状态'}")
    print("-" * 45)

    for i, (x, y, conf) in enumerate(kpts):
        # 状态标记
        if conf > 0.5: status = "✅ 稳"
        elif conf > 0.1: status = "⚠️ 弱"
        else: status = "❌ 无"
        
        # 只打印有点信息的 (conf > 0.01)
        if conf > 0.01:
            print(f"{i:<4} | {conf:.4f} | {x:.1f} | {y:.1f} | {status}")
            
            # 画图：绿色=稳，黄色=弱，红色=极弱
            if conf > 0.5: color = (0, 255, 0)
            elif conf > 0.1: color = (0, 255, 255)
            else: color = (0, 0, 255)

            cv2.circle(frame, (int(x), int(y)), 5, color, -1)
            # 写上 ID 和 Conf
            label = f"{i} ({conf:.2f})"
            cv2.putText(frame, label, (int(x), int(y)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imwrite("deep_diagnostic.jpg", frame)
    print("\n📸 已保存 deep_diagnostic.jpg，请打开查看！")
    print("====================================")

else:
    print("❌ 这一帧完全没检测到球场！请尝试换个视频或跳过前几帧。")