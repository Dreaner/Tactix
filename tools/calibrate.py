"""
Project: Tactix
File Created: 2026-02-02 23:24:31
Author: Xingnan Zhu
File Name: calibrate.py
Description: xxx...
"""

import cv2
import numpy as np
import sys
import os

# 确保能导入 tactix 模块
sys.path.append(os.getcwd())
from tactix.core.keypoints import KEY_POINTS

# 全局变量
current_click = None # 存储最新点击的坐标

def mouse_callback(event, x, y, flags, param):
    global current_click
    if event == cv2.EVENT_LBUTTONDOWN:
        current_click = (x, y)
        print(f"\n📍 捕获点击: ({x}, {y}) - 请在终端选择对应的点位...")

def print_menu():
    print("\n" + "="*40)
    print("📋 可用关键点列表 (请选择刚才点击的位置):")
    print("="*40)
    
    # 将字典转为列表方便索引
    keys = list(KEY_POINTS.keys())
    
    # 分类打印，方便查找
    categories = {
        "角落": ["CORNER"],
        "左禁区": ["L_PA", "L_GA", "L_PENALTY"],
        "右禁区": ["R_PA", "R_GA", "R_PENALTY"],
        "中场": ["MID", "CENTER", "CIRCLE"]
    }
    
    sorted_keys = []
    index = 0
    
    for cat, filters in categories.items():
        print(f"\n--- {cat} ---")
        for key in keys:
            if any(f in key for f in filters) and key not in sorted_keys:
                print(f"[{index}] {key}")
                sorted_keys.append(key)
                index += 1
    
    return sorted_keys

def main():
    # ⚠️ 修改你的视频路径
    video_path = "../assets/samples/InterGoalClip.mp4"
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print(f"❌ 无法读取视频: {video_path}")
        return

    window_name = "Calibration (Click a point, then check Terminal)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    # 存储最终结果
    collected_points = []   # 像素坐标 [(x,y), ...]
    collected_names = []    # 名字 ['TL_CORNER', ...]

    sorted_keys = print_menu() # 先打印一次菜单供参考
    
    global current_click
    
    print("\n🚀 开始校准！")
    print("步骤 1: 在视频窗口点击一个清晰的关键点。")
    print("步骤 2: 回到终端输入该点的编号。")
    print("我们需要采集 4 个点。")

    while len(collected_points) < 4:
        display_frame = frame.copy()
        
        # 画出已确认的点
        for i, pt in enumerate(collected_points):
            cv2.circle(display_frame, pt, 5, (0, 255, 0), -1)
            cv2.putText(display_frame, f"{collected_names[i]}", (pt[0]+10, pt[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 画出当前临时点击的点
        if current_click:
            cv2.circle(display_frame, current_click, 5, (0, 0, 255), -1)
            cv2.putText(display_frame, "Selected", (current_click[0]+10, current_click[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(100)

        # 核心交互逻辑
        if current_click is not None:
            # 暂停画面更新，等待用户输入
            try:
                # 重新打印菜单，防止刷屏看不到
                print_menu()
                choice = input(f"\n({len(collected_points)+1}/4) 请输入编号 (或按 'q' 重选): ")
                
                if choice.lower() == 'q':
                    print("🔄 取消本次点击，请重新在图片上点击。")
                    current_click = None
                    continue
                
                idx = int(choice)
                if 0 <= idx < len(sorted_keys):
                    key_name = sorted_keys[idx]
                    
                    # 检查是否重复
                    if key_name in collected_names:
                        print(f"⚠️ 警告: {key_name} 已经被选过了！")
                        current_click = None
                        continue

                    print(f"✅ 已绑定: 像素 {current_click} -> {key_name}")
                    
                    collected_points.append(current_click)
                    collected_names.append(key_name)
                    current_click = None # 重置
                else:
                    print("❌ 无效编号，请重试。")
            except ValueError:
                print("❌ 输入错误，请输入数字。")

        if key & 0xFF == ord('q'):
            break

    # 结束
    cv2.destroyAllWindows()
    cap.release()

    if len(collected_points) == 4:
        print("\n" + "="*50)
        print("🎉 校准完成！请直接复制下面的代码到 main.py 的初始化区域：")
        print("="*50)
        
        print("import numpy as np")
        print("from tactix.core.keypoints import get_target_points")
        print("")
        
        # 打印 Source Points
        print("# 1. 视频源坐标")
        print(f"SOURCE_POINTS = np.array({collected_points})")
        print("")
        
        # 打印 Target Keys
        print("# 2. 目标关键点名称")
        print(f"TARGET_KEYS = {collected_names}")
        print("TARGET_POINTS = get_target_points(TARGET_KEYS)")
        print("")
        print("# 3. 初始化")
        print("view_transformer = ViewTransformer(source_points=SOURCE_POINTS, target_points=TARGET_POINTS)")
        print("="*50)

if __name__ == "__main__":
    main()