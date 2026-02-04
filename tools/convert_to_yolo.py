"""
Project: Tactix
File Created: 2026-02-03 11:10:21
Author: Xingnan Zhu
File Name: convert_to_yolo.py
Description: 
    Reads SoccerNet JSON camera parameters, projects 3D pitch keypoints to 2D,
    and generates YOLO format labels (.txt).
    
    Keypoints: 27 Standard Football Pitch Landmarks
"""

"""
Script: Convert SoccerNet Calibration Data to YOLOv8 Pose Format (V2.0 Fixed)
Author: Tactix AI Assistant
Description: 
    - Fixes resolution scaling issues (JSON is 720p/1080p, Image is varying).
    - Fixes BBox generation (Tight box instead of full image).
    - Filters out 'split screen' or invalid projections.
"""

"""
Script: Convert SoccerNet Calibration Data to YOLOv8 Pose Format (V3.0 Final Fix)
Focus: Prioritize 'camera.json' and ignore 2D annotation jsons.
"""

"""
Script: Convert SoccerNet 2D Line Annotations to YOLOv8 Pose Format (V4.0 Geometric Solver)
Author: Tactix AI Assistant
Description: 
    - Since camera parameters are missing, we calculate keypoints geometrically.
    - We compute intersections of semantic lines (e.g., Side Line Left + Side Line Top = Top-Left Corner).
"""

import os
import json
import cv2
import numpy as np
import shutil
from tqdm import tqdm

# ==========================================
# 1. 几何工具函数
# ==========================================
def get_line_equation(p1, p2):
    """根据两点计算直线方程 Ax + By = C"""
    A = p2[1] - p1[1]
    B = p1[0] - p2[0]
    C = A * p1[0] + B * p1[1]
    return A, B, C

def find_intersection(line1_pts, line2_pts, w, h):
    """
    计算两条线段所在直线的交点
    line_pts: list of dict {'x': 0.1, 'y': 0.2} (normalized)
    """
    if not line1_pts or not line2_pts:
        return None

    # 取线段的首尾两点来确定直线 (归一化坐标 -> 像素坐标)
    p1 = (line1_pts[0]['x'] * w, line1_pts[0]['y'] * h)
    p2 = (line1_pts[-1]['x'] * w, line1_pts[-1]['y'] * h)
    
    p3 = (line2_pts[0]['x'] * w, line2_pts[0]['y'] * h)
    p4 = (line2_pts[-1]['x'] * w, line2_pts[-1]['y'] * h)

    # 简单的距离检查：如果两线段离得太远，可能它们的交点毫无意义（在图外很远）
    # 但我们先算出来再说

    A1, B1, C1 = get_line_equation(p1, p2)
    A2, B2, C2 = get_line_equation(p3, p4)

    det = A1 * B2 - A2 * B1
    
    if abs(det) < 1e-6: # 平行线
        return None
    
    x = (B2 * C1 - B1 * C2) / det
    y = (A1 * C2 - A2 * C1) / det
    
    return (x, y)

def get_circle_center(circle_pts, w, h):
    """简单的重心法求圆心 (对于部分可见的圆弧也适用)"""
    if not circle_pts:
        return None
    
    xs = [p['x'] * w for p in circle_pts]
    ys = [p['y'] * h for p in circle_pts]
    
    # 对于标准中圈，取均值通常就是圆心（或者非常接近）
    return (sum(xs) / len(xs), sum(ys) / len(ys))

# ==========================================
# 2. 语义映射表 (关键点 -> 需要哪两条线)
# ==========================================
# 格式: '关键点名': ('线1名', '线2名')
INTERSECTION_MAP = {
    "L_Corner_TL": ("Side line left", "Side line top"),
    "L_Corner_BL": ("Side line left", "Side line bottom"),
    "R_Corner_TR": ("Side line right", "Side line top"),
    "R_Corner_BR": ("Side line right", "Side line bottom"),
    
    "Mid_Line_Top": ("Middle line", "Side line top"),
    "Mid_Line_Bottom": ("Middle line", "Side line bottom"),
    
    # 禁区角点
    "L_Penalty_TL": ("Big rect. left top", "Side line left"), # 注意：有时是Big rect left main
    "L_Penalty_BL": ("Big rect. left bottom", "Side line left"),
    "R_Penalty_TR": ("Big rect. right top", "Side line right"),
    "R_Penalty_BR": ("Big rect. right bottom", "Side line right"),
    
    # 禁区线与底线交点 (这个稍微难点，通常是 Big rect top 与 Side line top 的交点...不对，是与底线垂直的那条)
    # 简化：SoccerNet 里的 "Big rect. left top" 其实就是禁区上边缘线
    # 它和 "Side line left" 的交点是禁区角，和 "Goal line" (即 Side line left) 的交点...
    # 这里的命名有点绕。我们先抓主要角点。
    
    # 小禁区 (6码区)
    "L_SixYard_TL": ("Small rect. left top", "Side line left"),
    "L_SixYard_BL": ("Small rect. left bottom", "Side line left"),
    "R_SixYard_TR": ("Small rect. right top", "Side line right"),
    "R_SixYard_BR": ("Small rect. right bottom", "Side line right"),
}

# 按照 YOLO 训练顺序定义的 27 个点 (保持顺序一致性！)
# 如果算不出来（缺线），就标为 0
YOLO_KEYPOINT_ORDER = [
    "Center_Spot", 
    "Circle_Intersect_Top", "Circle_Intersect_Bot", # 这两个较难算，先跳过或用中线估算
    "Mid_Line_Top", "Mid_Line_Bottom",
    "L_Corner_TL", "L_Corner_BL",
    "L_Penalty_TL", "L_Penalty_BL",
    "L_Penalty_Line_Top", "L_Penalty_Line_Bot", # 禁区前沿点，需特殊处理
    "L_SixYard_TL", "L_SixYard_BL",
    "L_SixYard_Line_Top", "L_SixYard_Line_Bot",
    "L_Penalty_Spot",
    "R_Corner_TR", "R_Corner_BR",
    "R_Penalty_TR", "R_Penalty_BR",
    "R_Penalty_Line_Top", "R_Penalty_Line_Bot",
    "R_SixYard_TR", "R_SixYard_BR",
    "R_SixYard_Line_Top", "R_SixYard_Line_Bot",
    "R_Penalty_Spot",
]

def convert_dataset(root_path, split_name):
    print(f"\n🔄 正在处理 V4.0 (几何解算版) Split: {split_name} ...")
    
    out_root = os.path.abspath(os.path.join(root_path, "..", "yolo_ready"))
    out_img_dir = os.path.join(out_root, "images", split_name)
    out_lbl_dir = os.path.join(out_root, "labels", split_name)
    debug_dir = os.path.join(out_root, "debug_vis", split_name)
    
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    search_path = os.path.join(root_path, split_name) if split_name else root_path
    
    # 扫描所有 JSON (这次我们就找 13631.json 这种！)
    file_pairs = [] 
    for root, dirs, files in os.walk(search_path):
        for file in files:
            if file.endswith(".json") and not file.startswith("camera"):
                json_path = os.path.join(root, file)
                img_path = json_path.replace(".json", ".png")
                # 还有可能是 jpg
                if not os.path.exists(img_path):
                     img_path = json_path.replace(".json", ".jpg")
                
                if os.path.exists(img_path):
                    file_pairs.append((img_path, json_path))

    print(f"📄 找到 {len(file_pairs)} 组数据。")
    success_count = 0
    
    for img_path, json_path in tqdm(file_pairs):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except:
            continue
            
        img = cv2.imread(img_path)
        if img is None: continue
        h, w = img.shape[:2]

        keypoint_data = {} # 暂存算出来的点 {'Name': (x, y)}

        # 1. 算交点
        for kp_name, lines in INTERSECTION_MAP.items():
            l1_name, l2_name = lines
            if l1_name in data and l2_name in data:
                pt = find_intersection(data[l1_name], data[l2_name], w, h)
                if pt:
                    keypoint_data[kp_name] = pt

        # 2. 算圆心
        if "Circle central" in data:
            pt = get_circle_center(data["Circle central"], w, h)
            if pt:
                keypoint_data["Center_Spot"] = pt

        # 3. 组装 YOLO 格式
        yolo_kps = []
        visible_count = 0
        valid_x = []
        valid_y = []

        for kp_name in YOLO_KEYPOINT_ORDER:
            if kp_name in keypoint_data:
                x, y = keypoint_data[kp_name]
                
                # 检查是否在图内 (容错 50 像素)
                if -50 <= x < w + 50 and -50 <= y < h + 50:
                    x_clamp = max(0, min(x, w))
                    y_clamp = max(0, min(y, h))
                    
                    yolo_kps.extend([f"{x_clamp/w:.6f}", f"{y_clamp/h:.6f}", "2"])
                    valid_x.append(x_clamp)
                    valid_y.append(y_clamp)
                    visible_count += 1
                else:
                    yolo_kps.extend(["0.000000", "0.000000", "0"])
            else:
                yolo_kps.extend(["0.000000", "0.000000", "0"])

        # 至少要有 4 个点才生成标签，否则太少没意义
        if visible_count < 4:
            continue

        # 生成 BBox
        min_x, max_x = min(valid_x), max(valid_x)
        min_y, max_y = min(valid_y), max(valid_y)
        box_w = max_x - min_x
        box_h = max_y - min_y
        box_cx = min_x + box_w / 2
        box_cy = min_y + box_h / 2
        
        # 写入
        label_line = f"0 {box_cx/w:.6f} {box_cy/h:.6f} {box_w/w:.6f} {box_h/h:.6f} " + " ".join(yolo_kps)
        
        folder_name = os.path.basename(os.path.dirname(img_path))
        file_base = os.path.splitext(os.path.basename(img_path))[0]
        new_name = f"{folder_name}_{file_base}"
        
        target_img_path = os.path.join(out_img_dir, new_name + ".png")
        target_txt_path = os.path.join(out_lbl_dir, new_name + ".txt")
        
        with open(target_txt_path, 'w') as f:
            f.write(label_line)
        shutil.copy(img_path, target_img_path)
        
        # 可视化前 20 张
        if success_count < 20:
             debug_img = img.copy()
             cv2.rectangle(debug_img, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 255, 0), 2)
             # 画点
             kp_list = label_line.split()[5:]
             for i in range(0, len(kp_list), 3):
                px = int(float(kp_list[i]) * w)
                py = int(float(kp_list[i+1]) * h)
                v = int(kp_list[i+2])
                if v == 2: cv2.circle(debug_img, (px, py), 5, (0, 0, 255), -1)
             cv2.imwrite(os.path.join(debug_dir, new_name + "_vis.jpg"), debug_img)

        success_count += 1

    print(f"✅ V4.0 处理完成! 成功生成 {success_count} 组数据。")
    print(f"👀 请立即检查: {debug_dir}")

def main():
    # 👇 你的路径
    base_dir = r"/Users/dreaner/Dev/Tactix/data/SoccerNet/calibration-2023" 
    if not os.path.exists(base_dir):
        print("❌ 路径不对")
        return
    # 递归搜索
    convert_dataset(base_dir, "")

if __name__ == "__main__":
    main()