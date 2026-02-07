"""
Project: Tactix
File Created: 2026-02-03 11:32:10
Author: Xingnan Zhu
File Name: train_pitch.py
Description: xxx...
"""

"""
Script: Train YOLOv8-Pose for Football Pitch Calibration
Device: Apple M3 Pro (MPS)
"""
from ultralytics import YOLO

def train():
    # 1. 加载预训练模型 (Pose版本)
    # 第一次运行会自动下载 yolov8n-pose.pt
    model = YOLO('yolov8n-pose.pt') 

    print("🚀 开始在 M3 Pro 上训练球场模型...")

    # 2. 开始训练
    # datasets: 指向我们刚才写的 yaml 配置文件
    # epochs: 训练轮数 (建议 50-100，先跑 50 看看效果)
    # imgsz: 图片大小 (640 是标准，追求精度可以上 1280 但会慢)
    # batch: 根据显存调整，M3 Pro 设为 16 或 32 比较稳妥
    # device: 'mps' 使用 Apple Silicon 加速
    results = model.train(
        data='football-pitch.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        device='mps',
        project='runs/pitch_calibration', # 结果保存路径
        name='v1_n_27pts',                # 实验名称
        plots=True,                       # 自动画出训练曲线
        save=True                         # 保存模型
    )

    print(f"✅ 训练完成！最佳模型保存在: {results.save_dir}/weights/best.pt")

if __name__ == "__main__":
    train()