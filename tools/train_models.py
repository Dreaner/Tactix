"""
Project: Tactix
File Created: 2026-02-02 17:29:25
Author: Xingnan Zhu
File Name: train_models.py
Description:
    A utility script for training YOLO models using the Roboflow dataset.
    It handles dataset downloading and initiates the training process on
    Apple Silicon (MPS) or other configured devices.
"""


from roboflow import Roboflow
from ultralytics import YOLO

def main():
    # 1. 下载数据 (这是你从 Roboflow 复制来的部分)
    # 确保你安装了 roboflow: pip install roboflow
    rf = Roboflow(api_key="gh35lz777aARwkML6oLx")
    project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
    version = project.version(20)
    dataset = version.download("yolov8")

    # dataset.location 会告诉你数据下载到哪里去了

    # 2. 开始训练 (利用 M3 Pro 加速)
    # 我们基于 yolov8n.pt (Nano) 微调，因为它最快，适合做实时分析
    # 如果你想要更高精度，可以改成 'yolov8m.pt' 或 'yolov8x.pt'
    print("开始在 M3 Pro (MPS) 上训练...")
    
    model = YOLO('yolov8n.pt') 

    results = model.train(
        data=f"{dataset.location}/data.yaml",  # 指向下载的数据集配置
        epochs=30,           # 训练 30 轮通常就够用了
        imgsz=640,           # 图像大小
        device='mps',        # 关键：强制使用 Apple Silicon 加速
        batch=16,            # M3 Pro 显存够大，可以适当调大
        project="tactix_train", # 训练结果保存目录
        name="football_v1"      # 训练任务名称
    )
    
    print("训练完成！最佳模型保存在 tactix_train/football_v1/weights/best.pt")

if __name__ == "__main__":
    main()