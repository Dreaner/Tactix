from ultralytics import YOLO
import torch

def main():
    # 检查 CUDA 是否可用
    if torch.cuda.is_available():
        print(f"🔥 GPU 就绪: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ 警告: 未检测到 GPU，请检查 PyTorch 安装！")

    # 1. 加载中等模型 (v8m-pose)
    # 这比 Nano 版精度更高，4090 跑起来毫无压力
    model = YOLO('yolov8m-pose.pt') 

    print("🚀 开始榨干 4090 性能...")

    # 2. 开始训练
    model.train(
        data='football-pitch.yaml', # 配置文件路径
        epochs=100,                 # 跑 100 轮，效果拉满
        imgsz=640,
        batch=64,                   # 4090 显存大，直接给 64 或 128
        device=0,                   # 强制使用第一块 NVIDIA 显卡
        workers=8,                  # 多线程加载数据
        project='runs/pitch_calibration',
        name='v8m_4090_result',     # 结果文件夹名字
        exist_ok=True,
        plots=True
    )
    
    print("✅ 训练完成！请把 runs/pitch_calibration/v8m_4090_result/weights/best.pt 发回给表哥。")

if __name__ == "__main__":
    main()