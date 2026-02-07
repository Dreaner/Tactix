"""
Project: Tactix
File Created: 2026-02-07 18:00:00
Author: Xingnan Zhu
File Name: train_detector.py
Description:
    Standard script to train a YOLOv8 object detection model (Players/Ball).
    Usage: python training/train/train_detector.py --datasets training/configs/ball_player_detector.yaml
"""

import argparse
from ultralytics import YOLO

def train(args):
    # 1. Load a model
    # 'yolov8n.pt' for nano (fastest), 'yolov8x.pt' for extra large (most accurate)
    model = YOLO(args.model)  

    # 2. Train the model
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=50, # Early stopping
        save=True,
        exist_ok=True # Overwrite existing experiment
    )
    
    print(f"✅ Training completed. Best model saved at: {results.save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 Detector")
    parser.add_argument("--datasets", type=str, default="football.yaml", help="Path to dataset YAML")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model weights")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="mps", help="Device (cpu, cuda, mps)")
    parser.add_argument("--project", type=str, default="runs/train", help="Save results to project/name")
    parser.add_argument("--name", type=str, default="exp", help="Experiment name")
    
    args = parser.parse_args()
    train(args)
