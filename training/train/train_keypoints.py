"""
Project: Tactix
File Created: 2026-02-07 18:00:00
Author: Xingnan Zhu
File Name: train_keypoints.py
Description:
    Standard script to train a YOLOv8-Pose model (Pitch Keypoints).
    Usage: python training/train/train_keypoints.py --data training/configs/pitch_keypoints.yaml
"""

import argparse
from ultralytics import YOLO

def train(args):
    # 1. Load a pose model
    # Note: Must use a -pose model weight (e.g., yolov8n-pose.pt)
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
        pose=True, # Enable pose mode (though YOLO usually infers this from weights)
        patience=50,
        save=True,
        exist_ok=True
    )
    
    print(f"✅ Training completed. Best model saved at: {results.save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 Pose Estimator")
    parser.add_argument("--data", type=str, default="pitch-pose.yaml", help="Path to dataset YAML")
    parser.add_argument("--model", type=str, default="yolov8n-pose.pt", help="Base model weights")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="mps", help="Device (cpu, cuda, mps)")
    parser.add_argument("--project", type=str, default="runs/pose", help="Save results to project/name")
    parser.add_argument("--name", type=str, default="exp", help="Experiment name")
    
    args = parser.parse_args()
    train(args)
