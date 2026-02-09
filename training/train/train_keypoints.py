"""
Project: Tactix
File Created: 2026-02-07 18:00:00
Author: Xingnan Zhu
File Name: train_keypoints.py
Description:
    Standard script to train a YOLOv8-Pose model (Pitch Keypoints).
    Usage: python training/train/train_keypoints.py
"""

import argparse
import os
import shutil
from ultralytics import YOLO

def get_weights_path(model_name):
    """
    Resolves the path to the model weights.
    Checks 'assets/weights/' first. If not found, returns the name directly.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../../"))
    weights_dir = os.path.join(project_root, "assets/weights")
    
    os.makedirs(weights_dir, exist_ok=True)
    
    target_path = os.path.join(weights_dir, model_name)
    
    if os.path.exists(target_path):
        return target_path
    
    return model_name

def train(args):
    # 1. Resolve model path
    model_path = get_weights_path(args.model)
    print(f"🚀 Loading model: {model_path}")
    
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"❌ Error loading model '{model_path}': {e}")
        return

    # 2. Train the model
    print(f"🏋️ Starting training on {args.device}...")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        pose=True, # Enable pose mode
        patience=50,
        save=True,
        exist_ok=True
    )
    
    print(f"✅ Training completed. Best model saved at: {results.save_dir}")
    
    # 3. Optional: Move downloaded base weights to assets/weights
    if not os.path.isabs(model_path) and os.path.exists(model_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, "../../"))
        dest = os.path.join(project_root, "assets/weights", model_path)
        print(f"📦 Moving downloaded base weights to {dest}...")
        shutil.move(model_path, dest)

if __name__ == "__main__":
    # Default configuration
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "../configs/pitch_kpt.yaml")

    parser = argparse.ArgumentParser(description="Train YOLO Pose Estimator")
    parser.add_argument("--data", type=str, default=config_path, help="Path to dataset YAML")
    parser.add_argument("--model", type=str, default="yolov8n-pose.pt", help="Base model weights")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="mps", help="Device (cpu, cuda, mps)")
    parser.add_argument("--project", type=str, default="runs/pose", help="Save results to project/name")
    parser.add_argument("--name", type=str, default="pitch_pose", help="Experiment name")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data):
        print(f"❌ Error: Config file not found at {args.data}")
        exit(1)

    train(args)
