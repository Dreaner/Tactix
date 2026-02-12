"""
Project: Tactix
File Created: 2026-02-07 18:00:00
Author: Xingnan Zhu
File Name: train_detector.py
Description:
    Standard script to train a YOLO object detection model (Players/Ball).
    Usage: python training/train/train_detector.py
"""

import argparse
import os
import shutil
from ultralytics import YOLO

def get_weights_path(model_name):
    """
    Resolves the path to the model weights.
    Checks 'assets/weights/' first. If not found, returns the name directly
    (so YOLO will download it to the current dir, which we can then move).
    """
    # Assuming script is run from project root or training/train/
    # We try to find assets/weights relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../../"))
    weights_dir = os.path.join(project_root, "assets/weights")
    
    # Ensure weights dir exists
    os.makedirs(weights_dir, exist_ok=True)
    
    target_path = os.path.join(weights_dir, model_name)
    
    if os.path.exists(target_path):
        return target_path
    
    # If not found locally, return name so YOLO downloads it.
    # Note: YOLO downloads to current working directory.
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
        patience=50, # Early stopping
        save=True,
        exist_ok=True, # Overwrite existing experiment
        plots=True     # Save plots
    )
    
    print(f"✅ Training completed. Best model saved at: {results.save_dir}")
    
    # 3. Optional: Move downloaded base weights to assets/weights if they were downloaded
    if not os.path.isabs(model_path) and os.path.exists(model_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, "../../"))
        dest = os.path.join(project_root, "assets/weights", model_path)
        print(f"📦 Moving downloaded base weights to {dest}...")
        shutil.move(model_path, dest)

if __name__ == "__main__":
    # Default configuration for Football Player Detection
    
    # Use the dataset's built-in data.yaml (relative to project root)
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.abspath(os.path.join(_script_dir, "../../"))
    dataset_config_path = os.path.join(_project_root, "datasets/football-players-detection.v20-rf-detr-m.yolo26/data.yaml")
    
    parser = argparse.ArgumentParser(description="Train YOLO Detector")
    
    # Set defaults to your specific task
    parser.add_argument("--data", type=str, default=dataset_config_path, help="Path to dataset YAML")
    # Updated to yolo26x.pt as requested
    parser.add_argument("--model", type=str, default="yolo26x.pt", help="Base model weights")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    # Reduced batch size for Large model on Mac
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, cuda, mps)")
    parser.add_argument("--project", type=str, default="runs/train", help="Save results to project/name")
    parser.add_argument("--name", type=str, default="football_v20_yolo26x", help="Experiment name")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data):
        print(f"❌ Error: Config file not found at {args.data}")
        exit(1)

    train(args)
