#!/usr/bin/env python3
"""
train.py
========
Train the hybrid YOLOv8 + DINOv2 object detection model.
The model injects global semantic representations from Meta's pre-trained DINOv2
foundation model into the deepest backbone layer of YOLOv8.

Reference:
IISc I3D Lab - Intelligent Inclusive Interaction Design Lab
"YOLOv8 + DINOv2: Injecting Global Semantics for Better Detection"
https://cambum.net/I3DLab/AI4DM.htm
"""

import os
import sys
from pathlib import Path

# Ensure the local custom ultralytics package is prioritized in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
ULTRALYTICS_DIR = SCRIPT_DIR / "ultralytics"
if str(ULTRALYTICS_DIR) not in sys.path:
    sys.path.insert(0, str(ULTRALYTICS_DIR))

import torch
from ultralytics import YOLO


def get_device():
    """Detect the best available compute device (MPS for Apple Silicon, CUDA, or CPU)."""
    if torch.cuda.is_available():
        return 0
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_model(
    config_path=SCRIPT_DIR / "yolov8-dino.yaml",
    data_path=SCRIPT_DIR / "coco8_local.yaml",
    epochs=10,
    imgsz=640,
    batch=4,
    device=None,
    project=SCRIPT_DIR / "runs",
    name="yolov8_dino_experiment",
):
    if device is None:
        device = get_device()

    print("=" * 65)
    print(" Hybrid YOLOv8 + DINOv2 Object Detection Training ")
    print("=" * 65)
    print(f"Architecture Config : {config_path}")
    print(f"Dataset             : {data_path}")
    print(f"Epochs              : {epochs}")
    print(f"Batch Size          : {batch}")
    print(f"Image Size          : {imgsz}")
    print(f"Compute Device      : {device}")
    print("=" * 65)

    # 1. Initialize YOLO with the custom DINOv2 hybrid architecture
    print("\n[1/3] Loading hybrid model architecture...")
    model = YOLO(str(config_path))

    # Display layer summary
    model.info(detailed=False)

    # 2. Train on the selected dataset
    # We freeze layers up to 12 so DINOv2 foundation weights stay frozen
    print("\n[2/3] Starting model training...")
    results = model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(project),
        name=name,
        freeze=12,
        workers=2,
        exist_ok=True,
    )

    print("\n[3/3] Training complete!")
    print(f"Results saved to: {Path(project) / name}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train YOLOv8 + DINOv2 model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=2, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--device", type=str, default=None, help="Device ('mps', 'cuda', 'cpu')")
    args = parser.parse_args()

    train_model(epochs=args.epochs, batch=args.batch, imgsz=args.imgsz, device=args.device)
