#!/usr/bin/env python3
"""
infer.py
========
Run inference using the hybrid YOLOv8 + DINOv2 object detection model.
Loads validation sample images, processes them through the hybrid network,
and visualizes the detections with bounding boxes and labels.

Reference:
IISc I3D Lab - Intelligent Inclusive Interaction Design Lab
https://cambum.net/I3DLab/AI4DM.htm
"""

import sys
import glob
from pathlib import Path
import cv2
import torch

# Prioritize local custom ultralytics
SCRIPT_DIR = Path(__file__).resolve().parent
ULTRALYTICS_DIR = SCRIPT_DIR / "ultralytics"
if str(ULTRALYTICS_DIR) not in sys.path:
    sys.path.insert(0, str(ULTRALYTICS_DIR))

# Ensure torch hub dinov2 repo is in sys.path for checkpoint unpickling
import torch
import torch.hub
hub_dir = Path(torch.hub.get_dir())
for dinov2_dir in hub_dir.glob("*dinov2*"):
    if dinov2_dir.is_dir() and str(dinov2_dir) not in sys.path:
        sys.path.insert(0, str(dinov2_dir))

from ultralytics import YOLO


def run_inference(
    weights_or_yaml=SCRIPT_DIR / "yolov8-dino.yaml",
    source=SCRIPT_DIR / "datasets/coco8/images/val",
    output_dir=SCRIPT_DIR / "output_predictions",
    conf_thresh=0.25,
    imgsz=640,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(" YOLOv8 + DINOv2 Object Detection Inference ")
    print("=" * 65)
    print(f"Model / Weights : {weights_or_yaml}")
    print(f"Input Source    : {source}")
    print(f"Output Directory: {output_dir}")
    print(f"Confidence      : {conf_thresh}")
    print("=" * 65)

    # 1. Load model
    model = YOLO(str(weights_or_yaml))

    # 2. Gather image paths
    source_path = Path(source)
    if source_path.is_dir():
        image_files = sorted(
            [f for f in source_path.glob("*.*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        )
    elif source_path.is_file():
        image_files = [source_path]
    else:
        raise FileNotFoundError(f"Source path not found: {source_path}")

    print(f"\nProcessing {len(image_files)} image(s)...")

    # 3. Predict on each image
    for idx, img_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] Inferring on: {img_path.name}")
        results = model.predict(
            source=str(img_path),
            imgsz=imgsz,
            conf=conf_thresh,
            save=False,
            verbose=True,
        )

        res = results[0]
        orig_img = res.orig_img.copy()

        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else []
        confs = res.boxes.conf.cpu().numpy() if res.boxes is not None else []
        classes = res.boxes.cls.cpu().numpy() if res.boxes is not None else []
        names = res.names

        print(f"  --> Detected {len(boxes)} object(s)")

        # Draw annotations
        for box, score, cls_id in zip(boxes, confs, classes):
            x1, y1, x2, y2 = map(int, box)
            label_name = names.get(int(cls_id), str(int(cls_id)))
            caption = f"{label_name} {score:.2f}"

            # Draw box
            cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label banner
            (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(orig_img, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), (0, 255, 0), -1)
            cv2.putText(
                orig_img,
                caption,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        # Save result
        out_file = output_dir / f"pred_{img_path.name}"
        cv2.imwrite(str(out_file), orig_img)
        print(f"  --> Saved annotation to: {out_file}")

    print(f"\nAll predictions saved in: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference with YOLOv8 + DINOv2")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(SCRIPT_DIR / "yolov8-dino.yaml"),
        help="Path to trained .pt weights or model .yaml",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=str(SCRIPT_DIR / "datasets/coco8/images/val"),
        help="Path to image or image folder",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(SCRIPT_DIR / "output_predictions"),
        help="Output folder for annotated predictions",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    run_inference(
        weights_or_yaml=args.weights,
        source=args.source,
        output_dir=args.output,
        conf_thresh=args.conf,
    )
