# Assignment 1: Hybrid Object Detection with Pre-Trained Foundation Model
**Course**: Intelligent User Interfaces (IUI) / AI for Design & Manufacturing  
**Lab Reference**: IISc Intelligent Inclusive Interaction Design Lab ([I3D Lab](https://cambum.net/I3DLab/AI4DM.htm))  
**Topic**: *YOLOv8 + DINOv2: Injecting Global Semantics for Better Detection*

---

## 1. Overview & Analogy

In computer vision, standard object detectors (like YOLO) face a fundamental trade-off: **speed vs. contextual semantic awareness**.
- **The Fast Detective (YOLO):** Extremely quick. Operates with a local magnifying glass examining edges, textures, and bounding boxes. However, it only learns from labeled boxes and lacks a holistic visual understanding of the scene. As a result, it can mistake unusual shadows, textures, or distant objects for target classes (false positives).
- **The Seasoned Advisor (Pre-trained Foundation Model — DINOv2):** Meta's DINOv2 is a Vision Transformer trained on 142M+ web images using Self-Supervised Learning (SSL). It understands scene structure, depth priors, and semantic relationships without ever seeing a single manual bounding box label.
- **The Collaboration:** DINOv2 takes a single holistic look at the raw input image and produces a rich **Global Scene Vector** (e.g., *"This is an outdoor street scene with vehicles"*). This vector is projected and broadcasted directly into the deepest layer of YOLO before the detection heads make their final predictions. The detector head now has both **local precision** and **global scene awareness**.

---

## 2. Model Architecture & Flow

```
                     Input Raw Image [B, 3, H, W]
                                  │
         ┌────────────────────────┴────────────────────────┐
         │                                                 │
   [Layer 0: ConvDummy]                              [Layer 1..10: YOLO Backbone]
         │ (Bypass)                                        │ (CSP-Darknet + SPPF)
         ▼                                                 ▼
   [Layer 11: DINOv2 (Frozen)]                       [SPPF Spatial Features]
   • Input: [B, 3, 518, 518]                         • Output: [B, 256, 20, 20]
   • ViT-S/14 Embedding: [B, 384]                           │
   • 1x1 Projector + Upsample                              │
   • Global Feature Map: [B, 256, 20, 20]                   │
         │                                                 │
         └────────────────────────┬────────────────────────┘
                                  ▼
                     [Layer 12: Concat Fusion]
                     Output: [B, 512, 20, 20]
                                  │
                                  ▼
                   [YOLO Neck & Detection Heads]
                                  │
                                  ▼
             Final Detections: Bounding Boxes + Classes
```

### Architectural Key Modifications
1. **`ConvDummy` (Layer 0):** A zero-overhead passthrough module allowing the raw image to bypass intermediate convolutions and route directly to DINOv2.
2. **`DINOv2` Module (Layer 11):** Loads frozen `dinov2_vits14` weights, resizes to a patch-divisible grid (518x518), computes a 384-dimensional latent vector, and projects it via a learnable $1\times 1$ convolution into the YOLO channel space.
3. **Global-to-Spatial Broadcast:** The projected vector is spatially interpolated to match the $H/32 \times W/32$ grid of the YOLO P5 layer.
4. **`Concat` Fusion (Layer 12):** Combines spatial detail (from SPPF) and global context (from DINOv2) to condition the multi-scale detection heads.

---

## 3. Inputs and Outputs

* **Inference Input:** Raw RGB image or video frame of dimensions $[B, 3, H, W]$ (default: $640 \times 640$).
* **Intermediate Representation:** 384-dimensional DINOv2 semantic vector projected to $[B, 256, 20, 20]$.
* **Training Input:** Images paired with normalized bounding box annotations:
  $$\text{Target} = [\text{class\_id}, x_{\text{center}}, y_{\text{center}}, \text{width}, \text{height}]$$
* **Final Model Output:**
  * Bounding boxes: $[x_1, y_1, x_2, y_2]$
  * Class label probabilities: $P(C_k)$
  * Confidence scores: $[0.0, 1.0]$

---

## 4. Standard Benchmark Dataset

* **Dataset Selected:** **COCO8** (Standard official 8-image benchmark subset of MS-COCO with 80 object categories).
* **Location:** `datasets/coco8/`
* **Configuration:** `coco8_local.yaml`
* **Splits:**
  * `images/train`: Training images and labels
  * `images/val`: Validation evaluation images and labels

---

## 5. Folder Structure

```
Assignment 1/
├── .venv/                      # Isolated Python 3.14 virtual environment
├── ultralytics/                # Cloned and editable Ultralytics codebase with custom layers
│   └── ultralytics/
│       ├── nn/modules/conv.py  # Injected ConvDummy module
│       ├── nn/modules/block.py # Injected DINOv2 fusion module
│       └── nn/tasks.py         # Registered custom parser hooks
├── datasets/
│   └── coco8/                  # Standard COCO8 dataset
├── yolov8-dino.yaml            # Custom hybrid model architecture definition
├── coco8_local.yaml            # Dataset YAML descriptor
├── explain_utility.py          # Script explaining the utility of DINOv2 pre-trained model
├── train.py                    # Script to train/fine-tune the hybrid detector
├── infer.py                    # Script to run object detection inference
├── run.sh                      # Unified command-line runner
├── utility_explanation.png     # Generated feature comparison plot
└── README.md                   # Assignment documentation
```

---

## 6. How to Run

Activate the environment or use the provided `./run.sh` runner:

### 1. Explain the Utility of the Pre-Trained Model
Extracts the global semantic vector, computes activation statistics, and saves a 3-panel visualization:
```bash
./run.sh explain
```
*(Outputs saved to `utility_explanation.png`)*

### 2. Train the Hybrid Model
Trains the model on the COCO8 dataset while keeping the foundation weights frozen:
```bash
./run.sh train --epochs 5 --batch 2 --imgsz 640
```
*(Checkpoints and training curves saved to `runs/yolov8_dino_experiment/`)*

### 3. Run Inference on Test / Val Images
Runs detections and draws bounding boxes and confidence scores:
```bash
./run.sh infer --conf 0.25
```
*(Annotated images saved to `output_predictions/`)*
