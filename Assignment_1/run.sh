#!/usr/bin/env bash
# Helper script to activate environment and run Assignment 1 tasks

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/ultralytics:${PYTHONPATH}"

# Detect python in venv
PYTHON="${SCRIPT_DIR}/.venv/bin/python"
if [ ! -f "$PYTHON" ]; then
    PYTHON="python3"
fi

echo "============================================================"
echo " Intelligent User Interfaces (IUI) - Assignment 1"
echo " Hybrid Object Detection: YOLOv8 + DINOv2 Pre-trained Model"
echo "============================================================"
echo "Python Interpreter: $PYTHON"
echo "Working Directory : $SCRIPT_DIR"
echo ""

case "$1" in
    explain)
        echo "[Action] Explaining the Utility of the Pre-trained Model..."
        "$PYTHON" "${SCRIPT_DIR}/explain_utility.py"
        ;;
    train)
        echo "[Action] Training YOLOv8 + DINOv2 on COCO8..."
        shift
        "$PYTHON" "${SCRIPT_DIR}/train.py" "$@"
        ;;
    infer)
        echo "[Action] Running Inference on Validation Set..."
        shift
        "$PYTHON" "${SCRIPT_DIR}/infer.py" "$@"
        ;;
    *)
        echo "Usage: ./run.sh {explain | train | infer} [options]"
        echo ""
        echo "Examples:"
        echo "  ./run.sh explain                   # Extract embeddings and generate explanation plot"
        echo "  ./run.sh train --epochs 5          # Train model on COCO8 benchmark"
        echo "  ./run.sh infer --conf 0.25         # Run inference and draw bounding boxes"
        ;;
esac
