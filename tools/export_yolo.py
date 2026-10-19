#!/usr/bin/env python3
"""
Export the pretrained COCO YOLO models to ONNX for the three input sizes used in
the experiments. Run ONCE on the PC (needs Python >= 3.8), then copy models/ to the Pi.

YOLOv5n (DEFAULT, arch=5) - the Pi's OpenCV 4.5.5 can run it:
    git clone --depth 1 https://github.com/ultralytics/yolov5
    cd yolov5 && pip install pandas tqdm seaborn gitpython onnxslim
    python export.py --weights yolov5n.pt --include onnx --opset 12 --imgsz 320
    python -c "from onnxslim import slim; slim('yolov5n.onnx', 'models/yolov5n_320.onnx')"
    -> repeat for --imgsz 224 and 160. The slim step is required: recent PyTorch
       writes Floor nodes into the resize layers, which OpenCV 4.5.5 cannot parse.

YOLOv8n (arch=8) - needs OpenCV >= 4.7 (inference fails on the Pi's 4.5.5); this script:
    pip install ultralytics
    python tools/export_yolo.py
    -> models/yolov8n_160.onnx, models/yolov8n_224.onnx, models/yolov8n_320.onnx
"""
import os
import shutil

from ultralytics import YOLO

MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

if __name__ == "__main__":
    os.makedirs(MODELS, exist_ok=True)
    for size in (160, 224, 320):
        exported = YOLO("yolov8n.pt").export(format="onnx", imgsz=size, opset=12, dynamic=False)
        target = os.path.join(MODELS, "yolov8n_%d.onnx" % size)
        shutil.move(str(exported), target)
        print("->", target)
