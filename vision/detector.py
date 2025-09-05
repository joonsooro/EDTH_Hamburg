from __future__ import annotations
import torch
import cv2
import numpy as np
from typing import List, Tuple
from ultralytics import YOLO

# NOTE: Keep classes generic. For Hamburg we proxy with common classes.
# Map your chosen weights' class ids to names.
DEFAULT_CLASS_MAP = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus",
    # You will filter to "airplane" as fixed_wing proxy and optionally "bird"/"drone" if your head has it.
}

def get_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"

class YoloTinyDetector:
    def __init__(self, weights: str = "yolov8n.pt", imgsz: int = 720, conf: float = 0.25, iou: float = 0.45):
        self.device = get_device()
        self.model = YOLO(weights)
        self.model.fuse()
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.class_map = DEFAULT_CLASS_MAP

    def infer_bgr(self, img_bgr: np.ndarray):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        res = self.model.predict(img_rgb, imgsz=self.imgsz, device=self.device, conf=self.conf, iou=self.iou, verbose=False)[0]
        # Return xyxy, conf, cls (ints)
        boxes = res.boxes.xyxy.detach().cpu().numpy() if res.boxes.xyxy is not None else np.zeros((0,4))
        confs = res.boxes.conf.detach().cpu().numpy() if res.boxes.conf is not None else np.zeros((0,))
        clses = res.boxes.cls.detach().cpu().numpy().astype(int) if res.boxes.cls is not None else np.zeros((0,), dtype=int)
        return boxes, confs, clses
