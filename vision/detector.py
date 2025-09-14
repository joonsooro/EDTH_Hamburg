from __future__ import annotations
import torch
import cv2
import numpy as np
from typing import List, Tuple
from ultralytics import YOLO
# import onnxruntime as ort

# NOTE: Keep classes generic. For Hamburg we proxy with common classes.
# Map your chosen weights' class ids to names.
DEFAULT_CLASS_MAP = {0: "fixed_wing", 1: "quad"}

class Detector:
    def __init__(self, weights="models/best.pt", conf=0.03, imgsz=960):
        self.model = YOLO(weights)
        self.conf  = float(conf)
        self.imgsz = int(imgsz)
        print(f"[Detector] weights={weights} conf={self.conf} imgsz={self.imgsz} names={self.model.names}")

    def infer_bgr(self, img_bgr: np.ndarray):
        assert img_bgr is not None and img_bgr.ndim==3 and img_bgr.dtype==np.uint8, \
            f"bad frame: {None if img_bgr is None else (img_bgr.shape, img_bgr.dtype)}"
        # Try RGB first
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        r = self.model.predict(img_rgb, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            # Fallback: some custom builds expect BGR
            r = self.model.predict(img_bgr, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
            if r.boxes is None or len(r.boxes) == 0:
                return (np.zeros((0,4)), np.zeros((0,)), np.zeros((0,), dtype=int))
            else:
                print("[Detector] Fallback BGR produced boxes")
        return (r.boxes.xyxy.cpu().numpy(),
                r.boxes.conf.cpu().numpy(),
                r.boxes.cls.cpu().numpy().astype(int))

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
