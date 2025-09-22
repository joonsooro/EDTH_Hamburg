from __future__ import annotations
import numpy as np
from collections import deque
from typing import List

def _valid_box(b) -> bool:
    if b is None: return False
    b = np.asarray(b, dtype=float).reshape(-1)
    if b.size != 4: return False
    x1,y1,x2,y2 = b
    if not np.isfinite([x1,y1,x2,y2]).all(): return False
    return (x2 > x1) and (y2 > y1)

def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Safe IoU: returns 0.0 on any invalid box."""
    if not (_valid_box(a) and _valid_box(b)):
        return 0.0
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return float(inter / union)

class Track:
    def __init__(self, tid: int, box: np.ndarray, cls: int, conf: float,
                 max_history: int = 12, smooth_alpha: float = 0.2):
        self.tid = tid
        self.box = np.asarray(box, dtype=float).reshape(4)
        self.cls = int(cls)
        self.conf = float(conf)
        self.age = 0
        self.missed = 0
        self.history = deque([self.box.copy()], maxlen=int(max_history))
        self.smooth_alpha = float(smooth_alpha)   # NEW

    def update(self, box: np.ndarray, conf: float):
        box = np.asarray(box, dtype=float).reshape(4)
        a = self.smooth_alpha                        # NEW
        self.box = (1.0 - a) * self.box + a * box    # NEW: parametric smoothing
        self.conf = float(conf)
        self.age += 1
        self.missed = 0
        self.history.append(self.box.copy())

class IOUTracker:
    def __init__(self, iou_thresh: float = 0.3, max_missed: int = 10,
                 max_history: int = 12, smooth_alpha: float = 0.2):  # NEW
        self.iou_thresh = float(iou_thresh)
        self.max_missed = int(max_missed)
        self.max_history = int(max_history)
        self.smooth_alpha = float(smooth_alpha)      # NEW
        self.next_id = 1
        self.tracks: List[Track] = []

    def update(self, det_boxes: np.ndarray, det_confs: np.ndarray, det_clses: np.ndarray):
        # Normalize detector outputs
        det_boxes = np.asarray(det_boxes, dtype=float).reshape((-1, 4)) if det_boxes is not None else np.zeros((0,4))
        det_confs = np.asarray(det_confs, dtype=float).reshape((-1,))    if det_confs is not None else np.zeros((0,))
        det_clses = np.asarray(det_clses, dtype=int).reshape((-1,))      if det_clses is not None else np.zeros((0,), dtype=int)

        assigned = set()

        # Age and try to match
        for tr in self.tracks:
            tr.age += 1
            tr.missed += 1
            best_idx, best_i = -1, 0.0
            for i, db in enumerate(det_boxes):
                if i in assigned: 
                    continue
                iouv = iou(tr.box, db)          # ✅ always a float
                if iouv > best_i:
                    best_i, best_idx = iouv, i
            if best_idx >= 0 and best_i >= self.iou_thresh:
                tr.update(det_boxes[best_idx], float(det_confs[best_idx]))
                tr.cls = int(det_clses[best_idx])
                assigned.add(best_idx)

        # Create new tracks for unmatched detections (valid boxes only)
        for i, db in enumerate(det_boxes):
            if i not in assigned and _valid_box(db):
                self.tracks.append(
                    Track(self.next_id, db, int(det_clses[i]), float(det_confs[i]),
                          max_history=self.max_history, smooth_alpha=self.smooth_alpha)  # NEW
                )
                self.next_id += 1

        # Prune stale tracks
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        return self.tracks
