from __future__ import annotations
import numpy as np
from typing import List, Tuple

# Minimal IoU tracker (stable and dependency-light). Good enough for Hamburg demo.
# Tracks with linear assignment + IoU, no Kalman for simplicity. Extend if needed.

def iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return inter / union

class Track:
    def __init__(self, tid: int, box: np.ndarray, cls: int, conf: float):
        self.tid = tid
        self.box = box.astype(float)
        self.cls = cls
        self.conf = conf
        self.age = 0
        self.missed = 0
        self.history = [box.astype(float)]

    def update(self, box: np.ndarray, conf: float):
        self.box = 0.8 * self.box + 0.2 * box.astype(float)  # light smoothing
        self.conf = conf
        self.age += 1
        self.missed = 0
        self.history.append(self.box.copy())

class IOUTracker:
    def __init__(self, iou_thresh: float = 0.3, max_missed: int = 10):
        self.iou_thresh = iou_thresh
        self.max_missed = max_missed
        self.next_id = 1
        self.tracks: List[Track] = []

    def update(self, det_boxes: np.ndarray, det_confs: np.ndarray, det_clses: np.ndarray):
        assigned = set()
        # Match existing tracks to detections by IoU (greedy)
        for tr in self.tracks:
            tr.age += 1
            tr.missed += 1
            best_idx, best_iou = -1, 0.0
            for i, db in enumerate(det_boxes):
                if i in assigned: 
                    continue
                iouv = iou(tr.box, db)
                if iouv > best_iou:
                    best_iou, best_idx = iouv, i
            if best_idx >= 0 and best_iou >= self.iou_thresh:
                tr.update(det_boxes[best_idx], float(det_confs[best_idx]))
                tr.cls = int(det_clses[best_idx])
                assigned.add(best_idx)
        # New tracks for unmatched detections
        for i, db in enumerate(det_boxes):
            if i not in assigned:
                self.tracks.append(Track(self.next_id, db, int(det_clses[i]), float(det_confs[i])))
                self.next_id += 1
        # Prune stale
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        return self.tracks
