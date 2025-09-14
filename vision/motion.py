# vision/motion.py
from __future__ import annotations
import numpy as np
import cv2

def motion_candidates(prev_gray: np.ndarray | None,
                      gray: np.ndarray | None,
                      region_xyxy: np.ndarray,
                      mag_thresh: float = 1.6,
                      min_area: int = 25,
                      max_area: int = 2000) -> list[np.ndarray]:
    """
    Find small moving blobs inside region using dense optical flow magnitude thresholding.
    Returns a list of [x1,y1,x2,y2] boxes in full-frame coordinates.
    """
    if prev_gray is None or gray is None:
        return []
    x1, y1, x2, y2 = [int(v) for v in region_xyxy]
    if x2 <= x1 or y2 <= y1:
        return []

    roi_prev = prev_gray[y1:y2, x1:x2]
    roi_curr = gray[y1:y2, x1:x2]
    if roi_prev.size == 0 or roi_curr.size == 0:
        return []

    flow = cv2.calcOpticalFlowFarneback(
        roi_prev, roi_curr, None,
        pyr_scale=0.5, levels=2, winsize=15,
        iterations=2, poly_n=5, poly_sigma=1.1, flags=0
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    m = (mag > mag_thresh).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if min_area <= area <= max_area:
            out.append(np.array([x1 + x, y1 + y, x1 + x + w, y1 + y + h], dtype=float))
    return out
    