from __future__ import annotations
import numpy as np
from typing import List, Tuple, Dict

# Relation logic: detect "mothership (fixed-wing proxy) with parasites (small drones) under wings"
# Works over *current-frame detections + last known track states*.

def center(box: np.ndarray):
    x1,y1,x2,y2 = box
    return (0.5*(x1+x2), 0.5*(y1+y2))

def velocity(track) -> np.ndarray:
    if len(track.history) < 3:
        return np.array([0.0, 0.0])
    x1,y1,_,_ = track.history[-3]
    x2,y2,_,_ = track.history[-1]
    return np.array([x2 - x1, y2 - y1])

def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a)+1e-6, np.linalg.norm(b)+1e-6
    return float(np.dot(a, b) / (na*nb))

class CarrierRelationConfig:
    def __init__(
        self,
        mothership_cls_names=("airplane","fixed_wing"),
        parasite_cls_names=("bird","drone","quadcopter"),
        band_rel_top=0.15, band_rel_bottom=0.35, band_rel_width=0.7,
        size_ratio_min=0.01,    # from 0.05 → 0.01
        size_ratio_max=0.5,     # from 0.18 → 0.5
        min_persist_frames=2,   # from 8 → 2
        vel_cos_min=0.0,        # from 0.8 → 0.0
        window_frames=12,
    ):
        self.mship_names = set(mothership_cls_names)
        self.parasite_names = set(parasite_cls_names)
        self.band_rel_top = band_rel_top
        self.band_rel_bottom = band_rel_bottom
        self.band_rel_width = band_rel_width
        self.size_ratio_min = size_ratio_min
        self.size_ratio_max = size_ratio_max
        self.min_persist_frames = min_persist_frames
        self.window_frames = window_frames
        self.vel_cos_min = vel_cos_min

def clsname(id_: int, class_map: Dict[int,str]) -> str:
    return class_map.get(id_, str(id_))

def box_area(b):
    x1,y1,x2,y2 = b
    return max(0.0, x2-x1) * max(0.0, y2-y1)

def pylon_band(mbox: np.ndarray, cfg: CarrierRelationConfig):
    x1,y1,x2,y2 = mbox
    w = x2 - x1
    h = y2 - y1
    cx, cy = 0.5*(x1+x2), 0.5*(y1+y2)
    band_top = cy + cfg.band_rel_top * h
    band_bot = cy + cfg.band_rel_bottom * h
    half_w = 0.5 * cfg.band_rel_width * w
    return np.array([cx - half_w, band_top, cx + half_w, band_bot])

def is_in_band(box: np.ndarray, band: np.ndarray) -> bool:
    bx, by = center(box)
    x1,y1,x2,y2 = band
    return (x1 <= bx <= x2) and (y1 <= by <= y2)

def relation_alert(tracks, class_map: Dict[int,str], cfg: CarrierRelationConfig):
    alerts = []
    # Extract candidate motherships and parasites from tracks
    mships = [t for t in tracks if clsname(t.cls, class_map) in cfg.mship_names]
    candidates = [t for t in tracks if clsname(t.cls, class_map) in cfg.parasite_names or True]  # permissive; filtered by size/band
    for ms in mships:
        band = pylon_band(ms.box, cfg)
        ms_v = velocity(ms)
        ms_area = box_area(ms.box)
        # Find parasite candidates meeting size, band, and velocity-correlation
        parasites = []
        for t in candidates:
            if t.tid == ms.tid:
                continue
            pr_area = box_area(t.box)
            if pr_area <= 0: 
                continue
            ratio = pr_area / (ms_area + 1e-6)
            if not (cfg.size_ratio_min <= ratio <= cfg.size_ratio_max):
                continue
            if not is_in_band(t.box, band):
                continue
            v = velocity(t)
            if cos_sim(ms_v, v) < cfg.vel_cos_min:
                continue
            # Temporal persistence check
            persist = min(len(t.history), cfg.window_frames)
            # crude proxy: require last persist frames to exist
            if persist < cfg.min_persist_frames:
                continue
            parasites.append(t)

        # Simple symmetry heuristic (left/right roughly around center)
        lefts, rights = 0, 0
        cx, _ = center(ms.box)
        for p in parasites:
            px, _ = center(p.box)
            if px < cx: lefts += 1
            else: rights += 1

        if len(parasites) >= 1 and (lefts >= 1 and rights >= 1 or len(parasites) >= 2):
            alerts.append({
                "mothership_tid": ms.tid,
                "parasite_tids": [p.tid for p in parasites],
                "band": band.tolist(),
                "parasite_count": len(parasites),
                "note": f"carrier_with_parasites persisted >= {cfg.min_persist_frames}/{cfg.window_frames} frames"
            })
    return alerts
