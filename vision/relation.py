from __future__ import annotations
import numpy as np
from typing import List, Tuple, Dict
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from vision.classes import MOTHERSHIP_NAMES, PARASITE_NAMES

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
        mothership_cls_names=MOTHERSHIP_NAMES,       # ✅ your trained names
        parasite_cls_names=PARASITE_NAMES,               # ✅ your trained names
        band_rel_top=0.12, band_rel_bottom=0.22, band_rel_width=0.60,
        size_ratio_min=0.015, size_ratio_max=0.50,
        min_persist_frames=3,                      # start easier; tune up later
        vel_cos_min=0.6,                           # start easier; guard below
        window_frames=12,
        min_speed_px=0.5                           # NEW: tiny-speed guard
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
        self.min_speed_px = min_speed_px

def clsname(id_: int, class_map: Dict[int, str]) -> str:
    try:
        k = int(id_)
    except Exception:
        k = int(float(id_))  # last-ditch
    return class_map.get(k, str(k))


def box_area(b):
    x1,y1,x2,y2 = b
    return max(0.0, x2-x1) * max(0.0, y2-y1)

def pylon_band(mbox: np.ndarray, cfg: CarrierRelationConfig):
    x1, y1, x2, y2 = mbox
    w = x2 - x1
    h = y2 - y1
    cx = 0.5 * (x1 + x2)
    # ✅ reference from TOP edge, not box center
    band_top = y1 + cfg.band_rel_top * h
    band_bot = y1 + cfg.band_rel_bottom * h
    half_w = 0.5 * cfg.band_rel_width * w
    return np.array([cx - half_w, band_top, cx + half_w, band_bot], dtype=float)

def is_in_band(box: np.ndarray, band: np.ndarray) -> bool:
    bx, by = center(box)
    x1,y1,x2,y2 = band
    return (x1 <= bx <= x2) and (y1 <= by <= y2)

def relation_alert_debug(tracks, class_map, cfg):
    """
    Returns (alerts, debug) where debug is a dict with:
      - mship_box, band
      - candidates: list of {tid, cls_name, in_band, ratio, persist, vel_cos, reason}
    """
    dbg = {"mship_tid": None, "mship_box": None, "band": None, "candidates": []}
    alerts = []

    # pick mothership by name
    mships = [t for t in tracks if clsname(t.cls, class_map) in cfg.mship_names]
    if not mships:
        dbg["reason"] = "NO_MOTHERSHIP_FOUND"
        return alerts, dbg
    ms = mships[0]
    dbg["mship_tid"] = ms.tid
    dbg["mship_box"] = ms.box.tolist()

    band = pylon_band(ms.box, cfg)
    dbg["band"] = band.tolist()

    ms_v = velocity(ms)
    ms_speed = float(np.linalg.norm(ms_v))
    ms_area = box_area(ms.box)

    pr_cands = [t for t in tracks if clsname(t.cls, class_map) in cfg.parasite_names]
    parasites = []
    for t in pr_cands:
        entry = {"tid": int(t.tid), "cls_name": clsname(t.cls, class_map)}
        # skip self
        if t.tid == ms.tid:
            entry["reason"] = "SELF_SKIP"
            dbg["candidates"].append(entry); continue

        pr_area = box_area(t.box)
        ratio = pr_area / (ms_area + 1e-6) if pr_area > 0 else 0.0
        entry["ratio"] = round(float(ratio), 4)

        # band
        cx = 0.5*(t.box[0]+t.box[2]); cy = 0.5*(t.box[1]+t.box[3])
        in_band = (band[0] <= cx <= band[2]) and (band[1] <= cy <= band[3])
        entry["in_band"] = bool(in_band)

        # persistence
        persist = min(len(t.history), cfg.window_frames)
        entry["persist"] = int(persist)

        # velocity
        v = velocity(t)
        vel_cos = float(cos_sim(ms_v, v)) if (len(t.history)>=3 and len(ms.history)>=3) else None
        entry["vel_cos"] = None if vel_cos is None else round(vel_cos, 3)

        # gating with explicit reason capture
        if pr_area <= 0:
            entry["reason"] = "BAD_AREA"
        elif not (cfg.size_ratio_min <= ratio <= cfg.size_ratio_max):
            entry["reason"] = "RATIO_FAIL"
        elif not in_band:
            entry["reason"] = "OUT_OF_BAND"
        elif persist < cfg.min_persist_frames:
            entry["reason"] = "PERSIST_FAIL"
        elif (vel_cos is not None) and (max(np.linalg.norm(v), np.linalg.norm(ms_v)) >= cfg.min_speed_px) and (vel_cos < cfg.vel_cos_min):
            entry["reason"] = "VEL_COS_FAIL"
        else:
            entry["reason"] = "OK"
            parasites.append(t)

        dbg["candidates"].append(entry)

    # symmetry or ≥2
    if parasites:
        cx_ms, _ = center(ms.box)
        lefts = sum(1 for p in parasites if center(p.box)[0] < cx_ms)
        rights = len(parasites) - lefts
        if (lefts >= 1 and rights >= 1) or len(parasites) >= 2:
            alerts.append({
                "mothership_tid": ms.tid,
                "parasite_tids": [p.tid for p in parasites],
                "band": band.tolist(),
                "parasite_count": len(parasites),
                "note": f"carrier_with_parasites ≥{cfg.min_persist_frames}/{cfg.window_frames}"
            })
        else:
            dbg["reason"] = f"SYMMETRY_FAIL (L={lefts}, R={rights}, N={len(parasites)})"

    if not alerts and "reason" not in dbg:
        dbg["reason"] = "NO_PARASITES_OK"
    return alerts, dbg

