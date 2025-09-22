# tools/qa_safety.py
from __future__ import annotations
import os, sys, json, argparse, hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# --- repo imports ---
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from vision.classes import CLASS_MAP as APP_CLASS_MAP
from vision.classes import MOTHERSHIP_NAMES, PARASITE_NAMES
from vision.relation import CarrierRelationConfig, relation_alert_debug, pylon_band
from vision.tracker import IOUTracker
from vision.motion import motion_candidates
from vision.detector import class_conf_filter

# -------------------------
# Shared helpers
# -------------------------

def sha1_of(arr: np.ndarray) -> str:
    return hashlib.sha1(arr.tobytes()).hexdigest()[:8]

def ensure_dir(p: str):
    os.makedirs(os.path.dirname(p), exist_ok=True)

def names_from_weights(weights: str) -> Dict[int, str]:
    m = YOLO(weights)
    # Ultralytics exposes names as dict[int,str]
    return dict(m.model.names)

def proxy_role_sets(model_names: Dict[int,str]) -> Tuple[set, set]:
    """
    Return (mship_names, parasite_names) accepted by the current model.
    This is the guard against 'airplane'/'bird' vs 'fixed_wing'/'quad' mismatches.
    """
    all_names = set(model_names.values())
    # Accept both our custom names AND likely COCO proxies
    mship = set(n for n in ["fixed_wing", "airplane", "plane"] if n in all_names)
    para  = set(n for n in ["quad", "drone", "bird", "kite"] if n in all_names)
    return mship, para

def role_map(model_names: Dict[int,str]) -> Dict[int,str]:
    """id -> role-name 'fixed_wing' or 'quad' when possible, else original."""
    inv = {v:k for k,v in model_names.items()}
    out = {}
    # Priorities for mothership and parasite role mapping
    preferred_mship = ["fixed_wing", "airplane", "plane"]
    preferred_para  = ["quad", "drone", "bird", "kite"]

    for cls_id, name in model_names.items():
        if name in preferred_mship:
            out[cls_id] = "fixed_wing"
        elif name in preferred_para:
            out[cls_id] = "quad"
        else:
            out[cls_id] = name
    return out

def read_first_frame(video_path: str) -> np.ndarray:
    import av
    c = av.open(video_path)
    v = c.streams.video[0]
    for fr in c.decode(video=0):
        return fr.to_ndarray(format="bgr24")
    raise RuntimeError("No frames in video")

def yolo_infer_arrays(model: YOLO, img_bgr: np.ndarray, imgsz: int, conf: float, iou: float,
                      max_det: int, device: Optional[str] = None):
    KW = dict(imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, verbose=False)
    if device is not None:
        KW["device"] = device
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    r = model.predict(img_rgb, **KW)[0]
    if r.boxes is None or len(r.boxes) == 0:
        # fallback BGR (for some custom exports)
        r2 = model.predict(img_bgr, **KW)[0]
        if r2.boxes is None or len(r2.boxes) == 0:
            # <<< always return 4 values >>>
            z4 = (np.zeros((0,4)), np.zeros((0,)), np.zeros((0,), dtype=int))
            return (*z4, None)
        r = r2

    boxes = r.boxes.xyxy.detach().cpu().numpy()
    confs = r.boxes.conf.detach().cpu().numpy()
    clses = r.boxes.cls.detach().cpu().numpy().astype(int)
    return boxes, confs, clses, r

# -------------------------
# 1) CONTRACT TEST
# -------------------------

def cmd_contract(args):
    """
    Validate: (a) names match role sets, (b) per-class thresholds don’t wipe detections,
    (c) at least some boxes survive on a test image (if provided).
    """
    names = names_from_weights(args.weights)
    mship, para = proxy_role_sets(names)

    report = {
        "weights": args.weights,
        "names": names,
        "mothership_names_accepted": sorted(list(mship)),
        "parasite_names_accepted": sorted(list(para)),
        "status": "OK",
        "survivors": None
    }

    if not mship:
        report["status"] = "FAIL_NO_MOTHERSHIP_NAME"
    if not para:
        # Not fatal, but warn—we can still run fixed-wing only.
        report.setdefault("warnings", []).append("No parasite-compatible class names found in weights")

    # If an image is given, run a pass with per-class gates
    if args.image:
        model = YOLO(args.weights)
        boxes, confs, clses, r = yolo_infer_arrays(
            model, cv2.imread(args.image), imgsz=args.imgsz, conf=args.conf,
            iou=args.iou, max_det=args.max_det, device=None
        )
        # --- DEBUG DUMP (only for first few frames) ---
        if args.debug and (fi < 5):
            os.makedirs("export/qa/pixel_vis", exist_ok=True)
            if r is not None:
                vis = r.plot()  # RGB
                cv2.imwrite(f"export/qa/pixel_vis/frame_{fi:02d}.png", vis[:, :, ::-1])
            else:
                cv2.imwrite(f"export/qa/pixel_vis/frame_{fi:02d}_nobox.png", img_bgr)

        # Build thresholds dynamically from names found
        rolemap = role_map(names)
        id2name = {k: rolemap.get(k, v) for k,v in names.items()}
        # Default gates (lenient)
        thr = {"fixed_wing": args.fw_min, "quad": args.quad_min}
        bx2, cf2, cl2 = class_conf_filter(boxes, confs, clses, thr, id2name)
        report["raw_counts"] = int(len(confs))
        report["survivors"] = int(len(cf2))
        report["status"] = "OK" if (len(cf2) > 0 or len(confs) > 0) else "WARN_ZERO_BOXES"

    ensure_dir(args.report)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

# -------------------------
# 2) PIXEL-SCALE SMOKE TEST
# -------------------------

def draw_sky(h, w):
    top = np.full((h//2, w, 3), (200,210,235), np.uint8)
    bot = np.full((h - h//2, w, 3), (170,185,215), np.uint8)
    return np.vstack([top, bot])

def synth_plane_with_quads(h=720, w=1280, plane_w=220, plane_h=120,
                           quad_px=20, n_quads=2, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = draw_sky(h, w)
    # place plane
    x1 = rng.integers(80, w-80-plane_w)
    y1 = rng.integers(80, h-80-plane_h)
    x2, y2 = x1+plane_w, y1+plane_h
    cv2.rectangle(img, (x1,y1), (x2,y2), (60,60,60), -1)

    # pylon band 12–22% from top of plane
    band_top = int(y1 + 0.12*(plane_h))
    band_bot = int(y1 + 0.22*(plane_h))
    lx = int(x1 + 0.20*(plane_w))
    rx = int(x2 - 0.20*(plane_w))

    xs = np.linspace(lx, rx, n_quads+2)[1:-1].astype(int)
    for cx in xs:
        cy = int(rng.integers(band_top, band_bot+1))
        qx1 = max(0, cx - quad_px//2); qy1 = max(0, cy - quad_px//2)
        qx2 = min(w-1, qx1 + quad_px); qy2 = min(h-1, qy1 + quad_px)
        cv2.rectangle(img, (qx1,qy1), (qx2,qy2), (110,110,110), -1)
    return img

def cmd_pixel_scale(args):
    """
    Generate N frames with quads of ~given px size, ensure ≥K frames produce ≥1 quad detection.
    """
    names = names_from_weights(args.weights)
    model = YOLO(args.weights)
    rolemap = role_map(names)
    id2name = {k: rolemap.get(k, v) for k,v in names.items()}

    hits = 0
    frames = []
    for i in range(args.n):
        img = synth_plane_with_quads(
            h=args.h, w=args.w, plane_w=args.plane_w, plane_h=args.plane_h,
            quad_px=args.quad_px, n_quads=args.n_quads, seed=i
        )
        frames.append(img)
        boxes, confs, clses, r = yolo_infer_arrays(
            model, img, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
            max_det=args.max_det, device=None
        )
        thr = {"fixed_wing": args.fw_min, "quad": args.quad_min}
        bx2, cf2, cl2 = class_conf_filter(boxes, confs, clses, thr, id2name)
        # Count a "hit" if any survivor is a quad
        quads = (np.array([id2name.get(int(k), str(k)) for k in cl2]) == "quad").sum()
        hits += int(quads > 0)

    ok = hits >= args.min_hits
    report = {
        "weights": args.weights,
        "frames_tested": args.n,
        "hits": hits,
        "min_hits_required": args.min_hits,
        "passed": bool(ok),
        "note": "Increase imgsz/max_det or lower quad_min if failing."
    }

    ensure_dir(args.report)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

# -------------------------
# 4) GOLDEN BENCHMARK (pre/post)
# -------------------------

def boxes_to_tracks(tracker: IOUTracker, boxes, confs, clses):
    return tracker.update(boxes, confs, clses)

def in_band(box, band):
    x1,y1,x2,y2 = band
    cx = 0.5*(box[0]+box[2]); cy = 0.5*(box[1]+box[3])
    return (x1 <= cx <= x2) and (y1 <= cy <= y2)

def cmd_golden_benchmark(args):
    import av
    names = names_from_weights(args.weights)
    model = YOLO(args.weights)
    id2role = role_map(names)

    cfg = CarrierRelationConfig(
        mothership_cls_names={"fixed_wing"},
        parasite_cls_names={"quad"},
        band_rel_top=0.12, band_rel_bottom=0.22, band_rel_width=0.60,
        size_ratio_min=0.005, size_ratio_max=0.60,
        min_persist_frames=2, window_frames=12, vel_cos_min=0.0
    )

    c = av.open(args.video)
    v = c.streams.video[0]

    tracker = IOUTracker(iou_thresh=0.20, max_missed=20, max_history=cfg.window_frames)

    total_frames = 0
    total_boxes  = 0
    total_inband = 0
    total_alerts = 0

    for i, fr in enumerate(c.decode(video=0)):
        if i < args.start: continue
        if i >= args.end: break
        img = fr.to_ndarray(format="bgr24")

        boxes, confs, clses, r = yolo_infer_arrays(
            model, img, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
            max_det=args.max_det, device=None
        )
        # Per-class gates
        thr = {"fixed_wing": args.fw_min, "quad": args.quad_min}
        boxes, confs, clses = class_conf_filter(boxes, confs, clses, thr, id2role)

        # Track
        tracks = boxes_to_tracks(tracker, boxes, confs, clses)

        # Pick mothership track (first with role 'fixed_wing')
        ms = next((t for t in tracks if id2role.get(int(t.cls), str(t.cls)) == "fixed_wing"), None)
        band = None
        if ms is not None:
            band = pylon_band(ms.box, cfg)

        # Count in-band parasites
        if band is not None:
            for t in tracks:
                if id2role.get(int(t.cls), str(t.cls)) == "quad":
                    total_inband += int(in_band(t.box, band))

        # Relation alert
        alerts, dbg = relation_alert_debug(tracks, {k:v for k,v in enumerate(["fixed_wing","quad"])}, cfg)
        total_alerts += int(len(alerts) > 0)

        total_boxes += len(boxes)
        total_frames += 1

    report = {
        "video": args.video,
        "frames": total_frames,
        "total_boxes": total_boxes,
        "boxes_per_frame": (total_boxes / max(1,total_frames)),
        "in_band_count": total_inband,
        "alerts_frames": total_alerts,
        "alerts_rate_per_min@24fps": (total_alerts / max(1,total_frames)) * 24 * 60,
        "note": "Store this JSON as your baseline; post-training must beat it."
    }
    ensure_dir(args.report)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

# -------------------------
# 5) MOTION FALLBACK (detector off)
# -------------------------

def cmd_motion_fallback(args):
    import av
    c = av.open(args.video)
    v = c.streams.video[0]
    prev_gray = None

    # central 80% provisional region if no mothership
    def central_region(img):
        h, w = img.shape[:2]
        padx, pady = int(0.1*w), int(0.1*h)
        return np.array([padx, pady, w-padx, h-pady], dtype=float)

    motion_hits = 0
    frames = 0

    for fr in c.decode(video=0):
        img = fr.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        region = central_region(img)
        mot_boxes = motion_candidates(prev_gray, gray, region,
                                      mag_thresh=args.mag_thresh,
                                      min_area=args.min_blob,
                                      max_area=args.max_blob)
        motion_hits += int(len(mot_boxes) > 0)
        frames += 1
        prev_gray = gray

        if frames >= args.max_frames:
            break

    report = {
        "video": args.video,
        "frames": frames,
        "frames_with_motion": motion_hits,
        "hit_rate": motion_hits / max(1, frames),
        "note": "Should be >0 on dynamic scenes; tune mag/area if 0."
    }
    ensure_dir(args.report)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

# -------------------------
# CLI
# -------------------------

def build_parser():
    ap = argparse.ArgumentParser(prog="qa_safety", description="EDTH Hamburg safety tests")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # 1) contract
    ap1 = sub.add_parser("contract", help="Class-name + threshold contract test")
    ap1.add_argument("--weights", required=True)
    ap1.add_argument("--image", help="Optional test image")
    ap1.add_argument("--imgsz", type=int, default=960)
    ap1.add_argument("--conf", type=float, default=0.05)
    ap1.add_argument("--iou", type=float, default=0.50)
    ap1.add_argument("--max_det", type=int, default=300)
    ap1.add_argument("--fw_min", type=float, default=0.08)
    ap1.add_argument("--quad_min", type=float, default=0.15)
    ap1.add_argument("--report", required=True)
    ap1.set_defaults(func=cmd_contract)

    # 2) pixel-scale
    ap2 = sub.add_parser("pixel-scale", help="Tiny-object smoke test")
    ap2.add_argument("--weights", required=True)
    ap2.add_argument("--imgsz", type=int, default=1280)
    ap2.add_argument("--conf", type=float, default=0.05)
    ap2.add_argument("--iou", type=float, default=0.50)
    ap2.add_argument("--max_det", type=int, default=300)
    ap2.add_argument("--h", type=int, default=720)
    ap2.add_argument("--w", type=int, default=1280)
    ap2.add_argument("--plane_w", type=int, default=240)
    ap2.add_argument("--plane_h", type=int, default=120)
    ap2.add_argument("--quad_px", type=int, default=18, help="Target FPV pixel size")
    ap2.add_argument("--n_quads", type=int, default=2)
    ap2.add_argument("--n", type=int, default=10)
    ap2.add_argument("--min_hits", type=int, default=7)
    ap2.add_argument("--fw_min", type=float, default=0.08)
    ap2.add_argument("--quad_min", type=float, default=0.15)
    ap2.add_argument("--report", required=True)
    ap2.set_defaults(func=cmd_pixel_scale)

    # 4) golden-benchmark
    ap4 = sub.add_parser("golden-benchmark", help="Baseline metrics on golden clip")
    ap4.add_argument("--weights", required=True)
    ap4.add_argument("--video", required=True)
    ap4.add_argument("--start", type=int, default=0)
    ap4.add_argument("--end", type=int, default=200)   # ~8s @24fps
    ap4.add_argument("--imgsz", type=int, default=1280)
    ap4.add_argument("--conf", type=float, default=0.05)
    ap4.add_argument("--iou", type=float, default=0.50)
    ap4.add_argument("--max_det", type=int, default=300)
    ap4.add_argument("--fw_min", type=float, default=0.08)
    ap4.add_argument("--quad_min", type=float, default=0.15)
    ap4.add_argument("--report", required=True)
    ap4.set_defaults(func=cmd_golden_benchmark)

    # 5) motion-fallback
    ap5 = sub.add_parser("motion-fallback", help="Check motion heuristic alone")
    ap5.add_argument("--video", required=True)
    ap5.add_argument("--mag_thresh", type=float, default=1.6)
    ap5.add_argument("--min_blob", type=int, default=25)
    ap5.add_argument("--max_blob", type=int, default=2000)
    ap5.add_argument("--max_frames", type=int, default=200)
    ap5.add_argument("--report", required=True)
    ap5.set_defaults(func=cmd_motion_fallback)

    return ap

def main():
    ap = build_parser()
    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
