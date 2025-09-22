from __future__ import annotations
import sys, os, io, json, hashlib, time
import av
import numpy as np, time, hashlib, cv2
import streamlit as st
from datetime import datetime
from typing import List, Dict
import torch
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vision.classes import CLASS_MAP as DEFAULT_CLASS_MAP
from vision.classes import MOTHERSHIP_NAMES, PARASITE_NAMES
from vision.detector import Detector, class_conf_filter
from vision.tracker import IOUTracker, Track
from vision.relation import CarrierRelationConfig, relation_alert_debug
from vision.motion import motion_candidates
from vision.relation import pylon_band  # ensure it's exported in relation.py
from app.salute_schema import default_salute, Location

st.set_page_config(page_title="EDTH Hamburg ISR Demo", layout="wide")

# Relation config must match trained class names
rel_cfg = CarrierRelationConfig(
    mothership_cls_names=MOTHERSHIP_NAMES,
    parasite_cls_names=PARASITE_NAMES,
    band_rel_top=0.05, band_rel_bottom=0.55, band_rel_width=0.90,
    size_ratio_min=0.005, size_ratio_max=0.60,
    min_persist_frames=2, window_frames=12,
    vel_cos_min=0.0
)

# # Use your trained model (pt) with small-object friendly settings
detector = Detector(
    conf=0.25,          # starting point; you still expose conf_thresh slider
    iou=0.50,
    max_det=100,
    agnostic=True,
    imgsz=960
)

# optional: expose as a slider in the sidebar (0=no update, 1=no smoothing)
smooth_alpha = st.sidebar.slider("Track smoothing (alpha)", 0.0, 1.0, 0.2, 0.05)
tracker = IOUTracker(
    iou_thresh=0.20,
    max_missed=20,
    max_history=rel_cfg.window_frames,
    smooth_alpha=float(smooth_alpha)
)


EXPORT_DIR = "export/incidents"
st.caption(f"class_map={DEFAULT_CLASS_MAP} | mothership={MOTHERSHIP_NAMES} | parasite={PARASITE_NAMES}")

os.makedirs(EXPORT_DIR, exist_ok=True)

st.sidebar.text(f"Model: models/best.pt @ imgsz={detector.imgsz} conf={detector.conf}")
st.sidebar.text(f"Classes: {detector.model.names}")

def tracks_from_synth_labels(json_path, frame_idx):
    """Return a list of Track objects for this frame from synth JSON labels."""
    with open(json_path, "r") as f:
        lbl = json.load(f)
    if frame_idx >= len(lbl["frames"]):
        return []

    fr = lbl["frames"][frame_idx]
    tracks = []

    # Mothership as tid=1
    ms_box = np.array(fr["fixed_wing"], dtype=float)
    ms = Track(tid=1, box=ms_box, cls=0, conf=0.99)  # cls=4 ~ 'airplane' in COCO
    # give it a short history so velocity isn't zero
    ms.history = [ms_box.copy()] * 10
    tracks.append(ms)

    # Parasites start at tid=100
    tid = 100
    for pbox in fr["parasites"]:
        pb = np.array(pbox, dtype=float)
        pt = Track(tid=tid, box=pb, cls=1, conf=0.80)  # cls=14 ~ 'bird' proxy
        pt.history = [pb.copy()] * 10
        tracks.append(pt)
        tid += 1

    return tracks

def rect_halo(mbox, grow=0.35):
    x1,y1,x2,y2 = mbox
    w, h = (x2-x1), (y2-y1)
    cx, cy = 0.5*(x1+x2), 0.5*(y1+y2)
    gw, gh = w*(1.0+grow), h*(1.0+grow)
    return np.array([cx-gw/2, cy-gh/2, cx+gw/2, cy+gh/2], dtype=float)

st.sidebar.title("Controls")
source = st.sidebar.selectbox("Video source", ["Sample clip", "Upload MP4"])
ew_mode = st.sidebar.toggle("EW MODE (simulate link degradation)", value=False)
target_fps = st.sidebar.slider("Target FPS (cap)", 10, 30, 24)
conf_thresh = st.sidebar.slider("Detector conf", 0.1, 0.7, 0.25, 0.05)
st.sidebar.info("Tip: keep 720p sources for stable throughput.")

DEV_MODE = st.sidebar.toggle("Developer mode", value=False)
# These toggles are for debugging mode/force-alert mode that is used for crisis
if DEV_MODE:
    debug_labels = st.sidebar.toggle("Use labeled tracks (debug)", value=False)
    force_alert  = st.sidebar.toggle("Force alert (demo)", value=False)
else:
    debug_labels = False
    force_alert  = False

st.sidebar.text(f"Model: models/best.pt @ imgsz={detector.imgsz} conf={detector.conf}")

st.sidebar.subheader("Heuristic (no-ML) parasites")
use_motion = st.sidebar.toggle("Enable motion-parasite heuristic", value=False)
mag_thresh = st.sidebar.slider("Flow mag threshold", 0.5, 5.0, 1.6, 0.1)
min_blob = st.sidebar.slider("Min blob area", 5, 200, 25, 5)
max_blob = st.sidebar.slider("Max blob area", 200, 5000, 2000, 50)
halo_fallback = st.sidebar.toggle("Use leader halo if no fixed-wing", value=True)
halo_grow = st.sidebar.slider("Leader halo grow", 0.1, 1.0, 0.35, 0.05)
fw_min = st.sidebar.slider("FW conf min",   0.0, 0.9, 0.12, 0.01)
quad_min = st.sidebar.slider("Quad conf min", 0.0, 0.9, 0.20, 0.01)


if source == "Sample clip":
    sample_path = st.sidebar.text_input("Path", "data/golden/sample1.mp4")
else:
    up = st.sidebar.file_uploader("Upload MP4", type=["mp4"])
    if up:
        sample_path = os.path.join("data/golden", f"upload_{int(time.time())}.mp4")
        with open(sample_path, "wb") as f:
            f.write(up.read())
    else:
        sample_path = None

col_live, col_inc, col_sal = st.columns([4,3,3])

with col_live:
    st.header("Live")
    frame_out = st.empty()
    metrics_out = st.empty()
with col_inc:
    st.header("Incidents")
    incidents_list = st.container()
with col_sal:
    st.header("SALUTE")
    salute_json_area = st.empty()
    salute_btn = st.button("Generate SALUTE from last incident")

# state
if "last_incident" not in st.session_state:
    st.session_state.last_incident = None
if "incidents" not in st.session_state:
    st.session_state.incidents = []

def draw_boxes(image, tracks, alerts):
    img = image.copy()
    # draw tracks
    for t in tracks:
        name = DEFAULT_CLASS_MAP.get(int(t.cls), str(int(t.cls)))
        color = (255,128,0) if name == "fixed_wing" else (0,255,255)
        x1,y1,x2,y2 = [int(v) for v in t.box]
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
        cv2.putText(img, f"{name} id{t.tid}", (x1, y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # draw alert band(s)
    for al in (alerts or []):
        if not isinstance(al, dict) or "band" not in al:
            continue
        band = al["band"]                     # <-- missing line caused the crash
        if band is None:
            continue
        band = list(band)
        if len(band) != 4:
            continue

        x1,y1,x2,y2 = [int(v) for v in band]
        label = f"Carrier+{al.get('parasite_count', '?')}"
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,165,255), 2)
        cv2.putText(img, label, (x1, max(0, y1-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)

    return img


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def save_incident(frame_bgr: np.ndarray, alerts: List[Dict]) -> Dict:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    png_path = os.path.join(EXPORT_DIR, f"incident_{ts}.png")
    cv2.imwrite(png_path, frame_bgr)
    # JSON minimal
    meta = {
        "time_utc": datetime.utcnow().isoformat() + "Z",
        "alerts": alerts,
        "png": png_path,
    }
    json_path = os.path.join(EXPORT_DIR, f"incident_{ts}.json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
    return {"png": png_path, "json": json_path}

def render_salute(incident: Dict):
    loc = Location(lat=None, lon=None, source="dead_reckon", confidence_radius_m=50.0)
    size = "1 fixed-wing + ≥1 small drones"
    activity = "possible strike prep; FPVs under-wing pattern"
    sal = default_salute(size=size, activity=activity, loc=loc, equipment=["fixed_wing","fpv_drone"], notes="carrier_with_parasites demo")
    # attach evidence
    sal.evidence.frame_png_uri = incident["png"]
    # optional: compute sha256 over both json+png
    sh = sha256_file(incident["png"])
    sal.evidence.sha256 = sh
    salute_json_area.code(sal.model_dump_json(indent=2), language="json")
    # persist JSON
    out_json = incident["json"].replace(".json", ".salute.json")
    with open(out_json, "w") as f:
        f.write(sal.model_dump_json(indent=2))
    st.success(f"SALUTE exported: {out_json}")

def _show_count(tag, r):
    n = 0 if (r is None or r.boxes is None) else len(r.boxes)
    st.caption(f"[{tag}] boxes={n}")
    return n

def _to_numpy(x):
    return x.cpu().numpy() if torch.is_tensor(x) else x

# MAIN LOOP (single pass over file; for real live, replace with camera)
# MAIN LOOP (single pass over file; for real live, replace with camera)
run_clicked = st.sidebar.button("Run")
if run_clicked and sample_path and os.path.exists(sample_path):
    container = av.open(sample_path)
    video = container.streams.video[0]
    frame_count = 0
    last_alert_t = 0.0
    t_start = time.perf_counter()

    prev_gray = None
    detector.conf = float(conf_thresh)
    
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format="bgr24")
        # ---- NORMAL DETECTION (always) ----
        boxes, confs, clses = detector.infer_bgr(img)

        # Per-class gates (tunable). For synth w/ GT labels we won't use them.
        THRESH = {"fixed_wing": float(fw_min), "quad": float(quad_min)}
        boxes, confs, clses = class_conf_filter(boxes, confs, clses, THRESH, DEFAULT_CLASS_MAP)


        # === STEP 7: A/B sanity check (use SAME thresholds as Detector) ===
        if DEV_MODE and not st.session_state.get("ab_done", False):
            st.session_state.ab_done = True
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            r = detector.model.predict(
                img_rgb,
                conf=detector.conf,
                iou=detector.iou,
                max_det=detector.max_det,
                agnostic_nms=detector.agnostic,
                imgsz=detector.imgsz,
                verbose=False,
                device=detector.device
            )[0]
            st.caption(f"A/B check: r.boxes={0 if r.boxes is None else len(r.boxes)}")
            st.image(r.plot(), caption="Ultralytics plot() with app thresholds", use_column_width=True)
            st.session_state.ab_done = True
        # === END STEP 7 ===


        # optional downscale to 720p height
        # scale = 720.0 / img.shape[0]
        # if abs(scale - 1.0) > 1e-3:
        #     img = cv2.resize(img, (int(img.shape[1]*scale), 720), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        boxes = np.zeros((0, 4), dtype=float)
        confs = np.zeros((0,), dtype=float)
        clses = np.zeros((0,), dtype=int)

        # detection / tracks
        if debug_labels and "synth_" in os.path.basename(sample_path):
            json_path = sample_path.replace(".mp4", ".json")  # synth_gen writes the json next to mp4
            tracks = tracks_from_synth_labels(json_path, frame_count)  # <-- frame_count now advances
        else:
            h,w = img.shape[:2]
            st.caption(f"frame {frame_count} | shape={img.shape} dtype={img.dtype} rng={np.min(img)}..{np.max(img)}")
            # optional: hash a downsample to prove frames change
            hsh = hashlib.sha1(cv2.resize(img, (min(64,w), min(64,h))).tobytes()).hexdigest()[:8]
            st.caption(f"frame hash={hsh}")

            if DEV_MODE and (frame_count % 10 == 0):
                # ======== HARD DIAGNOSTIC: bypass wrapper & probe three ways ========

                # 0) Basic sanity on the frame
                st.caption(f"[Sanity] img shape={img.shape} dtype={img.dtype} rng={int(img.min())}..{int(img.max())}")
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_rgb = Image.fromarray(img_rgb)

                # Common kwargs (super permissive)
                KW = dict(
                    conf=detector.conf,
                    iou=detector.iou,
                    imgsz=detector.imgsz,
                    max_det=detector.max_det,
                    agnostic_nms=detector.agnostic,
                    verbose=False
                )

                # 1) Force CPU (avoids quirky MPS/CUDA issues)
                r_cpu = detector.model.predict(img_rgb, device="cpu", half=False, **KW)[0]
                n1 = _show_count("predict(cpu)", r_cpu)

                # 2) Same on default device (whatever Ultralytics picked)
                r_def = detector.model.predict(img_rgb, **KW)[0]
                n2 = _show_count("predict(default)", r_def)

                # 3) PIL input on CPU (exercises a different preprocessing path)
                r_pil = detector.model.predict(pil_rgb, device="cpu", half=False, **KW)[0]
                n3 = _show_count("predict(PIL+cpu)", r_pil)

                # Pick the first non-empty result to visualize/extract
                Rs = [("cpu", r_cpu), ("def", r_def), ("pil", r_pil)]
                tag, r_best = next(((t, r) for t, r in Rs if r is not None and r.boxes is not None and len(r.boxes) > 0), (None, None))

                if r_best is not None:
                    # Show Ultralytics’ own rendering so we know it sees boxes
                    vis = r_best.plot()  # expects RGB
                    st.image(vis, caption=f"Ultralytics plot() [{tag}] — should show boxes", use_column_width=True)

                    if r_best is not None and (len(confs) == 0):   # <-- only if normal path was empty
                        boxes = _to_numpy(r_best.boxes.xyxy)
                        confs = _to_numpy(r_best.boxes.conf)
                        clses = _to_numpy(r_best.boxes.cls).astype(int)
                        boxes, confs, clses = class_conf_filter(boxes, confs, clses, THRESH, DEFAULT_CLASS_MAP)
                        st.info("DEV: using r_best because normal path was empty.")

                    # (optional) quick peek at top scores after filtering
                    topk = min(5, len(confs))
                    if topk > 0:
                        idx = confs.argsort()[-topk:][::-1]
                        sample = [(int(clses[i]), float(confs[i])) for i in idx]
                        st.caption(f"[{tag}] top5 confs (cls, conf): {sample}")
                else:
                    st.error("No detections from any probe (cpu/default/PIL). Check model/weights/classes.")
                    boxes = np.zeros((0,4), dtype=float)
                    confs = np.zeros((0,), dtype=float)
                    clses = np.zeros((0,), dtype=int)

                # ======== END HARD DIAGNOSTIC ========
            else:
                # ======== NORMAL INFERENCE ========
                boxes, confs, clses = detector.infer_bgr(img)

                # Apply per-class thresholds to cut noise (esp. leaves, textures)
                boxes, confs, clses = class_conf_filter(
                    boxes, confs, clses, THRESH, DEFAULT_CLASS_MAP
                )

            # --- RAW DETECTION DEBUG (temporary) ---
            if 'show_raw' not in st.session_state: st.session_state.show_raw = True
            if st.session_state.show_raw:
                dbg = img.copy()
                for (x1,y1,x2,y2), c, k in zip(boxes, confs, clses):
                    name = DEFAULT_CLASS_MAP.get(int(k), str(int(k)))
                    color = (0,255,255) if name=='quad' else (255,128,0)
                    cv2.rectangle(dbg, (int(x1),int(y1)), (int(x2),int(y2)), color, 2)
                    cv2.putText(dbg, f"{name} {c:.2f}", (int(x1), max(0,int(y1)-4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                frame_out.image(dbg, channels="BGR", use_column_width=True)
                st.caption(f"RAW: boxes={len(clses)} | quads={(clses==1).sum()} | fw={(clses==0).sum()}")
            # ---------------------------------------
            tracks = tracker.update(boxes, confs, clses)
        
                # --- SINGLE SOURCE OF TRUTH: mothership & region (once per frame) ---
        # pick mothership from DETECTOR tracks only (names come from DEFAULT_CLASS_MAP)
                # --- SINGLE SOURCE OF TRUTH: mothership & region (once per frame) ---
        mship_candidates = [t for t in tracks if DEFAULT_CLASS_MAP.get(int(t.cls)) == "fixed_wing"]
        ms = mship_candidates[0] if mship_candidates else None
        has_mship = ms is not None

        region = None
        region_label = None

        if has_mship:
            band = pylon_band(ms.box, rel_cfg)
            region = band
            region_label = "band"
        elif halo_fallback and tracks:
            leader = max(tracks, key=lambda t: (t.box[2]-t.box[0])*(t.box[3]-t.box[1]))
            region = rect_halo(leader.box, grow=halo_grow)
            region_label = "halo"

        # --------------------------------------------------------------------

        # Motion parasites → pseudo-tracks (use the SAME region)
        motion_tracks = []
        if use_motion and region is not None:
            mot_boxes = motion_candidates(prev_gray, gray, region,
                                          mag_thresh=mag_thresh,
                                          min_area=min_blob, max_area=max_blob)
            tid_base = 5000
            for i, b in enumerate(mot_boxes):
                tt = Track(tid=tid_base+i, box=b, cls=1, conf=0.6, max_history=rel_cfg.window_frames)  # cls=1: 'quad'
                tt.history.append(b.copy())
                motion_tracks.append(tt)

        # Pool used by relation logic
        tracks_for_relation = tracks + motion_tracks

        # --- RELATION DIAGNOSTIC (optional, uses the SAME ms/band) ---
        if has_mship:
            paras = [t for t in tracks_for_relation if DEFAULT_CLASS_MAP.get(int(t.cls)) == "quad"]
            inside, ratios, persists = [], [], []
            def _area(b): return max(1,(b[2]-b[0])) * max(1,(b[3]-b[1]))
            for t in paras:
                cx = 0.5*(t.box[0]+t.box[2]); cy = 0.5*(t.box[1]+t.box[3])
                if band[0] <= cx <= band[2] and band[1] <= cy <= band[3]:
                    inside.append(t)
                    ratios.append(_area(t.box)/_area(ms.box))
                    persists.append(len(t.history))
            st.caption(f"band_in={len(inside)}/{len(paras)} | ratios={[round(r,3) for r in ratios[:4]]} | persist={[min(p,99) for p in persists[:4]]}")
        # ----------------------------------------------------------------------

        # --- RELATION (simple trigger → full rule) ---
        alerts = []
        if has_mship:
            paras = [t for t in tracks_for_relation if DEFAULT_CLASS_MAP.get(int(t.cls)) == "quad"]
            inside = []
            for t in paras:
                cx = 0.5*(t.box[0]+t.box[2]); cy = 0.5*(t.box[1]+t.box[3])
                if band[0] <= cx <= band[2] and band[1] <= cy <= band[3]:
                    inside.append(t)
            if len(inside) >= 1:
                alerts = [{
                    "mothership_tid": int(ms.tid),
                    "parasite_tids": [int(t.tid) for t in inside],
                    "band": [int(x) for x in band],
                    "parasite_count": len(inside),
                    "note": "SIMPLE_BAND_TRIGGER"
                }]

        if not alerts:
            alerts, rdbg = relation_alert_debug(tracks_for_relation, DEFAULT_CLASS_MAP, rel_cfg)
            if rdbg:
                why = rdbg.get("reason", "")
                cand = rdbg.get("candidates", [])
                ok = [c for c in cand if c.get("reason") == "OK"]
                st.caption(f"REL dbg: mship={rdbg.get('mship_tid')} OK={len(ok)}/{len(cand)} reason={why or '—'}")
        # ----------------------------------------------------------------------

        # --- optional force alert (doesn't mutate ms/has_mship) ---
        if force_alert and not alerts:
            ms_force = ms if has_mship else (max(tracks, key=lambda t: (t.box[2]-t.box[0])*(t.box[3]-t.box[1])) if tracks else None)
            if ms_force is not None:
                band_force = pylon_band(ms_force.box, rel_cfg)
                alerts = [{
                    "mothership_tid": int(ms_force.tid),
                    "parasite_tids": [int(t.tid) for t in tracks if t.tid != ms_force.tid][:2],
                    "band": [int(x) for x in band_force],
                    "parasite_count": min(2, max(0, len(tracks)-1)),
                    "note": "FORCED_ALERT_DEMO"
                }]
        # ----------------------------------------------------------------------

        # --- draw & show (ALWAYS) ---
        out = draw_boxes(img, tracks, alerts)
        if region is not None:
            x1,y1,x2,y2 = [int(v) for v in region]
            cv2.rectangle(out, (x1,y1), (x2,y2), (255, 200, 0), 2)
            cv2.putText(out, region_label if region_label else ("band" if has_mship else "halo"),
                        (x1, max(10,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,200,0), 2)
        frame_out.image(out, channels="BGR", use_column_width=True)

        # incidents
        now = time.perf_counter()
        if alerts and (now - last_alert_t > 2.0):
            inc = save_incident(out, alerts)
            st.session_state.incidents.append(inc)
            st.session_state.last_incident = inc
            last_alert_t = now

        # metrics / pacing
        frame_count += 1
        dt = now - t_start
        fps = frame_count / max(1e-6, dt)
        lag_note = "TEXT-ONLY PINGS" if ew_mode else "FULL CLIPS"
        metrics_out.info(f"FPS: {fps:.1f} | Tracks: {len(tracks)} | Alerts total: {len(st.session_state.incidents)} | EW: {lag_note}")

        if ew_mode:
            time.sleep(0.02)

        sleep_budget = max(0.0, (1.0/float(target_fps)) - (time.perf_counter()-now))
        if sleep_budget > 0:
            time.sleep(sleep_budget)
        prev_gray = gray


with incidents_list:
    for i, inc in enumerate(reversed(st.session_state.incidents[-10:])):
        st.image(inc["png"], caption=os.path.basename(inc["png"]), use_column_width=True)

if salute_btn:
    if st.session_state.last_incident:
        render_salute(st.session_state.last_incident)
    else:
        st.warning("No incident yet. Run and wait for an alert.")
