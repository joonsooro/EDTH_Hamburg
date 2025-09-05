from __future__ import annotations
import os, io, json, hashlib, time
import av
import numpy as np
import cv2
import streamlit as st
from datetime import datetime
from typing import List, Dict

from vision.detector import YoloTinyDetector, DEFAULT_CLASS_MAP
from vision.tracker import IOUTracker, Track
from vision.relation import CarrierRelationConfig, relation_alert
from app.salute_schema import default_salute, Location

EXPORT_DIR = "export/incidents"
os.makedirs(EXPORT_DIR, exist_ok=True)

st.set_page_config(page_title="EDTH Hamburg ISR Demo", layout="wide")

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
    ms = Track(tid=1, box=ms_box, cls=4, conf=0.99)  # cls=4 ~ 'airplane' in COCO
    # give it a short history so velocity isn't zero
    ms.history = [ms_box.copy()] * 10
    tracks.append(ms)

    # Parasites start at tid=100
    tid = 100
    for pbox in fr["parasites"]:
        pb = np.array(pbox, dtype=float)
        pt = Track(tid=tid, box=pb, cls=14, conf=0.80)  # cls=14 ~ 'bird' proxy
        pt.history = [pb.copy()] * 10
        tracks.append(pt)
        tid += 1

    return tracks


@st.cache_resource
def load_models():
    detector = YoloTinyDetector(weights="yolov8n.pt", imgsz=720, conf=0.25, iou=0.45)
    tracker = IOUTracker(iou_thresh=0.3, max_missed=10)
    rel_cfg = CarrierRelationConfig(
        mothership_cls_names=("airplane","fixed_wing"),
        parasite_cls_names=("bird","drone","quadcopter"),  # permissive; rule filters by size + band + velocity
        band_rel_top=0.15, band_rel_bottom=0.35, band_rel_width=0.7,
        size_ratio_min=0.05, size_ratio_max=0.18,
        min_persist_frames=8, window_frames=12,
        vel_cos_min=0.8
    )
    return detector, tracker, rel_cfg

detector, tracker, rel_cfg = load_models()

st.sidebar.title("Controls")
DEV_MODE = st.sidebar.toggle("Developer mode", value=False)
# These toggles are for debugging mode/force-alert mode that is used for crisis
if DEV_MODE:
    debug_labels = st.sidebar.toggle("Use labeled tracks (debug)", value=False)
    force_alert  = st.sidebar.toggle("Force alert (demo)", value=False)
else:
    debug_labels = False
    force_alert  = False
source = st.sidebar.selectbox("Video source", ["Sample clip", "Upload MP4"])
ew_mode = st.sidebar.toggle("EW MODE (simulate link degradation)", value=False)
target_fps = st.sidebar.slider("Target FPS (cap)", 10, 30, 24)
conf_thresh = st.sidebar.slider("Detector conf", 0.1, 0.7, 0.25, 0.05)
st.sidebar.info("Tip: keep 720p sources for stable throughput.")

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
    for t in tracks:
        x1,y1,x2,y2 = [int(v) for v in t.box]
        color = (0,255,0)
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
        cv2.putText(img, f"id{t.tid}", (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    for al in alerts:
        x1,y1,x2,y2 = [int(v) for v in al["band"]]
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,165,255), 2)
        cv2.putText(img, f"Carrier+{al['parasite_count']}", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)
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

# MAIN LOOP (single pass over file; for real live, replace with camera)
# MAIN LOOP (single pass over file; for real live, replace with camera)
run_clicked = st.sidebar.button("Run")
if run_clicked and sample_path and os.path.exists(sample_path):
    container = av.open(sample_path)
    video = container.streams.video[0]
    frame_count = 0
    last_alert_t = 0.0
    t_start = time.perf_counter()

    for frame in container.decode(video=0):
        img = frame.to_ndarray(format="bgr24")

        # optional downscale to 720p height
        scale = 720.0 / img.shape[0]
        if abs(scale - 1.0) > 1e-3:
            img = cv2.resize(img, (int(img.shape[1]*scale), 720), interpolation=cv2.INTER_AREA)

        # detection / tracks
        detector.conf = float(conf_thresh)
        if debug_labels and "synth_" in os.path.basename(sample_path):
            json_path = sample_path.replace(".mp4", ".json")  # synth_gen writes the json next to mp4
            tracks = tracks_from_synth_labels(json_path, frame_count)  # <-- frame_count now advances
        else:
            boxes, confs, clses = detector.infer_bgr(img)
            tracks = tracker.update(boxes, confs, clses)

        # relation
        alerts = relation_alert(tracks, DEFAULT_CLASS_MAP, rel_cfg)

        # optional force alert (only override if nothing fired)
        if force_alert and not alerts:
            mships = [t for t in tracks if t.cls == 4]  # COCO 'airplane' proxy
            ms = mships[0] if mships else (max(tracks, key=lambda t: (t.box[2]-t.box[0])*(t.box[3]-t.box[1])) if tracks else None)
            if ms:
                alerts = [{
                    "mothership_tid": ms.tid,
                    "parasite_tids": [t.tid for t in tracks if t.tid != ms.tid][:2],
                    "band": [int(ms.box[0]), int((ms.box[1]+ms.box[3])//2), int(ms.box[2]), int(ms.box[3])],
                    "parasite_count": min(2, max(0, len(tracks)-1)),
                    "note": "FORCED_ALERT_DEMO"
                }]

        # --- draw & show (ALWAYS) ---
        out = draw_boxes(img, tracks, alerts)
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
            time.sleep(0.02)  # ~20ms throttle

        sleep_budget = max(0.0, (1.0/float(target_fps)) - (time.perf_counter()-now))
        if sleep_budget > 0:
            time.sleep(sleep_budget)

with incidents_list:
    for i, inc in enumerate(reversed(st.session_state.incidents[-10:])):
        st.image(inc["png"], caption=os.path.basename(inc["png"]), use_column_width=True)

if salute_btn:
    if st.session_state.last_incident:
        render_salute(st.session_state.last_incident)
    else:
        st.warning("No incident yet. Run and wait for an alert.")
