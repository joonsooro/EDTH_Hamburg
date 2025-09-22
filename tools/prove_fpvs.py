# tools/prove_fpvs.py
import json, os, sys
import av, cv2
import numpy as np
from ultralytics import YOLO

# Fixed "golden" profile — same across app & tests
IMGSZ = 1536
MAX_DET = 1000
CONF = 0.02
IOU = 0.50
AGNOSTIC = False
FW_MIN, QUAD_MIN = 0.05, 0.08  # per-class gates

CLASS_MAP = {0: "fixed_wing", 1: "quad"}

def class_conf_filter(boxes, confs, clses):
    if len(confs) == 0: return boxes, confs, clses
    thr = np.array([FW_MIN if CLASS_MAP.get(int(k))=="fixed_wing"
                    else QUAD_MIN if CLASS_MAP.get(int(k))=="quad"
                    else 0.0 for k in clses], dtype=float)
    keep = confs >= thr
    return boxes[keep], confs[keep], clses[keep]

def run(video_path, weights="models/best.pt", frames=24, out_dir="export/prove_fpvs"):
    os.makedirs(out_dir, exist_ok=True)
    m = YOLO(weights)

    kw = dict(conf=CONF, iou=IOU, imgsz=IMGSZ, max_det=MAX_DET,
              agnostic_nms=AGNOSTIC, verbose=False)

    container = av.open(video_path)
    stream = container.streams.video[0]

    stats = {"weights": weights, "video": video_path,
             "imgsz": IMGSZ, "max_det": MAX_DET, "conf": CONF,
             "iou": IOU, "agnostic": AGNOSTIC,
             "frames_tested": 0, "quad_hits": 0,
             "per_frame": []}

    for fi, frame in enumerate(container.decode(video=0)):
        if fi >= frames: break
        bgr = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        r = m.predict(rgb, **kw)[0]
        if r.boxes is None:
            boxes = np.zeros((0,4)); confs = np.zeros((0,)); clses = np.zeros((0,), int)
        else:
            boxes = r.boxes.xyxy.detach().cpu().numpy()
            confs = r.boxes.conf.detach().cpu().numpy()
            clses = r.boxes.cls.detach().cpu().numpy().astype(int)

        boxes, confs, clses = class_conf_filter(boxes, confs, clses)

        # summarize
        quads = (clses == 1)
        stats["frames_tested"] += 1
        stats["quad_hits"] += int(quads.any())
        q_sizes = []
        for (x1,y1,x2,y2), is_q in zip(boxes, quads):
            if is_q:
                q_sizes.append(int(min(x2-x1, y2-y1)))
        stats["per_frame"].append({
            "frame": fi,
            "quads": int(quads.sum()),
            "quad_sizes_px": q_sizes[:8]
        })

        # dump a quick visual for the 1st few frames
        if fi < 6:
            vis = r.plot()  # Ultralytics’ own rendering (RGB)
            cv2.imwrite(os.path.join(out_dir, f"prove_{fi:03d}.png"), vis[:, :, ::-1])

    stats["passed"] = (stats["quad_hits"] >= max(3, stats["frames_tested"]//4))
    with open(os.path.join(out_dir, "prove_fpvs.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/prove_fpvs.py /path/to/golden.mp4 [weights]")
        sys.exit(1)
    video = sys.argv[1]
    weights = sys.argv[2] if len(sys.argv) > 2 else "models/best.pt"
    run(video, weights)
