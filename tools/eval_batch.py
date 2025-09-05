import os, json, time, argparse
from glob import glob
import av, cv2
import numpy as np
from vision.detector import YoloTinyDetector, DEFAULT_CLASS_MAP
from vision.tracker import IOUTracker
from vision.relation import CarrierRelationConfig, relation_alert

def eval_dir(dirpath: str, conf: float, size_ratio_min: float, size_ratio_max: float, persist: int):
    detector = YoloTinyDetector(conf=conf)
    tracker = IOUTracker()
    cfg = CarrierRelationConfig(size_ratio_min=size_ratio_min, size_ratio_max=size_ratio_max, min_persist_frames=persist)
    alerts, frames = 0, 0
    for mp4 in glob(os.path.join(dirpath, "*.mp4")):
        container = av.open(mp4)
        video = container.streams.video[0]
        for frame in container.decode(video=0):
            img = frame.to_ndarray(format="bgr24")
            boxes, confs, clses = detector.infer_bgr(img)
            tracks = tracker.update(boxes, confs, clses)
            al = relation_alert(tracks, DEFAULT_CLASS_MAP, cfg)
            alerts += int(len(al) > 0)
            frames += 1
    return {"alerts": alerts, "frames": frames, "alerts_per_min@24fps": (alerts / max(1, frames)) * 24 * 60}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()
    res = eval_dir(args.clips, conf=0.25, size_ratio_min=0.05, size_ratio_max=0.18, persist=8)
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
