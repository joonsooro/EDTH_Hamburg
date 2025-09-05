import os, argparse, random, json, shutil, subprocess, tempfile
from pathlib import Path
from PIL import Image, ImageFilter
import numpy as np
import cv2

def rand_bg(size=(1280,720)):
    w,h = size
    # simple sky-ish gradient background
    base = np.linspace(
        np.array([180, 200, 230], dtype=np.uint8),
        np.array([160, 180, 210], dtype=np.uint8),
        h, axis=0
    )
    return np.tile(base[:,None,:], (1,size[0],1))

def jitter(bgr: np.ndarray):
    if random.random() < 0.7:
        k = random.choice([3,5])
        bgr = cv2.GaussianBlur(bgr, (k,k), random.uniform(0.6,1.2))
    return bgr

def write_with_cv2(frames, out_path, fps=24, fourcc_str="mp4v"):
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    vw = cv2.VideoWriter(out_path, fourcc, fps, (w,h))
    if not vw.isOpened():
        return False
    for frm in frames:
        vw.write(frm)
    vw.release()
    return True

def write_with_ffmpeg(frames, out_path, fps=24):
    if shutil.which("ffmpeg") is None:
        return False
    tmpdir = tempfile.mkdtemp(prefix="synth_")
    try:
        # dump PNGs
        for i, frm in enumerate(frames):
            cv2.imwrite(os.path.join(tmpdir, f"frame_{i:04d}.png"), frm)
        # encode via VideoToolbox (fast) or libx264 fallback
        cmd = [
            "ffmpeg", "-y", "-framerate", str(fps), "-i", os.path.join(tmpdir, "frame_%04d.png"),
            "-vf", "format=yuv420p",
            "-c:v", "h264_videotoolbox", "-b:v", "4M",
            "-movflags", "+faststart",
            out_path
        ]
        # If videotoolbox fails, try libx264
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            cmd = [
                "ffmpeg", "-y", "-framerate", str(fps), "-i", os.path.join(tmpdir, "frame_%04d.png"),
                "-vf", "format=yuv420p",
                "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
                "-movflags", "+faststart",
                out_path
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def generate_clip(w=1280, h=720, seconds=1.5, fps=24):
    frames = []
    # mothership proxy box
    mw, mh = random.randint(w//12, w//9), random.randint(h//12, h//9)
    mx, my = random.randint(50, w-50-mw), random.randint(50, h-50-mh)
    vx, vy = random.randint(-3,3), random.randint(-2,2)
    # parasites (0–3)
    pcount = random.choice([0,1,2,2,3])
    paras = []
    for _ in range(pcount):
        pw, ph = max(8, mw//6), max(8, mh//6)
        px = mx + random.randint(int(mw*0.15), int(mw*0.85)-pw)
        py = my + int(mh*0.55) + random.randint(0, int(mh*0.25))
        paras.append([px,py,pw,ph])

    labels = {"frames":[]}
    total = int(seconds*fps)
    bg = rand_bg((w,h))
    for t in range(total):
        canvas = bg.copy()
        # move mothership
        mx += vx; my += vy
        ms_box = [mx, my, mx+mw, my+mh]
        cv2.rectangle(canvas, (mx,my), (mx+mw,my+mh), (60,60,60), thickness=-1)
        pboxes=[]
        for j in range(len(paras)):
            paras[j][0] += vx; paras[j][1] += vy
            px,py,pw,ph = paras[j]
            cv2.rectangle(canvas, (px,py), (px+pw,py+ph), (100,100,100), thickness=-1)
            pboxes.append([px,py,px+pw,py+ph])
        canvas = jitter(canvas)
        frames.append(canvas.astype(np.uint8))
        labels["frames"].append({"fixed_wing": ms_box, "parasites": pboxes})
    return frames, labels

def main(out_dir, n, fps, seconds):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    wrote = 0
    for k in range(n):
        frames, labels = generate_clip(seconds=seconds, fps=fps)
        mp4_path = os.path.join(out_dir, f"synth_{k:04d}.mp4")
        json_path = os.path.join(out_dir, f"synth_{k:04d}.json")
        ok = write_with_cv2(frames, mp4_path, fps=fps, fourcc_str="mp4v")
        if not ok:
            print(f"[info] OpenCV mp4v failed for {mp4_path}, trying 'avc1'…")
            ok = write_with_cv2(frames, mp4_path, fps=fps, fourcc_str="avc1")
        if not ok:
            print(f"[info] OpenCV failed; falling back to ffmpeg for {mp4_path}…")
            ok = write_with_ffmpeg(frames, mp4_path, fps=fps)
        if not ok:
            print(f"[ERROR] Could not write video {mp4_path}. Skipping.")
            continue
        with open(json_path, "w") as f:
            json.dump(labels, f)
        wrote += 1
    print(f"[done] Generated {wrote}/{n} clips in {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--seconds", type=float, default=1.5)
    args = ap.parse_args()
    main(args.out, args.n, args.fps, args.seconds)
