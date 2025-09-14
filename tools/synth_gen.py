import os, argparse, random, json, shutil, subprocess, tempfile
from pathlib import Path
from PIL import Image, ImageFilter
import numpy as np
import cv2

# --- helpers for parasite placement and motion ---

def clamp_box_xyxy(b, W, H):
    x1,y1,x2,y2 = b
    x1 = max(0, min(W-1, int(x1)))
    y1 = max(0, min(H-1, int(y1)))
    x2 = max(0, min(W-1, int(x2)))
    y2 = max(0, min(H-1, int(y2)))
    if x2 <= x1: x2 = min(W-1, x1+1)
    if y2 <= y1: y2 = min(H-1, y1+1)
    return [x1,y1,x2,y2]

def move_box_xyxy(b, dx, dy, W, H):
    x1,y1,x2,y2 = b
    return clamp_box_xyxy([x1+dx, y1+dy, x2+dx, y2+dy], W, H)

def place_parasites(ms_box, W, H, min_px=20, max_px=40, min_q=2, max_q=3):
    """
    Place 2–3 parasite boxes in the pylon band under mothership.
    Returns list of [x1,y1,x2,y2].
    """
    import numpy as np
    x1,y1,x2,y2 = [float(v) for v in ms_box]
    mw, mh = x2-x1, y2-y1
    band_top = y1 + 0.12*mh
    band_bot = y1 + 0.22*mh
    lx = x1 + 0.20*mw
    rx = x2 - 0.20*mw

    n_quads = np.random.randint(min_q, max_q+1)
    quads = []
    for i in range(n_quads):
        side = int(np.random.uniform(min_px, max_px))
        frac = np.linspace(0.25, 0.75, max(2,n_quads))[i]
        cx = lx + frac*(rx-lx)
        cy = np.random.uniform(band_top, band_bot)
        qx1 = cx - side/2; qy1 = cy - side/2
        qx2 = qx1 + side;  qy2 = qy1 + side
        quads.append(clamp_box_xyxy([qx1,qy1,qx2,qy2], W, H))
    return quads


# --- config defaults (can be overridden by CLI) ---
MIN_QUAD_PX = 20     # min width/height of a quad (px) on the FINAL frame
MAX_QUAD_PX = 40     # max width/height (px)
MIN_QUADS   = 2
MAX_QUADS   = 3

def clamp_quad_size(px, min_px=MIN_QUAD_PX, max_px=MAX_QUAD_PX):
    return max(min_px, min(max_px, int(round(px))))

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

import numpy as np

def place_parasites(ms_box, frame_w, frame_h,
                    min_px=20, max_px=40,
                    min_q=2, max_q=3):
    """
    Place 2–3 parasites in the pylon band under the mothership.
    Ensures each parasite box has width/height >= min_px.
    Returns a list of [x1,y1,x2,y2].
    """
    x1,y1,x2,y2 = [float(v) for v in ms_box]
    mw, mh = x2-x1, y2-y1

    # define band under wing: 12–22% below top edge
    band_top = y1 + 0.12*mh
    band_bot = y1 + 0.22*mh
    lx = x1 + 0.20*mw
    rx = x2 - 0.20*mw

    n_quads = np.random.randint(min_q, max_q+1)
    quads = []
    for i in range(n_quads):
        side = int(np.random.uniform(min_px, max_px))
        frac = np.linspace(0.25, 0.75, max(2,n_quads))[i]
        cx = lx + frac*(rx-lx)
        cy = np.random.uniform(band_top, band_bot)
        qx1 = int(np.clip(cx - side/2, 0, frame_w-1))
        qy1 = int(np.clip(cy - side/2, 0, frame_h-1))
        qx2 = qx1 + side
        qy2 = qy1 + side
        if qx2>qx1+2 and qy2>qy1+2:
            quads.append([qx1,qy1,qx2,qy2])
    return quads

def generate_clip(w=1280, h=720, seconds=1.5, fps=24):
    frames = []
    labels = {"frames": []}

    # mothership init
    mw = random.randint(w//12, w//9)
    mh = random.randint(h//12, h//9)
    mx = random.randint(50, w-50-mw)
    my = random.randint(50, h-50-mh)
    vx = random.randint(-3, 3)
    vy = random.randint(-2, 2)

    # frame 0 boxes
    ms_box = [mx, my, mx+mw, my+mh]
    paras  = place_parasites(ms_box, w, h, min_px=20, max_px=40, min_q=2, max_q=3)

    total = int(seconds * fps)
    bg = rand_bg((w, h))  # your existing background generator

    for t in range(total):
            # debug stats every ~20th frame
        if t % 20 == 0:  
            debug_parasite_stats("synth", t, ms_box, paras)
        canvas = bg.copy()

        # move mothership
        mx += vx; my += vy
        ms_box = clamp_box_xyxy([mx, my, mx+mw, my+mh], w, h)

        # move parasites by the SAME delta + tiny jitter (simulate dangling FPVs)
        sx = random.randint(-1, 1)
        sy = random.randint(-1, 1)
        paras = [move_box_xyxy(p, vx+sx, vy+sy, w, h) for p in paras]


        # draw mothership
        x1,y1,x2,y2 = ms_box
        cv2.rectangle(canvas, (x1,y1), (x2,y2), (60,60,60), thickness=-1)

        # draw parasites
        pboxes = []
        for p in paras:
            px1,py1,px2,py2 = p
            cv2.rectangle(canvas, (px1,py1), (px2,py2), (100,100,100), thickness=-1)
            pboxes.append([px1,py1,px2,py2])

        # optional corruption
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

def debug_parasite_stats(clip_id, frame_idx, ms_box, parasites):
    """Print simple stats: mothership size, parasite sizes, avg displacement."""
    mx1,my1,mx2,my2 = ms_box
    mw, mh = mx2-mx1, my2-my1
    sizes = []
    for (x1,y1,x2,y2) in parasites:
        sizes.append((x2-x1, y2-y1))
    if sizes:
        avg_w = sum(s[0] for s in sizes)/len(sizes)
        avg_h = sum(s[1] for s in sizes)/len(sizes)
        print(f"[synth dbg] clip={clip_id} frame={frame_idx:03d} "
              f"ms=({mw}x{mh}) "
              f"parasites={len(parasites)} "
              f"avg_px={avg_w:.1f}x{avg_h:.1f} "
              f"sizes={sizes}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--seconds", type=float, default=1.5)
    args = ap.parse_args()
    main(args.out, args.n, args.fps, args.seconds)
