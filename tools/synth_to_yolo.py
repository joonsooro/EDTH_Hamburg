# tools/synth_to_yolo.py
import os, json, glob
from PIL import Image

def xyxy_to_yolo(x1,y1,x2,y2,w,h):
    cx=(x1+x2)/(2*w); cy=(y1+y2)/(2*h); bw=(x2-x1)/w; bh=(y2-y1)/h
    return cx,cy,bw,bh

def convert_split(split):
    imgs_glob=f"data/synth_frames/images/{split}/*.jpg"
    for img_path in glob.glob(imgs_glob):
        # img name pattern: synth_XXXX_NNNN.jpg
        stem=os.path.splitext(os.path.basename(img_path))[0]
        clip_id, frame_str = stem.rsplit("_",1)
        frame_idx = int(frame_str) - 1            # 🔧 FIX: filenames start at 0001 → 0-based index

        json_path=os.path.join("data","synth", f"{clip_id}.json")
        if not os.path.exists(json_path):
            continue

        with open(json_path,"r") as f:
            lbl=json.load(f)
        frames = lbl.get("frames", [])            # 🔧 guard if missing

        if not (0 <= frame_idx < len(frames)):    # 🔧 robust bounds check
            continue
        fr = frames[frame_idx]

        with Image.open(img_path) as im:
            w,h=im.size

        lines=[]
        # class 0 = fixed_wing
        x1,y1,x2,y2 = fr["fixed_wing"]
        cx,cy,bw,bh = xyxy_to_yolo(x1,y1,x2,y2,w,h)
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        # class 1 = quad (parasites)
        for p in fr.get("parasites", []):         # 🔧 guard if empty/missing
            x1,y1,x2,y2 = p
            cx,cy,bw,bh = xyxy_to_yolo(x1,y1,x2,y2,w,h)
            lines.append(f"1 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        out_txt=img_path.replace("/images/","/labels/").replace(".jpg",".txt")
        os.makedirs(os.path.dirname(out_txt), exist_ok=True)
        with open(out_txt,"w") as f:
            f.write("\n".join(lines))

if __name__=="__main__":
    convert_split("train")
    convert_split("val")
    print("YOLO labels written for train/val.")
