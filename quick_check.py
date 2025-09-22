# quick_check.py
from ultralytics import YOLO
import cv2
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from vision.classes import CLASS_MAP

IMG = "data/synth_frames/images/val/synth_0001_0002.jpg"  # pick a real val frame that has quads

m = YOLO("models/best.pt")  # your trained weights
r = m.predict(IMG, conf=0.05, imgsz=960, verbose=False)[0]
print("names:", m.names)
print("class naems:", CLASS_MAP)
print("num boxes:", 0 if r.boxes is None else len(r.boxes))
vis = r.plot()
cv2.imwrite("export/debug/quick_check.jpg", vis[:, :, ::-1])
print("wrote export/debug/quick_check.jpg")
