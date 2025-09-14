# quick_check.py
from ultralytics import YOLO
import cv2

IMG = "data/synth_frames/images/val/synth_0001_0002.jpg"  # pick a real val frame that has quads

m = YOLO("models/best.pt")  # your trained weights
r = m.predict(IMG, conf=0.05, imgsz=960, verbose=False)[0]
print("names:", m.model.names)
print("num boxes:", 0 if r.boxes is None else len(r.boxes))
vis = r.plot()
cv2.imwrite("export/debug/quick_check.jpg", vis[:, :, ::-1])
print("wrote export/debug/quick_check.jpg")
