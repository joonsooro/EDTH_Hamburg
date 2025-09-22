# detector.py
from __future__ import annotations
import os, sys, cv2, torch, numpy as np
from ultralytics import YOLO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# If you want, you can import CLASS_MAP here to sanity-check class ids vs names.
# from vision.classes import CLASS_MAP as DEFAULT_CLASS_MAP


def get_device() -> str:
    """
    Prefer CUDA when available (fastest), then Apple MPS, else CPU.
    """
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _has_boxes(r) -> bool:
    """Small helper so we don't repeat None/len checks."""
    return (r is not None) and (r.boxes is not None) and (len(r.boxes) > 0)


def class_conf_filter(boxes, confs, clses, thresholds, id2name):
    """
    thresholds: dict like {"fixed_wing": 0.15, "quad": 0.35}
    id2name:    CLASS_MAP dict

    NOTE: I’m leaving this here as a utility, but keep using it from streamlit.py,
    right after calling infer_bgr(), so you can live-tune thresholds in the UI.
    """
    if len(confs) == 0:
        return boxes, confs, clses
    thr_per_id = np.array([thresholds.get(id2name.get(int(k), ""), 0.0) for k in clses], dtype=float)
    keep = confs >= thr_per_id
    return boxes[keep], confs[keep], clses[keep]


class Detector:
    def __init__(
        self,
        weights: str = "models/best.pt",
        conf: float = 0.25,
        iou: float = 0.50,
        max_det: int = 100,
        agnostic: bool = True,
        imgsz: int = 960,
        allow_bgr_fallback: bool = True,   # NEW: make fallback optional
    ):
        if not os.path.exists(weights):
            raise FileNotFoundError(f"weights not found: {weights}")
        self.model = YOLO(weights)

        # Fuse once (small speed gain); harmless if already fused.
        try:
            self.model.fuse()
        except Exception:
            pass

        self.conf = float(conf)
        self.iou = float(iou)
        self.max_det = int(max_det)
        self.agnostic = bool(agnostic)
        self.imgsz = int(imgsz)
        self.device = get_device()
        self.allow_bgr_fallback = bool(allow_bgr_fallback)

        print(
            f"[Detector] weights={weights} device={self.device} "
            f"conf={self.conf} iou={self.iou} max_det={self.max_det} "
            f"agnostic={self.agnostic} imgsz={self.imgsz} names={self.model.names}"
        )

    def _kw(self):
        """
        Centralize Ultralytics predict kwargs so A/B checks match app thresholds.
        Use half-precision on CUDA only (MPS/CPU doesn’t support it well).
        """
        return dict(
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            max_det=self.max_det,
            agnostic_nms=self.agnostic,  # Ultralytics still accepts this alias
            verbose=False,
            device=self.device,
            half=self.device.startswith("cuda"),
        )

    def infer_bgr(self, img_bgr: np.ndarray):
        """
        Try RGB first (Ultralytics preproc expects RGB); optionally fall back to BGR
        if no boxes are produced (useful for some custom/ONNX paths).
        Returns (boxes[x1,y1,x2,y2], confs, clses[int]).
        """
        assert (
            img_bgr is not None and img_bgr.ndim == 3 and img_bgr.dtype == np.uint8
        ), f"bad frame: {None if img_bgr is None else (img_bgr.shape, img_bgr.dtype)}"

        KW = self._kw()

        # 1) RGB path
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        try:
            r = self.model.predict(img_rgb, **KW)[0]
        except Exception as e:
            # If something odd happens in Ultralytics preproc, fail gracefully.
            print(f"[Detector] predict(RGB) error: {e}")
            r = None

        # 2) Optional BGR fallback (only if RGB yielded nothing)
        if not _has_boxes(r) and self.allow_bgr_fallback:
            try:
                r_bgr = self.model.predict(img_bgr, **KW)[0]
            except Exception as e:
                print(f"[Detector] predict(BGR) error: {e}")
                r_bgr = None
            if _has_boxes(r_bgr):
                # Print once per session would be ideal; keep it lightweight.
                print("[Detector] Fallback BGR produced boxes")
                r = r_bgr

        if not _has_boxes(r):
            # no detections from either path → return empty arrays of the right dtype
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
            )

        # Normalize outputs (CPU tensors → numpy, correct dtypes)
        boxes = r.boxes.xyxy.detach().cpu().numpy().astype(np.float32, copy=False)
        confs = r.boxes.conf.detach().cpu().numpy().astype(np.float32, copy=False)
        clses = r.boxes.cls.detach().cpu().numpy().astype(np.int32, copy=False)

        return boxes, confs, clses
