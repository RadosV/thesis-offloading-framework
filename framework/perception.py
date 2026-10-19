"""
perception.py  -- framework extension point 1: perception modules
=================================================================
A perception module turns one BGR camera frame into a Detection of the
navigation target. The SAME class runs on the Pi (local tier) and on the edge
server (remote tier), so the two tiers differ only in WHERE the computation
happens, never in the algorithm.

Modules (PERCEPTION env var / frame header):
  aruco          classical CV: ArUco fiducial marker (ID 0), computationally cheap
  yolo:<size>    CNN: YOLOv5n (default, arch=5) or YOLOv8n (arch=8), pretrained on COCO,
                 run with OpenCV DNN; target = stop sign (COCO class 11);
                 <size> = input 160 | 224 | 320. YOLOv5n is the default because the
                 Pi's OpenCV 4.5.5 cannot load YOLOv8 models (needs >= 4.7).

A new workload is added by subclassing Perception and registering its name in
canonical_spec() / make_perception(); nothing else in the framework changes.
"""

import os
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .spec import parse_spec

# --- Steering (shared by both tiers) ---
# Thresholds are defined for a 320 px wide frame and scaled for other widths.
REF_WIDTH = 320.0
LEFT_X = 110.0
RIGHT_X = 210.0

COCO_STOP_SIGN = 11
MODEL_DIR = os.environ.get(
    "YOLO_MODEL_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))


@dataclass
class Detection:
    visible: bool = False
    center_x: Optional[float] = None   # target centre, frame pixels
    center_y: Optional[float] = None   # vertical centre, for aiming the camera
    width: Optional[float] = None      # apparent target width, pixels (arrival test)
    conf: float = 0.0
    infer_ms: float = 0.0


# Proportional steering: the servo deflection scales with how far off-centre the
# target is, instead of slamming to a fixed angle. A relay (left/forward/right)
# controller cannot settle at this sensing rate - it limit-cycles around the
# deadband edge and walks the target out of the narrow frame.
STEER_DEADBAND = 0.05      # |v| below this counts as straight


def steer_value(center_x, frame_width):
    """Normalised steering demand in [-1, 1]. Positive = target right of centre.

    None when there is no usable detection. The caller turns this into a servo
    pulse width; the mapping lives in motor_controller.py (STEER_GAIN_US).
    """
    if center_x is None:
        return None
    half = frame_width / 2.0
    v = max(-1.0, min(1.0, (center_x - half) / half))
    return 0.0 if abs(v) < STEER_DEADBAND else v


def steer_payload(center_x, frame_width):
    """MQTT payload for proportional steering, or 'stop' when nothing is visible."""
    v = steer_value(center_x, frame_width)
    return "stop" if v is None else "go:%.3f" % v


def steer_command(det, frame_width):
    """Map a detection to forward / left / right / stop (relay; kept for E5/tests)."""
    if det is None or not det.visible or det.center_x is None:
        return "stop"
    scale = frame_width / REF_WIDTH
    if det.center_x < LEFT_X * scale:
        return "left"
    if det.center_x > RIGHT_X * scale:
        return "right"
    return "forward"


class Perception:
    """Interface: detect(frame) -> Detection. `spec` recreates the same module."""
    spec = "base"

    def detect(self, frame):
        raise NotImplementedError


class ArucoPerception(Perception):
    def __init__(self, target_id=0):
        aruco = cv2.aruco
        self.target_id = int(target_id)
        self.spec = "aruco:id=%d" % self.target_id
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        if hasattr(aruco, "ArucoDetector"):                       # OpenCV >= 4.7
            self._detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
            self._detect = self._detector.detectMarkers
        else:                                                     # legacy API
            params = aruco.DetectorParameters_create()
            self._detect = lambda gray: aruco.detectMarkers(gray, dictionary, parameters=params)

    def detect(self, frame):
        t0 = time.perf_counter()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detect(gray)
        det = Detection()
        if ids is not None:
            ids_list = ids.flatten().tolist()
            if self.target_id in ids_list:
                c = corners[ids_list.index(self.target_id)][0]
                det = Detection(visible=True, center_x=float((c[0][0] + c[2][0]) / 2.0),
                                center_y=float((c[0][1] + c[2][1]) / 2.0),
                                width=float(c[:, 0].max() - c[:, 0].min()), conf=1.0)
        det.infer_ms = (time.perf_counter() - t0) * 1000.0
        return det


class YoloPerception(Perception):
    """YOLOv5n / YOLOv8n exported to ONNX (tools/export_yolo.py), run with OpenCV DNN.
    Falls back to onnxruntime if the installed OpenCV cannot load the model
    (YOLO_BACKEND=auto | cv2 | ort)."""

    def __init__(self, imgsz=320, target_cls=COCO_STOP_SIGN, conf=0.2, model_path=None,
                 backend=None, threads=0, arch=5):
        self.imgsz, self.target_cls, self.conf = int(imgsz), int(target_cls), float(conf)
        self.arch = int(arch)
        self.spec = "yolo:%d,cls=%d,conf=%g,arch=%d" % (self.imgsz, self.target_cls, self.conf, self.arch)
        self.model_path = model_path or os.path.join(MODEL_DIR, "yolov%dn_%d.onnx" % (self.arch, self.imgsz))
        if not os.path.exists(self.model_path):
            raise FileNotFoundError("YOLO model not found: %s (run tools/export_yolo.py and copy "
                                    "models/ next to framework/)" % self.model_path)
        backend = (backend or os.environ.get("YOLO_BACKEND", "auto")).lower()
        if threads:
            cv2.setNumThreads(int(threads))
        self.net = self.session = None
        if backend in ("auto", "cv2"):
            try:
                self.net = cv2.dnn.readNetFromONNX(self.model_path)
            except cv2.error:
                if backend == "cv2":
                    raise
        if self.net is None:
            import onnxruntime as ort
            self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
            self._input = self.session.get_inputs()[0].name
        self.backend = "cv2" if self.net is not None else "onnxruntime"

    def _letterbox(self, frame):
        """Scale the longer side to imgsz and pad the rest (top-left aligned)."""
        h, w = frame.shape[:2]
        scale = self.imgsz / float(max(h, w))
        nw, nh = int(round(w * scale)), int(round(h * scale))
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        canvas[:nh, :nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return canvas, scale

    def detect(self, frame):
        t0 = time.perf_counter()
        img, scale = self._letterbox(frame)
        blob = cv2.dnn.blobFromImage(img, 1.0 / 255.0, (self.imgsz, self.imgsz), swapRB=True)
        if self.net is not None:
            self.net.setInput(blob)
            out = self.net.forward()
        else:
            out = self.session.run(None, {self._input: blob})[0]
        det = self._parse(out, scale)
        det.infer_ms = (time.perf_counter() - t0) * 1000.0
        return det

    def _parse(self, out, scale):
        """Per candidate cx, cy, w, h, then
             YOLOv5 (1, N, 85): objectness + 80 class scores -> score = obj * class
             YOLOv8 (1, 84, N): 80 class scores
        Only the best box of the target class is needed, so no NMS."""
        out = np.squeeze(np.asarray(out))
        if out.shape[0] < out.shape[1]:
            out = out.T                                    # -> (N, 84 or 85)
        if out.shape[1] == 85:
            scores = out[:, 4] * out[:, 5 + self.target_cls]
        else:
            scores = out[:, 4 + self.target_cls]
        i = int(np.argmax(scores))
        if scores[i] < self.conf:
            return Detection()
        cx, cy, bw, _ = out[i, :4]
        return Detection(visible=True, center_x=float(cx / scale), center_y=float(cy / scale),
                         width=float(bw / scale), conf=float(scores[i]))


def canonical_spec(spec="aruco"):
    """Full spec with all defaults filled in, without loading a model. It travels
    in every frame header so the edge brain builds exactly the same module."""
    name, p = parse_spec(spec)
    if name == "aruco":
        return "aruco:id=%d" % int(p.get("id", 0))
    if name == "yolo":
        return "yolo:%d,cls=%d,conf=%g,arch=%d" % (int(p.get("", p.get("imgsz", 320))),
                                                   int(p.get("cls", COCO_STOP_SIGN)),
                                                   float(p.get("conf", 0.2)),
                                                   int(p.get("arch", 5)))
    raise ValueError("unknown perception %r (use aruco | yolo:<size>)" % spec)


def make_perception(spec="aruco"):
    name, p = parse_spec(canonical_spec(spec))
    if name == "aruco":
        return ArucoPerception(p["id"])
    return YoloPerception(p[""], p["cls"], p["conf"],
                          model_path=os.environ.get("YOLO_MODEL") or None,
                          threads=int(os.environ.get("YOLO_THREADS", "0")), arch=p["arch"])
