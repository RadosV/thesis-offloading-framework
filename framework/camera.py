"""
camera.py  -- newest-frame camera grabber
=========================================
Reads frames continuously in its own thread and keeps only the newest one, so
consumers never get stale frames out of the driver's buffer. Every frame gets
a frame id and a capture timestamp (Pi clock) that travel with it through
both tiers.

Optional digital zoom: capture at a higher resolution (CAPTURE, e.g. 1280x720) and keep
only the middle CROP fraction of the width before scaling to the working frame size.
That narrows the field of view of a wide-angle webcam without costing the detector
anything (the frame stays 320x240), so a distant target covers more pixels:
  crop 1.00 -> 90 deg (the raw view of a 90 deg webcam)
  crop 0.50 -> 53 deg,  crop 0.35 -> 38 deg,  crop 0.30 -> 33 deg,  crop 0.25 -> 28 deg

SOURCE:
  0, 1, ...            camera index (the USB camera on the Pi)
  path/to/video.mp4    video file, replayed in a loop at `fps`
  path/to/image.png    one still image, repeated at `fps`
  synthetic[:marker]   target image sliding left/right on a white canvas
                       (default: services/marker_0.png) - for tests without the car
"""

import math
import os
import threading
import time

import cv2
import numpy as np

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp")
DEFAULT_MARKER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "services", "marker_0.png")


class CameraThread(threading.Thread):
    def __init__(self, source="0", width=320, height=240, fps=30.0, capture=None, crop=1.0):
        super().__init__(daemon=True)
        self.source, self.width, self.height, self.fps = str(source), int(width), int(height), float(fps)
        self.crop = float(crop)
        self.capture_size = (self.width, self.height)
        if capture:
            self.capture_size = tuple(int(v) for v in str(capture).lower().split("x"))
        self.error = None
        self._cond = threading.Condition()
        self._latest = None                      # (fid, t_capture, frame)
        self._fid = 0
        self._stop_evt = threading.Event()
        self._cap = None
        self._still = None
        if self.source.isdigit():
            self._cap = cv2.VideoCapture(int(self.source))
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_size[0])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_size[1])
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.capture_size = (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                 int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        elif self.source.startswith("synthetic"):
            path = self.source.partition(":")[2] or DEFAULT_MARKER
            self._still = cv2.imread(path)
            if self._still is None:
                raise RuntimeError("cannot read synthetic target image %s" % path)
        elif self.source.lower().endswith(IMAGE_EXT):
            self._still = cv2.imread(self.source)
            if self._still is None:
                raise RuntimeError("cannot read image %s" % self.source)
        else:
            self._cap = cv2.VideoCapture(self.source)
        if self._cap is not None and not self._cap.isOpened():
            raise RuntimeError("cannot open camera/video source %r" % self.source)

    # --- consumer API ---
    def latest(self):
        with self._cond:
            return self._latest

    def wait_new(self, last_fid, timeout=0.5):
        """Newest frame with an id > last_fid, or None after `timeout` s."""
        with self._cond:
            if self._latest is None or self._latest[0] <= last_fid:
                self._cond.wait(timeout)
            if self._latest is not None and self._latest[0] > last_fid:
                return self._latest
            return None

    def stop(self):
        self._stop_evt.set()

    # --- producer ---
    def _publish(self, frame):
        if self.crop < 1.0:                       # central crop = digital zoom
            h, w = frame.shape[:2]
            cw = max(int(w * self.crop), 16)
            ch = min(int(round(cw * self.height / float(self.width))), h)
            x, y = (w - cw) // 2, (h - ch) // 2
            frame = frame[y:y + ch, x:x + cw]
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        with self._cond:
            self._fid += 1
            self._latest = (self._fid, time.time(), frame)
            self._cond.notify_all()

    def run(self):
        try:
            if self.source.startswith("synthetic"):
                self._run_synthetic()
            elif self._still is not None:
                while not self._stop_evt.wait(1.0 / self.fps):
                    self._publish(self._still.copy())
            else:
                self._run_capture()
        finally:
            if self._cap is not None:
                self._cap.release()

    def _run_capture(self):
        live = self.source.isdigit()
        failures = 0
        while not self._stop_evt.is_set():
            ret, frame = self._cap.read()
            if not ret:
                if not live:                       # video file: loop it
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                failures += 1
                if failures > 50:
                    self.error = "camera stopped delivering frames"
                time.sleep(0.01)
                continue
            failures = 0
            self._publish(frame)
            if not live:
                self._stop_evt.wait(1.0 / self.fps)

    def _run_synthetic(self):
        size = int(self.height * 0.35)
        target = cv2.resize(self._still, (size, size), interpolation=cv2.INTER_NEAREST)
        y = (self.height - size) // 2
        t0 = time.time()
        while not self._stop_evt.wait(1.0 / self.fps):
            cx = self.width / 2.0 + self.width * 0.3 * math.sin(2 * math.pi * (time.time() - t0) / 4.0)
            x = int(cx - size / 2.0)
            frame = np.full((self.height, self.width, 3), 255, dtype=np.uint8)
            frame[y:y + size, x:x + size] = target
            self._publish(frame)
