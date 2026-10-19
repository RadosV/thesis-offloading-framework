#!/usr/bin/env python3
"""
E1 bench: inference time of the perception modules on THIS machine.
Run it on the Pi AND on the PC, with the camera pointed at the target.

  python3 bench_inference.py                          # aruco, yolo:160, yolo:224, yolo:320; 300 frames each
  python3 bench_inference.py --perc aruco yolo:320 --n 500
  python3 bench_inference.py --source clip.mp4        # video/image file instead of the camera
  python3 bench_inference.py --live yolo:320          # live: visible / conf / center_x / width / command
        -> check the target is detected from the start mark
        -> read the width at the 'arrived' distance = ARRIVE_W in run_experiments.py

Writes bench_logs/bench_<host>.csv (one row per frame) and
bench_logs/bench_summary_<host>.csv (one row per module).
"""
import argparse
import csv
import math
import os
import socket
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.dirname(HERE)):
    if os.path.isdir(os.path.join(_p, "framework")) and _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from framework.camera import CameraThread
from framework.perception import make_perception, steer_command

try:
    import psutil
except ImportError:
    psutil = None


def next_frame(camera):
    while True:
        item = camera.latest()
        if item is not None:
            return item[2]
        time.sleep(0.01)


def live(camera, spec, threshold=0.2):
    # show the RAW best score, so a target that flickers around the threshold is visible
    # as a number instead of just disappearing
    # only YOLO takes a confidence parameter; ArUco either detects the marker or does not
    if spec.startswith("yolo") and "conf=" not in spec:
        spec = spec + ",conf=0.01"
    perception = make_perception(spec)
    print("Live %s - raw score; the experiments count a detection from %.2f. Ctrl-C to stop"
          % (perception.spec, threshold))
    best = 0.0
    try:
        while True:
            frame = next_frame(camera)
            det = perception.detect(frame)
            best = max(best, det.conf)
            print("\rscore=%.2f %-9s x=%6s y=%6s (mid 120) width=%6s cmd=%-7s %7.1f ms  (best %.2f)  " % (
                det.conf, "DETECTED" if det.conf >= threshold else "too low",
                "%.1f" % det.center_x if det.visible else "-",
                "%.1f" % det.center_y if det.visible else "-",
                "%.1f" % det.width if det.visible else "-",
                steer_command(det, frame.shape[1]) if det.conf >= threshold else "stop",
                det.infer_ms, best), end="", flush=True)
            time.sleep(0.2)
    except KeyboardInterrupt:
        print()


def bench(camera, spec, n, warmup, writer, host):
    t_load = time.time()
    perception = make_perception(spec)
    load_s = time.time() - t_load
    for _ in range(warmup):
        perception.detect(next_frame(camera))
    if psutil:
        psutil.cpu_percent(None)
    times, visible = [], 0
    for i in range(n):
        det = perception.detect(next_frame(camera))
        times.append(det.infer_ms)
        visible += int(det.visible)
        writer.writerow([host, perception.spec, i, "%.3f" % det.infer_ms, int(det.visible),
                         "%.3f" % det.conf, "%.1f" % det.center_x if det.visible else "",
                         "%.1f" % det.width if det.visible else ""])
    t = np.array(times)
    return dict(host=host, perception=perception.spec, backend=getattr(perception, "backend", "cv2"),
                n=n, load_s=round(load_s, 2), mean_ms=round(t.mean(), 2),
                median_ms=round(float(np.median(t)), 2), p95_ms=round(float(np.percentile(t, 95)), 2),
                max_ms=round(t.max(), 2), fps=round(1000.0 / t.mean(), 1),
                visible_rate=round(visible / float(n), 3),
                cpu_pct=psutil.cpu_percent(None) if psutil else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perc", nargs="+", default=["aruco", "yolo:160", "yolo:224", "yolo:320"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--source", default="0")
    ap.add_argument("--capture", help="camera resolution to grab, e.g. 1280x720")
    ap.add_argument("--crop", type=float, default=1.0,
                    help="keep this fraction of the width (digital zoom): 0.3 turns a 90 deg webcam into ~33 deg")
    ap.add_argument("--live", metavar="PERCEPTION")
    ap.add_argument("--out", default="bench_logs")
    args = ap.parse_args()

    camera = CameraThread(args.source, capture=args.capture, crop=args.crop)
    if args.crop < 1.0:
        print("capture %dx%d, central crop %.2f -> ~%.0f deg view (90 deg lens)" % (
            camera.capture_size[0], camera.capture_size[1], args.crop,
            2 * math.degrees(math.atan(args.crop * math.tan(math.radians(45))))))
    camera.start()
    if args.live:
        live(camera, args.live)
        camera.stop()
        return

    host = socket.gethostname()
    os.makedirs(args.out, exist_ok=True)
    summaries = []
    with open(os.path.join(args.out, "bench_%s.csv" % host), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["host", "perception", "i", "infer_ms", "visible", "conf", "center_x", "width"])
        for spec in args.perc:
            print("--- %s ..." % spec, flush=True)
            try:
                summaries.append(bench(camera, spec, args.n, args.warmup, writer, host))
            except Exception as e:
                print("    skipped: %s" % e)
    camera.stop()
    if not summaries:
        return
    with open(os.path.join(args.out, "bench_summary_%s.csv" % host), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print("\n%-28s %8s %8s %8s %7s %8s %6s" % ("perception", "mean ms", "median", "p95", "fps", "visible", "cpu%"))
    for s in summaries:
        print("%-28s %8.1f %8.1f %8.1f %7.1f %8.2f %6s" % (
            s["perception"], s["mean_ms"], s["median_ms"], s["p95_ms"], s["fps"], s["visible_rate"], s["cpu_pct"]))


if __name__ == "__main__":
    main()
