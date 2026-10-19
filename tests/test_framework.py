#!/usr/bin/env python3
"""
Unit tests for the framework (no car, no network needed):
    python code/tests/test_framework.py -v
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
for _p in (HERE, CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2
import numpy as np

from framework.netprofile import ImpairedChannel, make_profile
from framework.perception import (ArucoPerception, Detection, YoloPerception, canonical_spec,
                                  steer_command)
from framework.policies import CostModel, EwmaHysteresis, FixedTimeout, make_policy
from framework.protocol import (cloud_reply, decode_frame, encode_frame, frame_header,
                                parse_cloud_reply, parse_frame_header)
from framework.spec import label, parse_spec

MARKER = os.path.join(CODE, "services", "marker_0.png")


def marker_frame(x_center, size=80, width=320, height=240):
    frame = np.full((height, width, 3), 255, dtype=np.uint8)
    marker = cv2.resize(cv2.imread(MARKER), (size, size), interpolation=cv2.INTER_NEAREST)
    x, y = int(x_center - size / 2), (height - size) // 2
    frame[y:y + size, x:x + size] = marker
    return frame


class SpecTest(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_spec("yolo:320"), ("yolo", {"": 320.0}))
        self.assertEqual(parse_spec("ewma:hi=300,lo=200"), ("ewma", {"hi": 300.0, "lo": 200.0}))
        self.assertEqual(parse_spec("stable"), ("stable", {}))

    def test_label(self):
        self.assertEqual(label("yolo:320"), "yolo320")
        self.assertEqual(label("timeout:tau=0.5"), "timeout-tau0.5")

    def test_canonical(self):
        self.assertEqual(canonical_spec("aruco"), "aruco:id=0")
        self.assertEqual(canonical_spec("yolo"), "yolo:320,cls=11,conf=0.2,arch=5")
        self.assertEqual(canonical_spec("yolo:160,arch=8"), "yolo:160,cls=11,conf=0.2,arch=8")


class SteerTest(unittest.TestCase):
    def test_thresholds(self):
        det = lambda x: Detection(visible=True, center_x=x, width=10)
        self.assertEqual(steer_command(det(100), 320), "left")
        self.assertEqual(steer_command(det(160), 320), "forward")
        self.assertEqual(steer_command(det(220), 320), "right")
        self.assertEqual(steer_command(det(200), 640), "left")       # thresholds scale to 220/420
        self.assertEqual(steer_command(Detection(), 320), "stop")


@unittest.skipUnless(os.path.exists(MARKER), "marker image missing")
class ArucoTest(unittest.TestCase):
    def test_detects_marker_position(self):
        perception = ArucoPerception()
        for x, cmd in ((70, "left"), (160, "forward"), (250, "right")):
            det = perception.detect(marker_frame(x))
            self.assertTrue(det.visible)
            self.assertAlmostEqual(det.center_x, x, delta=3)
            self.assertAlmostEqual(det.width, 80, delta=4)
            self.assertEqual(steer_command(det, 320), cmd)

    def test_no_marker(self):
        det = ArucoPerception().detect(np.full((240, 320, 3), 255, np.uint8))
        self.assertFalse(det.visible)


class YoloParseTest(unittest.TestCase):
    def setUp(self):
        self.yolo = YoloPerception.__new__(YoloPerception)      # no model file needed
        self.yolo.imgsz, self.yolo.target_cls, self.yolo.conf = 320, 11, 0.4
        self.out = np.zeros((1, 84, 2100), np.float32)
        self.out[0, 0:4, 5] = [100, 80, 40, 40]      # stop sign: cx, cy, w, h (letterboxed px)
        self.out[0, 4 + 11, 5] = 0.9
        self.out[0, 0:4, 7] = [250, 80, 60, 60]      # a person with higher score: ignored
        self.out[0, 4 + 0, 7] = 0.95

    def test_best_target_box_and_scaling(self):
        _, scale = self.yolo._letterbox(np.zeros((240, 320, 3), np.uint8))
        det = self.yolo._parse(self.out, scale)
        self.assertTrue(det.visible)
        self.assertAlmostEqual(det.center_x, 100)
        self.assertAlmostEqual(det.width, 40)
        _, scale = self.yolo._letterbox(np.zeros((480, 640, 3), np.uint8))
        det = self.yolo._parse(self.out, scale)
        self.assertAlmostEqual(det.center_x, 200)

    def test_yolov5_format(self):
        out = np.zeros((1, 6300, 85), np.float32)              # YOLOv5 @320: 3 anchors x 2100 cells
        out[0, 10, 0:4] = [100, 80, 40, 40]
        out[0, 10, 4], out[0, 10, 5 + 11] = 0.9, 0.9          # score = 0.81
        out[0, 20, 0:4] = [250, 80, 60, 60]
        out[0, 20, 4], out[0, 20, 5 + 11] = 0.5, 0.9          # score = 0.45, lower
        det = self.yolo._parse(out, 1.0)
        self.assertTrue(det.visible)
        self.assertAlmostEqual(det.center_x, 100)
        self.assertAlmostEqual(det.conf, 0.81, places=5)

    def test_below_confidence(self):
        self.out[0, 4 + 11, 5] = 0.2
        self.assertFalse(self.yolo._parse(self.out, 1.0).visible)

    def test_letterbox_shape(self):
        img, scale = self.yolo._letterbox(np.zeros((240, 320, 3), np.uint8))
        self.assertEqual(img.shape, (320, 320, 3))
        self.assertEqual(scale, 1.0)


class PolicyTest(unittest.TestCase):
    def test_timeout(self):
        p = FixedTimeout(tau=0.5)
        self.assertEqual(p.decide(0, 50, 30, 0.1, "LOCAL"), "CLOUD")
        self.assertEqual(p.decide(0, 50, 30, 0.6, "CLOUD"), "LOCAL")

    def test_ewma_hysteresis(self):
        p = EwmaHysteresis(hi=300, lo=200, alpha=0.5, k=3)
        brain = "LOCAL"
        for _ in range(2):
            brain = p.decide(0, 50, 30, 0.05, brain)
        self.assertEqual(brain, "LOCAL")                  # k cycles not yet reached
        brain = p.decide(0, 50, 30, 0.05, brain)
        self.assertEqual(brain, "CLOUD")
        for _ in range(5):
            brain = p.decide(0, 900, 30, 0.05, brain)
        self.assertEqual((brain, p.reason), ("LOCAL", "latency"))
        self.assertEqual(p.decide(0, 50, 30, 0.05, brain), "LOCAL")    # EWMA still high

    def test_cost_model(self):
        p = CostModel(margin=30, alpha=1.0, k=2)
        for i in range(10):
            p.on_cloud_message(i * 0.05)
        self.assertAlmostEqual(p.rto(), 0.15)             # clamped to rto_min
        brain = "LOCAL"
        for _ in range(2):
            brain = p.decide(0, 100, 750, 0.05, brain)    # cloud much fresher (slow local YOLO)
        self.assertEqual((brain, p.reason), ("CLOUD", "cloud fresher"))
        for _ in range(2):
            brain = p.decide(0, 100, 40, 0.05, brain)     # cheap local (ArUco) is fresher
        self.assertEqual((brain, p.reason), ("LOCAL", "local fresher"))
        self.assertEqual(p.decide(0, 100, 750, 0.3, "CLOUD"), "LOCAL")  # silence > RTO: at once

    def test_cost_model_ignores_outage_gap(self):
        p = CostModel()
        for t in (0.0, 0.05, 0.10, 2.10):                 # 2 s gap = outage, not a sample
            p.on_cloud_message(t)
        self.assertLess(p.srtt, 0.1)

    def test_make_policy(self):
        self.assertIsInstance(make_policy("timeout:tau=0.2"), FixedTimeout)
        self.assertEqual(make_policy("ewma:hi=400").hi, 400)
        with self.assertRaises(ValueError):
            make_policy("nonsense")


class NetProfileTest(unittest.TestCase):
    def test_profiles(self):
        spike = make_profile("spike:ms=400,start=3,end=6", base_ms=50)
        self.assertEqual(spike.sample(2.0), 50)
        self.assertEqual(spike.sample(4.0), 450)
        self.assertEqual(spike.sample(None), 50)          # warm-up: base only
        self.assertIsNone(make_profile("outage").sample(4.0))
        self.assertEqual(make_profile("ramp:ms=600,start=2,end=12").sample(7.0), 300)
        self.assertEqual(make_profile("const:100").sample(1.0), 100)
        a = [make_profile("jitter", seed=1).sample(1.0) for _ in range(3)]
        self.assertEqual(a[0], make_profile("jitter", seed=1).sample(1.0))
        self.assertTrue(all(x >= 0 for x in a))

    def test_channel(self):
        ch = ImpairedChannel(make_profile("spike:ms=400,start=3,end=6"))
        ch.start(100.0)
        ch.push({"fid": 1}, now=101.0)                    # t=1: no delay
        ch.push({"fid": 2}, now=104.0)                    # t=4: +400 ms
        self.assertEqual([m["fid"] for m in ch.pop_due(101.0)], [1])
        self.assertEqual(ch.pop_due(104.2), [])
        self.assertEqual([m["fid"] for m in ch.pop_due(104.4)], [2])
        ch2 = ImpairedChannel(make_profile("outage"))
        ch2.start(0.0)
        ch2.push({"fid": 1}, now=4.0)
        self.assertEqual((ch2.n_arrived, ch2.n_dropped, ch2.pop_due(10.0)), (1, 1, []))


class ProtocolTest(unittest.TestCase):
    def test_frame_roundtrip(self):
        frame = np.random.RandomState(0).randint(0, 255, (240, 320, 3), dtype=np.uint8)
        payload, enc = encode_frame(frame, 0)
        header = parse_frame_header(frame_header(7, 1.5, enc, frame.shape, "aruco:id=0"))
        self.assertEqual((header["fid"], header["enc"], header["p"]), (7, "raw", "aruco:id=0"))
        self.assertTrue(np.array_equal(decode_frame(payload, header), frame))
        payload, enc = encode_frame(frame, 80)
        self.assertEqual(enc, "jpg")
        self.assertEqual(decode_frame(payload, {"enc": "jpg"}).shape, frame.shape)
        self.assertLess(len(payload), frame.nbytes)

    def test_legacy_header(self):
        self.assertEqual(parse_frame_header("raspberrypi"), {})

    def test_cloud_reply(self):
        det = Detection(visible=True, center_x=101.54, width=42.0, conf=1, infer_ms=3.21)
        rep = parse_cloud_reply(cloud_reply("left", det, {"fid": 5, "t": 9.5}).encode())
        self.assertEqual((rep["cmd"], rep["fid"], rep["t_cap"], rep["cx"]), ("left", 5, 9.5, 101.5))
        self.assertEqual(parse_cloud_reply(b"forward"), {"cmd": "forward"})


@unittest.skipUnless(os.path.exists(MARKER), "marker image missing")
class CameraCropTest(unittest.TestCase):
    """Digital zoom: keep the middle of a large frame instead of the whole wide-angle view."""

    def _publish(self, width, height, crop, frame):
        from framework.camera import CameraThread
        camera = CameraThread(MARKER, width=width, height=height, crop=crop)
        camera._publish(frame)
        return camera.latest()[2]

    def test_central_crop_without_resize(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        frame[240, 320] = (255, 255, 255)                  # centre of the wide frame
        out = self._publish(320, 240, 0.5, frame)
        self.assertEqual(out.shape, (240, 320, 3))
        self.assertTrue((out[120, 160] == 255).all())      # stays in the centre
        self.assertEqual(int(out.sum()), 3 * 255)          # nothing else kept

    def test_crop_then_resize(self):
        out = self._publish(160, 120, 0.5, np.zeros((480, 640, 3), np.uint8))
        self.assertEqual(out.shape, (120, 160, 3))

    def test_no_crop_keeps_full_view(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        frame[0, 0] = (255, 255, 255)                      # corner survives only without crop
        out = self._publish(320, 240, 1.0, frame)
        self.assertEqual(out.shape, (240, 320, 3))
        self.assertGreater(int(out.sum()), 0)


class ResourceSamplerTest(unittest.TestCase):
    def test_with_fake_psutil(self):
        import collections
        import tempfile
        import time
        from framework import resources
        Net = collections.namedtuple("Net", "bytes_sent bytes_recv")

        class FakeProcess:
            def cpu_percent(self, interval=None):
                return 50.0

            def memory_info(self):
                return collections.namedtuple("Mem", "rss")(100 * 2 ** 20)

        class FakePsutil:
            sent = 0
            Process = FakeProcess

            @classmethod
            def net_io_counters(cls, pernic=False):
                cls.sent += 125000
                net = Net(cls.sent, cls.sent // 2)
                return {"wlan0": net} if pernic else net

            @staticmethod
            def cpu_percent(interval=None):
                return 25.0

            @staticmethod
            def virtual_memory():
                return collections.namedtuple("VMem", "percent")(40.0)

        real, resources.psutil = resources.psutil, FakePsutil
        try:
            path = os.path.join(tempfile.mkdtemp(), "res.csv")
            sampler = resources.ResourceSampler(path, period=0.05)
            sampler.start()
            time.sleep(0.4)
            sampler.stop()
            sampler.join(2)
            summary = sampler.summary()
        finally:
            resources.psutil = real
        self.assertGreater(summary["samples"], 3)
        self.assertEqual((summary["cpu_pct"], summary["proc_cpu_pct"]), (25.0, 50.0))
        self.assertGreater(summary["tx_mbit_s"], summary["rx_mbit_s"])
        with open(path) as f:
            self.assertEqual(f.readline().strip().split(","), resources.ResourceSampler.COLUMNS)


class RunnerTest(unittest.TestCase):
    def test_blocks_have_unique_ids(self):
        import run_experiments as rx
        for name, block in rx.BLOCKS.items():
            runs, _ = block()
            ids = [rx.run_id(r) for r in runs]
            self.assertEqual(len(ids), len(set(ids)), name)
        self.assertEqual(len(rx.block_e3()[0]), 15 + 2 * 4 * 15)


if __name__ == "__main__":
    unittest.main()
