"""
protocol.py  -- wire formats between the Pi and the edge server
===============================================================
Uplink (ImageZMQ REQ/REP, TCP 5555): every frame travels as an ImageZMQ "jpg"
message = text header + byte buffer. The header is compact JSON:
    {"fid": 17, "t": 1779730845.2470, "enc": "jpg", "shape": [240, 320, 3], "p": "yolo:320,cls=11,conf=0.4"}
  fid / t  frame id and capture time on the Pi clock; they come back with the
           decision, so the Pi measures the true decision age without any
           clock synchronisation between the machines
  enc      "jpg" (JPEG_Q > 0) or "raw" (uncompressed BGR)
  p        canonical perception spec, so the edge brain runs the same module

Downlink (MQTT rados/robot/cloud_cmd): compact JSON
    {"cmd": "left", "fid": 17, "t_cap": 1779730845.247, "infer_ms": 12.3, "vis": 1, "cx": 101.5, "w": 42.0}
  A plain command string is also accepted.
"""

import json

import cv2
import numpy as np
import paho.mqtt.client as mqtt

DEFAULT_SHAPE = (240, 320, 3)


def mqtt_client():
    """paho-mqtt client for both the 1.x and 2.x API (callbacks use *args)."""
    try:
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except AttributeError:
        return mqtt.Client()


def encode_frame(frame, jpeg_q):
    if jpeg_q and int(jpeg_q) > 0:
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_q)])
        if ok:
            return buf.tobytes(), "jpg"
    return np.ascontiguousarray(frame).tobytes(), "raw"


def frame_header(fid, t_capture, enc, shape, perception_spec):
    return json.dumps({"fid": fid, "t": round(t_capture, 4), "enc": enc,
                       "shape": list(shape), "p": perception_spec}, separators=(",", ":"))


def parse_frame_header(msg):
    try:
        header = json.loads(msg)
    except (TypeError, ValueError):
        return {}                  # no JSON header (e.g. only a host name)
    return header if isinstance(header, dict) else {}


def decode_frame(buf, header):
    data = np.frombuffer(getattr(buf, "buffer", buf), dtype=np.uint8)
    if header.get("enc") == "jpg":
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data.reshape(header.get("shape") or DEFAULT_SHAPE)


def cloud_reply(cmd, det, header):
    rep = {"cmd": cmd, "fid": header.get("fid"), "t_cap": header.get("t"),
           "infer_ms": round(det.infer_ms, 2), "vis": int(det.visible)}
    if det.visible:
        rep["cx"] = round(det.center_x, 1)
        rep["w"] = round(det.width, 1)
    return json.dumps(rep, separators=(",", ":"))


def parse_cloud_reply(payload):
    text = payload.decode(errors="replace") if isinstance(payload, (bytes, bytearray)) else str(payload)
    text = text.strip()
    if text.startswith("{"):
        try:
            rep = json.loads(text)
        except ValueError:
            rep = None
        if isinstance(rep, dict) and "cmd" in rep:
            return rep
    return {"cmd": text}
