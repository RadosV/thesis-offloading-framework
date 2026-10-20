#!/usr/bin/env python3
"""
pc_edge_brain.py  -- runs on the PC / edge server
=================================================
Edge brain of the offloading framework.

Role: the offloaded ("cloud") brain. Receives frames from the Pi via ImageZMQ,
runs the perception module named in the frame header (the SAME code as the
Pi's local tier: ArUco or YOLO), and publishes the decision back over MQTT
together with the frame id and capture timestamp, so the Pi can measure the
true end-to-end decision age.

  python pc_edge_brain.py                        # broker = Pi at 192.168.2.106
  python pc_edge_brain.py --broker 192.168.2.106 --no-show

Design notes:
  * The MQTT network loop runs (loop_start): keep-alive pings and automatic reconnect.
    Without it, the broker drops the connection after ~90 s without traffic, and
    decisions are lost silently while the video stream keeps working.
  * Reply to the Pi AFTER processing a frame: natural flow control, no queue of
    old frames building up when inference is slower than the frame rate.
"""

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.dirname(HERE)):
    if os.path.isdir(os.path.join(_p, "framework")) and _p not in sys.path:
        sys.path.insert(0, _p)

import cv2
import imagezmq

from framework.perception import LEFT_X, REF_WIDTH, RIGHT_X, make_perception, steer_command
from framework.protocol import cloud_reply, decode_frame, mqtt_client, parse_frame_header
from framework.resources import ResourceSampler

MQTT_TOPIC = "rados/robot/cloud_cmd"   # hybrid manager subscribes to this


def show(frame, det, cmd):
    img = frame.copy()
    h, w = img.shape[:2]
    for x in (LEFT_X, RIGHT_X):
        x = int(x * w / REF_WIDTH)
        cv2.line(img, (x, 0), (x, h), (90, 90, 90), 1)
    if det.visible:
        cv2.line(img, (int(det.center_x), 0), (int(det.center_x), h), (0, 255, 0), 2)
    cv2.putText(img, "%s  %.0f ms" % (cmd, det.infer_ms), (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.imshow("edge brain", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        raise KeyboardInterrupt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--broker", default=os.environ.get("MQTT_BROKER", "192.168.2.106"),
                    help="MQTT broker = the Pi's IP (default %(default)s)")
    ap.add_argument("--port", type=int, default=5555, help="ImageZMQ port (default %(default)s)")
    ap.add_argument("--no-show", action="store_true", help="do not open the video window")
    ap.add_argument("--log-dir", default="pc_logs", help="server resource log (default %(default)s)")
    args = ap.parse_args()

    client = mqtt_client()
    client.on_connect = lambda c, u, *a: print("[MQTT] connected to %s" % args.broker)
    client.on_disconnect = lambda c, u, *a: print("[MQTT] disconnected - reconnecting ...")
    client.reconnect_delay_set(min_delay=1, max_delay=5)
    client.connect_async(args.broker, 1883, keepalive=10)
    client.loop_start()

    os.makedirs(args.log_dir, exist_ok=True)
    sampler = ResourceSampler(os.path.join(args.log_dir, time.strftime("res_pc_%Y%m%d_%H%M%S.csv")),
                              period=0.5)
    sampler.start()

    hub = imagezmq.ImageHub(open_port="tcp://*:%d" % args.port)
    perceptions, failed = {}, set()
    last_cmd = None
    print("Edge Brain active. Waiting for stream...")
    try:
        while True:
            msg, buf = hub.recv_jpg()
            det = cmd = frame = None
            try:
                header = parse_frame_header(msg)
                spec = header.get("p", "aruco")
                if spec not in perceptions and spec not in failed:
                    print("Loading perception %s ..." % spec)
                    try:
                        perceptions[spec] = make_perception(spec)
                    except Exception as e:
                        failed.add(spec)
                        print("!!! cannot load %s: %s" % (spec, e))
                if spec in perceptions:
                    frame = decode_frame(buf, header)
                    det = perceptions[spec].detect(frame)
                    cmd = steer_command(det, frame.shape[1])
                    client.publish(MQTT_TOPIC, cloud_reply(cmd, det, header))
            except Exception as e:
                print("!!! frame error: %s" % e)
            finally:
                hub.send_reply(b"OK")            # reply AFTER processing = flow control
            if cmd is None:
                continue
            if cmd != last_cmd:
                print("Decision: %-7s (%.1f ms)" % (cmd, det.infer_ms))
                last_cmd = cmd
            if not args.no_show:
                show(frame, det, cmd)
    except KeyboardInterrupt:
        pass
    finally:
        sampler.stop()
        client.loop_stop()
        client.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
