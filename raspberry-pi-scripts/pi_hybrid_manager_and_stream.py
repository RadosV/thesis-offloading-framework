#!/usr/bin/env python3
"""
pi_hybrid_manager_and_stream.py  -- runs on the Raspberry Pi
============================================================
VERSION 2 (framework-based, experiment-instrumented).

Role: the orchestrator. Once per control cycle (LOOP_HZ) it
  1. takes the newest camera frame (camera thread),
  2. offers it to the offload path (async streamer -> edge brain) and to the
     local perception worker - neither can block the control loop,
  3. releases cloud decisions through the network-impairment channel,
  4. asks the arbitration policy which brain drives (HYBRID),
  5. publishes the command to the motor controller and logs everything.

Environment knobs (all set by run_experiments.py):
  STRATEGY     LOCAL | CLOUD | HYBRID | IDLE (resources only)       default HYBRID
  POLICY       HYBRID arbitration: timeout:tau=0.5 | ewma:hi=300,lo=200 | cost:margin=30
  STANDBY      HYBRID: hot = local perception always runs, cold = only while LOCAL
  PERCEPTION   aruco | yolo:320 | yolo:224 | yolo:160                default aruco
  NET_PROFILE  stable | spike | ramp | outage | jitter | const:ms=100 (+ INJECT_MS base delay)
  JPEG_Q       uplink JPEG quality, 0 = raw frames                    default 80
  LOOP_HZ      control-loop rate; STREAM_HZ uplink rate limit        default 20 / LOOP_HZ
  MAX_S        auto-stop after this many seconds, 0 = until Ctrl-C
  ARRIVE_W     auto-stop when the target is >= ARRIVE_W px wide in ARRIVE_K new detections
  DET_HOLD_MS  keep the last direction this long when a frame has no detection; bridges
               YOLO dropouts, applied identically to BOTH tiers (0 = off, v1 behaviour)
  CAPTURE/CROP camera resolution and central crop (digital zoom), e.g. 1280x720 + 0.3
               narrows a 90 deg webcam to ~33 deg so a distant target covers more pixels
  RUN_ID, NET_COND, LOG_DIR, PC_IP, SOURCE (camera index or file), SEED, PREFLIGHT_S

Metrics logged per control cycle (run_<RUN_ID>.csv):
  cloud_stale_ms / local_stale_ms  age of the decision in use = now - capture time of its frame
  abs_error                        |target centre - 160 px| from the freshest detection known to
                                   the Pi (local or cloud), only on rows where that detection is new
Also: switch_<RUN_ID>.csv (every brain switch, RQ4), res_<RUN_ID>.csv (CPU/memory/network),
meta_<RUN_ID>.json (configuration + run summary, read by the runner).

Exit code 3 = run aborted by the pre-flight check (a required brain never answered).
"""

import csv
import json
import os
import signal
import socket
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.dirname(HERE)):          # flat layout on the Pi, or the repo layout
    if os.path.isdir(os.path.join(_p, "framework")) and _p not in sys.path:
        sys.path.insert(0, _p)

import imagezmq
import zmq

from framework.camera import CameraThread
from framework.netprofile import ImpairedChannel, make_profile
from framework.perception import canonical_spec, make_perception, steer_command, steer_payload
from framework.policies import make_policy
from framework.protocol import encode_frame, frame_header, mqtt_client, parse_cloud_reply
from framework.resources import ResourceSampler

# --- Fixed configuration ---
MQTT_BROKER = "localhost"
TOPIC_CONTROL = "rados/robot/control"
TOPIC_CLOUD_IN = "rados/robot/cloud_cmd"
FRAME_WIDTH, FRAME_HEIGHT = 320, 240
FRAME_CENTER = FRAME_WIDTH / 2.0
SEND_TIMEOUT_MS = 500     # uplink: no reply within this -> stream counted as down, socket rebuilt
CLOUD_STALE_S = 0.6       # CLOUD-only: stop when no cloud decision for this long (= motor watchdog)

# --- Experiment knobs (env vars) ---
_env = os.environ.get
STRATEGY = _env("STRATEGY", "HYBRID").upper()
POLICY_SPEC = _env("POLICY", "timeout:tau=0.5")
STANDBY = _env("STANDBY", "hot").lower()
PERCEPTION_SPEC = _env("PERCEPTION", "aruco")
NET_PROFILE = _env("NET_PROFILE", "stable")
INJECT_MS = float(_env("INJECT_MS", "0"))
NET_COND = _env("NET_COND", NET_PROFILE)
JPEG_Q = int(_env("JPEG_Q", "80"))
LOOP_HZ = float(_env("LOOP_HZ", "20"))
STREAM_HZ = float(_env("STREAM_HZ", str(LOOP_HZ)))
DET_HOLD_MS = float(_env("DET_HOLD_MS", "0"))
MAX_S = float(_env("MAX_S", "0"))
ARRIVE_W = float(_env("ARRIVE_W", "0"))
ARRIVE_K = int(_env("ARRIVE_K", "3"))
# "prop" = proportional steering (default); "relay" = the old left/forward/right
STEER_MODE = _env("STEER_MODE", "prop")
PREFLIGHT_S = float(_env("PREFLIGHT_S", "10"))
RUN_ID = _env("RUN_ID", time.strftime("%Y%m%d_%H%M%S"))
LOG_DIR = _env("LOG_DIR", "experiment_logs")
PC_IP = _env("PC_IP", "192.168.2.101")
SOURCE = _env("SOURCE", "0")
CAPTURE = _env("CAPTURE", "")
CROP = float(_env("CROP", "1"))
SEED = int(_env("SEED", "0"))

NEED_CLOUD = STRATEGY in ("CLOUD", "HYBRID")
NEED_LOCAL = STRATEGY in ("LOCAL", "HYBRID")
COLD = STRATEGY == "HYBRID" and STANDBY == "cold"

RUN_COLUMNS = ["t", "run_id", "strategy", "inject_ms", "net_cond", "brain", "marker_visible",
               "center_x", "abs_error", "command", "loop_dt",
               "t_rel", "policy", "perception", "net_delay_ms", "net_drop", "cloud_silence_s",
               "cloud_fid", "cloud_stale_ms", "cloud_infer_ms", "local_fid", "local_stale_ms",
               "local_infer_ms", "meas_src", "meas_fid", "meas_new", "meas_age_ms", "target_w",
               "stream_ok", "policy_state", "held"]
SWITCH_COLUMNS = ["t", "run_id", "from_brain", "to_brain", "cloud_silence_s",
                  "t_rel", "cloud_stale_ms", "local_stale_ms", "reason", "policy_state"]


def _f(x, nd=1):
    if x is None:
        return ""
    if x == float("inf"):
        return "inf"
    return "%.*f" % (nd, x)


def _age_ms(now, t_capture):
    return None if t_capture is None else (now - t_capture) * 1000.0


def _close_sender(sender):
    try:
        sender.close()
        return
    except Exception:
        pass
    sock = getattr(sender, "zmq_socket", None)
    if sock is not None:
        try:
            sock.close()
        except Exception:
            pass


class Streamer(threading.Thread):
    """Offload uplink. Sends the newest frame to the edge brain in its own thread,
    so a slow or dead PC never blocks the control loop (v1 streamed synchronously:
    while the PC was down every cycle waited for the send timeout and the local
    fallback ran at ~3 Hz). The edge brain replies after processing, which gives
    natural flow control: frames that arrive meanwhile are skipped, not queued."""

    def __init__(self, camera, address, perception_spec, jpeg_q, max_hz):
        super().__init__(daemon=True)
        self.camera, self.address = camera, address
        self.spec, self.jpeg_q = perception_spec, jpeg_q
        self.min_interval = 1.0 / max_hz if max_hz > 0 else 0.0
        self.ok = False
        self.n_sent = self.n_fail = self.bytes_sent = 0
        self._stop_evt = threading.Event()

    def _connect(self):
        sender = imagezmq.ImageSender(connect_to=self.address)
        sock = getattr(sender, "zmq_socket", None)
        if sock is not None:
            sock.setsockopt(zmq.RCVTIMEO, SEND_TIMEOUT_MS)
            sock.setsockopt(zmq.SNDTIMEO, SEND_TIMEOUT_MS)
            sock.setsockopt(zmq.LINGER, 0)
        return sender

    def run(self):
        sender = self._connect()
        last_fid = 0
        while not self._stop_evt.is_set():
            item = self.camera.wait_new(last_fid, 0.5)
            if item is None:
                continue
            t_send = time.time()
            fid, t_cap, frame = item
            last_fid = fid
            payload, enc = encode_frame(frame, self.jpeg_q)
            try:
                sender.send_jpg(frame_header(fid, t_cap, enc, frame.shape, self.spec), payload)
                self.ok = True
                self.n_sent += 1
                self.bytes_sent += len(payload)
            except Exception:
                # PC gone or slow: drop this frame, rebuild the REQ socket for a reconnect
                self.ok = False
                self.n_fail += 1
                _close_sender(sender)
                sender = self._connect()
            rest = self.min_interval - (time.time() - t_send)
            if rest > 0:
                self._stop_evt.wait(rest)
        _close_sender(sender)

    def stop(self):
        self._stop_evt.set()


class LocalWorker(threading.Thread):
    """Local perception tier: processes the newest frame whenever it is idle.
    `active` is cleared while a cold-standby hybrid drives on the cloud."""

    def __init__(self, camera, perception):
        super().__init__(daemon=True)
        self.camera, self.perception = camera, perception
        self.active = threading.Event()
        self.active.set()
        self.n_done, self.sum_ms = 0, 0.0
        self._latest = None                     # (fid, t_capture, Detection)
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()

    def run(self):
        last_fid = 0
        while not self._stop_evt.is_set():
            if not self.active.wait(0.1):
                continue
            item = self.camera.wait_new(last_fid, 0.5)
            if item is None:
                continue
            fid, t_cap, frame = item
            last_fid = fid
            det = self.perception.detect(frame)
            with self._lock:
                self._latest = (fid, t_cap, det)
            self.n_done += 1
            self.sum_ms += det.infer_ms

    def get(self):
        with self._lock:
            return self._latest

    def stop(self):
        self._stop_evt.set()


class HybridManager:
    def __init__(self):
        spec = canonical_spec(PERCEPTION_SPEC)
        self.perception = make_perception(spec) if NEED_LOCAL else None
        self.policy = make_policy(POLICY_SPEC) if STRATEGY == "HYBRID" else None
        self.channel = ImpairedChannel(make_profile(NET_PROFILE, base_ms=INJECT_MS, seed=SEED))

        # --- state ---
        self.brain = "LOCAL" if STRATEGY == "HYBRID" else STRATEGY
        self.cloud = None                     # latest released cloud decision (dict)
        self.last_release = None
        self.local_since = 0.0                # cold standby: ignore local results older than this
        self.meas_key = None
        self.arrive_n = 0
        self.last_seen = {"CLOUD": None, "LOCAL": None}   # last frame with a detection, per tier
        self.last_cmd = {"CLOUD": "stop", "LOCAL": "stop"}
        self.t0 = None
        self.n_released, self.cloud_infer_sum = 0, 0.0
        self.stats = dict(stop_reason="ctrl-c", aborted=None, t_arrive=None, n_cycles=0,
                          n_switches=0, n_held=0, loop_dt_sum=0.0)

        # --- logs ---
        path = lambda kind, ext="csv": os.path.join(LOG_DIR, "%s_%s.%s" % (kind, RUN_ID, ext))
        self.meta_path = path("meta", "json")
        self.log_file = open(path("run"), "w", newline="")
        self.log = csv.writer(self.log_file)
        self.log.writerow(RUN_COLUMNS)
        self.sw_file = open(path("switch"), "w", newline="")
        self.sw = csv.writer(self.sw_file)
        self.sw.writerow(SWITCH_COLUMNS)
        self.sw_file.flush()

        # --- MQTT (broker on the Pi; cloud decisions come in through the impairment channel) ---
        self.client = mqtt_client()
        self.client.on_connect = lambda c, u, *a: c.subscribe(TOPIC_CLOUD_IN)
        self.client.on_message = lambda c, u, msg: self.channel.push(parse_cloud_reply(msg.payload))
        self.client.connect(MQTT_BROKER, 1883, 60)
        self.client.loop_start()

        # --- threads ---
        self.camera = CameraThread(SOURCE, FRAME_WIDTH, FRAME_HEIGHT,
                                   capture=CAPTURE or None, crop=CROP)
        self.streamer = (Streamer(self.camera, "tcp://%s:5555" % PC_IP, spec, JPEG_Q, STREAM_HZ)
                         if NEED_CLOUD else None)
        self.worker = LocalWorker(self.camera, self.perception) if NEED_LOCAL else None
        if COLD:
            self.worker.active.clear()
        self.sampler = ResourceSampler(path("res"))
        for th in self._threads():
            th.start()

    def _threads(self):
        return [th for th in (self.camera, self.streamer, self.worker, self.sampler) if th]

    # --- pre-flight: do not start the run until every required brain answers ---
    def warm_up(self):
        t_end = time.time() + PREFLIGHT_S
        while time.time() < t_end:
            if self.camera.error:
                return "camera: " + self.camera.error
            cloud_ok = not NEED_CLOUD or self.channel.n_arrived > 0
            local_ok = not NEED_LOCAL or COLD or self.worker.get() is not None
            if cloud_ok and local_ok:
                return None
            self.client.publish(TOPIC_CONTROL, "stop")
            time.sleep(0.05)
        if NEED_CLOUD and self.channel.n_arrived == 0:
            return "no decision from the edge brain (pc_edge_brain.py running? PC_IP=%s)" % PC_IP
        return "local perception produced no result"

    def run(self):
        self.stats["aborted"] = self.warm_up()
        if self.stats["aborted"]:
            print("!!! RUN ABORTED: %s" % self.stats["aborted"])
            return 3
        self.t0 = time.time()
        self.channel.start(self.t0)
        print("--- Running. Ctrl-C to end the run. ---")
        period = 1.0 / LOOP_HZ
        next_t = t_prev = time.time()
        while True:
            now = time.time()
            if MAX_S > 0 and now - self.t0 >= MAX_S:
                self.stats["stop_reason"] = "max_s"
                break
            if not self.step(now, now - t_prev):
                break
            t_prev = now
            next_t += period
            delay = next_t - time.time()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.time()             # overrun: do not try to catch up
        return 0

    def step(self, now, loop_dt):
        t_rel = now - self.t0

        # 1. cloud decisions that are due (after injected delay / drops)
        for rep in self.channel.pop_due(now):
            fid, cur = rep.get("fid"), (self.cloud or {}).get("fid")
            if fid is not None and cur is not None and fid <= cur:
                continue                          # overtaken by a newer decision (jitter)
            self.cloud = rep
            self.last_release = now
            self.n_released += 1
            self.cloud_infer_sum += rep.get("infer_ms") or 0.0
            if self.policy:
                self.policy.on_cloud_message(now)
        silence = now - self.last_release if self.last_release else float("inf")
        cloud_stale = _age_ms(now, self.cloud.get("t_cap")) if self.cloud else None
        loc = self.worker.get() if self.worker else None
        if loc is not None and loc[1] < self.local_since:
            loc = None                            # cold standby: result from before activation
        local_stale = _age_ms(now, loc[1]) if loc else None

        # 2. arbitration
        if STRATEGY == "HYBRID":
            brain = self.policy.decide(now, cloud_stale, local_stale, silence, self.brain)
        else:
            brain = STRATEGY
        if brain != self.brain:
            self._switch(now, t_rel, brain, silence, cloud_stale, local_stale)

        # 3. command; DET_HOLD_MS bridges short detection dropouts by repeating the last
        #    direction, identically for both tiers, instead of stopping on every lost frame
        if brain == "CLOUD":
            usable = self.cloud is not None and (STRATEGY == "HYBRID" or silence < CLOUD_STALE_S)
            seen = bool(usable and self.cloud.get("vis"))
            if not usable:
                cmd = "stop"
            elif STEER_MODE == "prop":
                cmd = steer_payload(self.cloud.get("cx"), FRAME_WIDTH) if seen else "stop"
            else:
                cmd = self.cloud["cmd"]
        else:
            seen = bool(loc and loc[2].visible)
            if not loc:
                cmd = "stop"
            elif STEER_MODE == "prop":
                cmd = steer_payload(loc[2].center_x, FRAME_WIDTH) if seen else "stop"
            else:
                cmd = steer_command(loc[2], FRAME_WIDTH)
        held = 0
        if seen:
            self.last_seen[brain], self.last_cmd[brain] = now, cmd
        elif (DET_HOLD_MS > 0 and self.last_seen[brain] is not None
              and (now - self.last_seen[brain]) * 1000.0 <= DET_HOLD_MS):
            # Never keep TURNING blind: a repeated turn compounds until the target
            # leaves the frame. Hold the heading instead so the detector can reacquire.
            last = self.last_cmd[brain]
            if last.startswith("go:"):
                last = "go:0.000"          # hold the heading, never a blind turn
            elif last in ("left", "right"):
                last = "forward"
            cmd, held = last, 1
            self.stats["n_held"] += 1
        self.client.publish(TOPIC_CONTROL, cmd)

        # 4. tracking measurement: freshest detection known to the Pi
        meas = self._freshest(loc)
        new = meas is not None and (meas[0], meas[1]) != self.meas_key
        if new:
            self.meas_key = (meas[0], meas[1])
        visible = bool(meas and meas[3])
        width = meas[5] if visible else None
        cx = meas[4] if (new and visible) else None
        abs_err = abs(cx - FRAME_CENTER) if cx is not None else None

        # 5. log
        delay_ms, dropping = self.channel.nominal(now)
        policy_state = self.policy.state() if self.policy else ""
        self.log.writerow([
            "%.4f" % now, RUN_ID, STRATEGY, INJECT_MS, NET_COND, self.brain, int(visible),
            _f(cx), _f(abs_err), cmd, "%.4f" % loop_dt,
            "%.3f" % t_rel, POLICY_SPEC if self.policy else "", PERCEPTION_SPEC, _f(delay_ms),
            int(dropping), _f(silence, 4),
            (self.cloud or {}).get("fid", ""), _f(cloud_stale), _f((self.cloud or {}).get("infer_ms")),
            loc[0] if loc else "", _f(local_stale), _f(loc[2].infer_ms) if loc else "",
            meas[0] if meas else "", meas[1] if meas else "", int(new),
            _f(_age_ms(now, meas[2])) if meas else "", _f(width),
            int(self.streamer.ok) if self.streamer else "", policy_state, held])
        self.stats["n_cycles"] += 1
        self.stats["loop_dt_sum"] += loop_dt

        # 6. arrival: target wide enough in ARRIVE_K consecutive new detections
        if ARRIVE_W > 0 and new:
            self.arrive_n = self.arrive_n + 1 if (width is not None and width >= ARRIVE_W) else 0
            if self.arrive_n >= ARRIVE_K:
                self.stats.update(stop_reason="arrived", t_arrive=round(t_rel, 3))
                print("--- Target reached after %.2f s ---" % t_rel)
                return False
        return True

    def _freshest(self, loc):
        """(source, fid, t_capture, visible, center_x, width) of the newest detection."""
        cands = []
        c = self.cloud
        if c and c.get("t_cap") is not None and "vis" in c:
            cands.append(("cloud", c.get("fid"), c["t_cap"], c["vis"], c.get("cx"), c.get("w")))
        if loc:
            d = loc[2]
            cands.append(("local", loc[0], loc[1], d.visible, d.center_x, d.width))
        return max(cands, key=lambda m: m[2]) if cands else None

    def _switch(self, now, t_rel, new, silence, cloud_stale, local_stale):
        reason = self.policy.reason if self.policy else "strategy"
        self.sw.writerow(["%.4f" % now, RUN_ID, self.brain, new, _f(silence, 4), "%.3f" % t_rel,
                          _f(cloud_stale), _f(local_stale), reason,
                          self.policy.state() if self.policy else ""])
        self.sw_file.flush()
        print("[SWITCH] %s -> %s (%s, silence %.3fs)" % (self.brain, new, reason, silence))
        if COLD:
            if new == "LOCAL":
                self.local_since = now
                self.worker.active.set()
            else:
                self.worker.active.clear()
        self.brain = new
        self.stats["n_switches"] += 1

    def close(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)     # a second Ctrl-C must not break cleanup
        for _ in range(3):
            self.client.publish(TOPIC_CONTROL, "stop")
            time.sleep(0.02)
        for th in self._threads():
            th.stop()
        for th in self._threads():
            th.join(timeout=2.0)
        self.client.loop_stop()
        self.client.disconnect()
        self.log_file.close()
        self.sw_file.close()
        self.write_meta()
        print("Saved: %s (+ run/switch/res CSVs)" % self.meta_path)

    def write_meta(self):
        duration = time.time() - self.t0 if self.t0 else 0.0
        stats = dict(self.stats)
        n = max(stats["n_cycles"], 1)
        stats["mean_loop_dt"] = round(stats.pop("loop_dt_sum") / n, 4)
        meta = dict(version=2, run_id=RUN_ID, host=socket.gethostname(), strategy=STRATEGY,
                    policy=POLICY_SPEC if self.policy else None,
                    standby=STANDBY if STRATEGY == "HYBRID" else None,
                    perception=canonical_spec(PERCEPTION_SPEC), net_profile=NET_PROFILE,
                    inject_ms=INJECT_MS, net_cond=NET_COND, jpeg_q=JPEG_Q, loop_hz=LOOP_HZ,
                    stream_hz=STREAM_HZ, max_s=MAX_S, arrive_w=ARRIVE_W, arrive_k=ARRIVE_K,
                    det_hold_ms=DET_HOLD_MS, seed=SEED, source=SOURCE, capture=CAPTURE, crop=CROP,
                    capture_size=list(self.camera.capture_size), pc_ip=PC_IP, t_start=self.t0,
                    duration_s=round(duration, 3), **stats)
        meta["cloud"] = dict(arrived=self.channel.n_arrived, dropped=self.channel.n_dropped,
                             released=self.n_released,
                             mean_infer_ms=round(self.cloud_infer_sum / max(self.n_released, 1), 2))
        if self.streamer:
            meta["stream"] = dict(sent=self.streamer.n_sent, failed=self.streamer.n_fail,
                                  bytes=self.streamer.bytes_sent,
                                  mbit_s=round(self.streamer.bytes_sent * 8 / max(duration, 1e-6) / 1e6, 3))
        if self.worker:
            meta["local"] = dict(done=self.worker.n_done,
                                 mean_infer_ms=round(self.worker.sum_ms / max(self.worker.n_done, 1), 2),
                                 backend=getattr(self.perception, "backend", "cv2"))
        meta["resources"] = self.sampler.summary()
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2)


def idle_run():
    """E1 baseline: only sample resources (no camera, no MQTT) for MAX_S seconds."""
    sampler = ResourceSampler(os.path.join(LOG_DIR, "res_%s.csv" % RUN_ID))
    sampler.start()
    t0, reason = time.time(), "max_s"
    try:
        while MAX_S <= 0 or time.time() - t0 < MAX_S:
            time.sleep(0.2)
    except KeyboardInterrupt:
        reason = "ctrl-c"
    sampler.stop()
    sampler.join(timeout=2.0)
    meta = dict(version=2, run_id=RUN_ID, host=socket.gethostname(), strategy="IDLE", t_start=t0,
                duration_s=round(time.time() - t0, 3), stop_reason=reason, aborted=None,
                resources=sampler.summary())
    with open(os.path.join(LOG_DIR, "meta_%s.json" % RUN_ID), "w") as f:
        json.dump(meta, f, indent=2)
    return 0


def _raise_interrupt(signum, frame):
    raise KeyboardInterrupt


def main():
    signal.signal(signal.SIGTERM, _raise_interrupt)
    os.makedirs(LOG_DIR, exist_ok=True)
    print("=== EXPERIMENT RUN (v2) ===")
    print("  strategy  : %s   policy: %s   standby: %s" % (
        STRATEGY, POLICY_SPEC if STRATEGY == "HYBRID" else "-", STANDBY if STRATEGY == "HYBRID" else "-"))
    print("  perception: %s   net: %s + %g ms   jpeg: %d" % (PERCEPTION_SPEC, NET_PROFILE, INJECT_MS, JPEG_Q))
    print("  run_id    : %s   logs: %s" % (RUN_ID, LOG_DIR))
    if STRATEGY == "IDLE":
        return idle_run()
    manager = HybridManager()
    code = 0
    try:
        code = manager.run()
    except KeyboardInterrupt:
        print("\nStopping run...")
    finally:
        manager.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
