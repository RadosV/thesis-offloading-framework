#!/usr/bin/env python3
"""
Experiment runner, VERSION 2. Run ON THE PI.
Start pc_edge_brain.py on the PC first (and motor_controller.py on the Pi for driving runs).

  python3 run_experiments.py e1        # E1  power/CPU per mode, car on a box, 60 s runs (RQ2)
  python3 run_experiments.py e2        # E2  YOLO latency sweep + LOCAL-YOLO baseline (RQ1, RQ2)
  python3 run_experiments.py e3        # E3  switching policies, YOLO (RQ3, RQ4)
  python3 run_experiments.py e3aruco   # E3' RQ3 redo with ArUco (RQ3)
  python3 run_experiments.py e4        # E4  timeout sweep (RQ4)
  python3 run_experiments.py e5        # E5  real tc/netem validation (optional, uses sudo)
  python3 run_experiments.py e2pilot   # E2 shape check: one run per cell (9 runs)
  python3 run_experiments.py e2 --dry  # only print the run plan
  python3 run_experiments.py single STRATEGY=HYBRID POLICY=cost:margin=30 PERCEPTION=yolo:320 MAX_S=20

* Run order is shuffled (fixed seed) so battery drain, lighting and order
  effects spread over all conditions instead of hitting one block.
* The runner resumes: runs whose meta_<RUN_ID>.json exists and was not aborted
  are skipped, so a block can be split over several sessions.
* Network disturbances are scripted (NET_PROFILE) - never kill the edge brain by hand.
* Per run: ENTER = launch, s = skip, q = quit; Ctrl-C ends a run early.
  Afterwards score it (p/f + optional note) -> experiment_logs/runs_notes.csv.
"""
import csv
import json
import os
import random
import subprocess
import sys
import time
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.dirname(HERE)):
    if os.path.isdir(os.path.join(_p, "framework")) and _p not in sys.path:
        sys.path.insert(0, _p)

from framework.spec import label

MANAGER = "pi_hybrid_manager_and_stream.py"
LOG_DIR = os.environ.get("LOG_DIR", os.path.join(HERE, "experiment_logs"))
NOTES = os.path.join(LOG_DIR, "runs_notes.csv")
SEED = 2026

# ---- Settings to fix after E1 / calibration (see TODO.md) -------------------
# Measured 16.09: YOLOv5n per frame - Pi 249 / 472 / 991 ms at 160 / 224 / 320,
# PC 7 / 13 / 26 ms, ArUco 6.3 ms on the Pi. Only @320 sees the sign from a
# useful distance, so the driving blocks use @320; 160/224 stay in the E1 bench.
# Local decisions are therefore ~1.5 s old and cloud decisions ~0.1 s, which puts
# the expected crossover near 1.3 s - hence the sweep and thresholds below.
YOLO = "yolo:320"
# Target width (px) that counts as 'arrived' (0 = no auto-stop, run ends at MAX_S).
# Re-measured 20.09: reliable detection at ~195 cm (56-59 px, score 0.47-0.55, 5/5 frames), so the
# 17.09 "135 cm limit" was too conservative; start mark moved to 180 cm. Measured 17.09:
# 86 px at 135 cm,
# 102 px at 110 cm (old start mark, score ~0.87), 185 px at 50 cm (score 0.32, last reliable distance -
# closer than that the sign leaves the top of the narrow view). Arrival at 170 px ~ 55 cm; the
# ultrasonic reflex at 30 cm is the backstop. The usable window is ~63 px (180 cm start) to 185 px (50 cm,
# last distance the raised sign stays in the narrow view), so arrival is declared at 150 px ~ 70 cm:
# high enough not to fire at the start line, low enough to fire before the sign leaves the view.
# The manual p/f score stays the authoritative success metric.
# 150 px ~ 78 cm. Measured 21.09 over 18 runs: successful runs reach 250-288 px (41-47 cm),
# marginal ones 170-183 px, genuine failures top out at ~90 px (~130 cm). 150 separates them
# and fires on every run that gets near the sign; 200/250 lost t_arrive on marginal runs.
# ArUco measured 29.09: 37 px at the 180 cm mark (scale ~6700 px*cm), x~158, y~84.
# 86 px ~ 78 cm, the SAME physical arrival point as the stop sign at 150 px - so E3'
# and E3 declare arrival at the same distance and the blocks stay comparable.
ARRIVE_W = {"aruco": 86, "yolo": 150}
EWMA_HI, EWMA_LO = 1300, 900          # ms; E2 (21.09) measured the crossover at ~1100 ms, which these bracket
# Disturbance windows must cover the APPROACH. E2 runs arrive in ~4.1 s at 0 ms, so the
# original start=3 windows only delayed decisions that landed after arrival - E3 would have
# come out flat, like the first E2 block. Now they open at t=1 s, mid-approach.
SPIKE = "spike:ms=1500,start=1,end=3.5"  # 1500 ms > the ~1100 ms crossover: cloud should lose
OUTAGE = "outage:start=1,end=3"
JITTER = "jitter:mean=150,sd=80"
MAX_S = 25                            # auto-stop of a driving run (local YOLO is slow)
SECONDS_PER_RUN = 45                  # incl. resetting the car (time estimate only)
NETEM = "delay 100ms 30ms loss 5%"
NETEM_IFACE = "wlan0"
# Camera: the webcam is 90 deg wide, which makes a distant target tiny. Grab a larger
# frame and keep only the middle (digital zoom) - set after checking with
# bench_inference.py --live --capture ... --crop ...  ("" / 1.0 = raw wide view).
CAPTURE, CROP = "1280x720", 0.3
COMMON = {"JPEG_Q": 80, "LOOP_HZ": 20, "CAPTURE": CAPTURE, "CROP": CROP}
# -----------------------------------------------------------------------------

POLICY = {"P0": "timeout:tau=0.5",
          "P1": "ewma:hi=%d,lo=%d" % (EWMA_HI, EWMA_LO),
          "P2": "cost:margin=30"}


def drive(perception, **env):
    """Environment for a driving run with the given perception module."""
    yolo = perception.startswith("yolo")
    out = dict(COMMON, PERCEPTION=perception, MAX_S=MAX_S,
               ARRIVE_W=ARRIVE_W[perception.split(":")[0]],
               ARRIVE_K=1 if yolo else 3,          # local YOLO delivers ~1 detection/s: 2 consecutive never fires
               DET_HOLD_MS=1000 if yolo else 0)    # bridge YOLO detection dropouts
    out.update(env)
    return out


def cell(exp, reps, env):
    return [{"exp": exp, "rep": r, "env": {k: str(v) for k, v in env.items()}}
            for r in range(1, reps + 1)]


# --- Experiment blocks: (runs, options) ---
def block_e1():
    y = YOLO
    modes = [dict(STRATEGY="IDLE"),
             dict(STRATEGY="LOCAL", PERCEPTION="aruco"),
             dict(STRATEGY="LOCAL", PERCEPTION="yolo:160"),
             dict(STRATEGY="LOCAL", PERCEPTION="yolo:320"),
             dict(STRATEGY="CLOUD", PERCEPTION="aruco", JPEG_Q=0),
             dict(STRATEGY="CLOUD", PERCEPTION="aruco"),
             dict(STRATEGY="CLOUD", PERCEPTION=y),
             dict(STRATEGY="HYBRID", PERCEPTION=y, POLICY=POLICY["P0"], STANDBY="hot"),
             dict(STRATEGY="HYBRID", PERCEPTION=y, POLICY=POLICY["P0"], STANDBY="cold")]
    runs = []
    for m in modes:
        runs += cell("e1", 3, dict(COMMON, MAX_S=60, **m))   # 60 s: the meter shows Wh to only
                                                           # 2 dp, so a longer window is needed
    return runs, dict(ask_power=True, score=False, intro=(
        "E1 (stationary): car on a box with free wheels, or motor_controller.py not running.\n"
        "USB meter between power bank and Pi. Per run: RESET the meter, ENTER, wait 60 s,"
        " then read accumulated Wh and multiply by 60 -> average watts. Type that." \
        "\n" "Keep the camera on the target (marker / stop sign), scene identical."))


def block_e2():
    runs = []
    # 0-1600 brackets the arrival-time crossover (LOCAL ~6.0 s); 2000/2400 reach the
    # success collapse. Success rate alone saturates at 100% below ~2 s - t_arrive is the metric.
    for ms in (0, 200, 400, 800, 1200, 1600, 2000, 2400):
        runs += cell("e2", 10, drive(YOLO, STRATEGY="CLOUD", NET_PROFILE="stable", INJECT_MS=ms))
    runs += cell("e2", 15, drive(YOLO, STRATEGY="LOCAL"))
    return runs, dict(intro="E2: usual start mark, target = printed stop sign. Perception %s." % YOLO)


def block_e2pilot():
    """One run per E2 cell - check the shape of the curve before committing 95 runs.
    Uses exp id 'e2p' so these runs never satisfy the real e2 block's resume check."""
    runs = []
    for ms in (0, 200, 400, 800, 1200, 1600, 2000, 2400):
        runs += cell("e2p", 2, drive(YOLO, STRATEGY="CLOUD", NET_PROFILE="stable", INJECT_MS=ms))
    runs += cell("e2p", 2, drive(YOLO, STRATEGY="LOCAL"))
    return runs, dict(intro="E2 PILOT: 18 runs - every condition once from each side.")


def block_e2check():
    """3 reps of the three cells that decide the crossover: is the LOCAL baseline able to
    finish the task at all, and does arrival fire at +800/+1200? Run ids 'e2c_'."""
    runs = cell("e2c", 3, drive(YOLO, STRATEGY="CLOUD", NET_PROFILE="stable", INJECT_MS=800))
    runs += cell("e2c", 3, drive(YOLO, STRATEGY="CLOUD", NET_PROFILE="stable", INJECT_MS=1200))
    runs += cell("e2c", 3, drive(YOLO, STRATEGY="LOCAL"))
    runs += cell("e2c", 3, drive(YOLO, STRATEGY="CLOUD", NET_PROFILE="stable", INJECT_MS=2400))
    return runs, dict(intro="E2 CHECK: 12 runs - LOCAL baseline, the crossover cells "
                            "and +2400 ms. Start marks are now 20 cm off the centreline.")


def block_e3pilot():
    """10 runs: do the disturbances actually hurt pure CLOUD, and does HYBRID recover?
    One run per side per condition. Run ids 'e3p_'. Check this before the 135-run block."""
    runs = cell("e3p", 2, drive(YOLO, STRATEGY="LOCAL"))
    for prof in (SPIKE, OUTAGE):
        runs += cell("e3p", 2, drive(YOLO, STRATEGY="CLOUD", NET_PROFILE=prof))
        runs += cell("e3p", 2, drive(YOLO, STRATEGY="HYBRID", POLICY=POLICY["P0"], NET_PROFILE=prof))
    return runs, dict(intro="E3 PILOT: 10 runs. Same 20 cm marks as E2. Disturbances are scripted "
                            "- never kill the edge brain by hand.")


def _policy_block(exp, perception, reps, policies, profiles):
    runs = cell(exp, reps, drive(perception, STRATEGY="LOCAL"))       # pooled network-free baseline
    for prof in profiles:
        runs += cell(exp, reps, drive(perception, STRATEGY="CLOUD", NET_PROFILE=prof))
        for p in policies:
            runs += cell(exp, reps, drive(perception, STRATEGY="HYBRID", POLICY=POLICY[p],
                                          NET_PROFILE=prof))
    return runs


def block_e3():
    return (_policy_block("e3", YOLO, 15, ("P0", "P1", "P2"), (SPIKE, OUTAGE)),
            dict(intro="E3: same 20 cm marks as E2. Disturbances are scripted - "
                       "never kill the edge brain by hand."))


def block_e3short():
    """Every E3 condition once from each side: 18 runs. Includes P1 and P2, which the
    10-run e3pilot did not test. Run ids 'e3s_'."""
    return (_policy_block("e3s", YOLO, 2, ("P0", "P1", "P2"), (SPIKE, OUTAGE)),
            dict(intro="E3 SHORT: 18 runs, every condition once per side. Same 20 cm marks. "
                       "Disturbances are scripted - never kill the edge brain by hand."))


def block_e3aruco():
    return (_policy_block("e3a", "aruco", 10, ("P0", "P2"), (SPIKE, OUTAGE)),
            dict(intro="E3': ArUco marker as target. Disturbances are scripted."))


def block_e3ashort():
    """E3' shape check: every condition once from each side, 14 runs. Run ids 'e3as_'.
    Expect the policies to make little difference - ArUco is cheap, so LOCAL was never slow."""
    return (_policy_block("e3as", "aruco", 2, ("P0", "P2"), (SPIKE, OUTAGE)),
            dict(intro="E3' SHORT: 14 runs, ArUco marker as target. Same 20 cm marks. "
                       "Disturbances are scripted - never kill the edge brain by hand."))


def block_e4():
    runs = []
    for tau in (0.2, 0.35, 0.5):
        for prof in (OUTAGE, JITTER):
            runs += cell("e4", 8, drive(YOLO, STRATEGY="HYBRID", POLICY="timeout:tau=%g" % tau,
                                        NET_PROFILE=prof))
    return runs, {}


def block_e4pilot():
    """E4 shape check: every tau x profile once from each side, 12 runs. Run ids 'e4p_'.
    H4 expects detection delay ~ tau + T_loop, and more switching at small tau under jitter."""
    runs = []
    for tau in (0.2, 0.35, 0.5):
        for prof in (OUTAGE, JITTER):
            runs += cell("e4p", 2, drive(YOLO, STRATEGY="HYBRID", POLICY="timeout:tau=%g" % tau,
                                         NET_PROFILE=prof))
    return runs, dict(intro="E4 PILOT: 12 runs, tau sweep under outage and jitter. Same 20 cm marks.")


def block_e5():
    runs = cell("e5", 10, drive(YOLO, STRATEGY="CLOUD", NETEM=1))
    runs += cell("e5", 10, drive(YOLO, STRATEGY="HYBRID", POLICY=POLICY["P2"], NETEM=1))
    return runs, dict(netem=True, intro=(
        "E5: real impairment with tc/netem on %s (%s).\n"
        "It also affects your SSH session to the Pi." % (NETEM_IFACE, NETEM)))


BLOCKS = {"e1": block_e1, "e2": block_e2, "e2pilot": block_e2pilot, "e2check": block_e2check,
          "e3pilot": block_e3pilot, "e3short": block_e3short,
          "e4pilot": block_e4pilot,
          "e3ashort": block_e3ashort,
          "e3": block_e3, "e3aruco": block_e3aruco,
          "e4": block_e4, "e5": block_e5}


# --- Runner ---
def run_id(run):
    e = run["env"]
    strategy = e.get("STRATEGY", "HYBRID")
    parts = [run["exp"], strategy]
    if strategy == "HYBRID":
        parts.append(label(e.get("POLICY", "timeout:tau=0.5")))
        if e.get("STANDBY", "hot") != "hot":
            parts.append(e["STANDBY"])
    if strategy != "IDLE":
        parts.append(label(e.get("PERCEPTION", "aruco")))
        if strategy != "LOCAL":
            parts.append(label(e.get("NET_PROFILE", "stable")))
    if float(e.get("INJECT_MS", 0)):
        parts.append("inj%d" % float(e["INJECT_MS"]))
    if e.get("JPEG_Q") == "0":
        parts.append("raw")
    if e.get("NETEM"):
        parts.append("netem")
    return "_".join(parts) + "_r%d" % run["rep"]


def find_manager():
    for path in (os.path.join(HERE, MANAGER), os.path.join(HERE, "..", "raspberry-pi-scripts", MANAGER)):
        if os.path.exists(path):
            return os.path.abspath(path)
    sys.exit("cannot find %s next to the runner or in ../raspberry-pi-scripts/" % MANAGER)


def read_meta(rid):
    try:
        with open(os.path.join(LOG_DIR, "meta_%s.json" % rid)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def is_done(rid):
    meta = read_meta(rid)
    return meta is not None and not meta.get("aborted")


def launch(run):
    env = dict(os.environ)
    env.update(run["env"])
    env.update(RUN_ID=run["id"], LOG_DIR=LOG_DIR, SEED=str(zlib.crc32(run["id"].encode()) & 0xFFFF))
    proc = subprocess.Popen([sys.executable, find_manager()], env=env)
    interrupts = 0
    while True:
        try:
            return proc.wait()
        except KeyboardInterrupt:
            # the manager got the same Ctrl-C and is shutting down cleanly; a 2nd one terminates it
            interrupts += 1
            if interrupts >= 2:
                proc.terminate()


def print_summary(meta):
    if not meta:
        print("    (no meta file written)")
        return
    parts = ["stop=%s" % meta.get("stop_reason")]
    if meta.get("t_arrive") is not None:
        parts.append("arrived %.1fs" % meta["t_arrive"])
    parts.append("dur %.1fs" % meta.get("duration_s", 0))
    if "n_switches" in meta:
        parts.append("switches %d" % meta["n_switches"])
    cloud = meta.get("cloud") or {}
    if cloud.get("arrived"):
        parts.append("cloud rx %d / drop %d" % (cloud["arrived"], cloud.get("dropped", 0)))
    res = meta.get("resources") or {}
    if "cpu_pct" in res:
        parts.append("cpu %.0f%%" % res["cpu_pct"])
    if "tx_mbit_s" in res:
        parts.append("tx %.2f Mbit/s" % res["tx_mbit_s"])
    print("    " + " | ".join(parts))


def record(rid, opts, start_side=""):
    row = {"run_id": rid, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "start_side": start_side, "manual_success": "", "power_w": "", "note": ""}
    if opts.get("ask_power"):
        row["power_w"] = input("    USB meter average power in W (ENTER = none) > ").strip()
    if opts.get("score", True):
        answer = input("    Score: p = pass, f = fail, ENTER = no score (text after it = note) > ").strip()
        if answer[:1].lower() in ("p", "f"):
            row["manual_success"] = "1" if answer[0].lower() == "p" else "0"
            answer = answer[1:].strip()
        row["note"] = answer
    else:
        row["note"] = input("    Note (ENTER = none) > ").strip()
    new_file = not os.path.exists(NOTES)
    with open(NOTES, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new_file:
            w.writeheader()
        w.writerow(row)


def netem(on):
    cmd = ["sudo", "tc", "qdisc", "replace" if on else "del", "dev", NETEM_IFACE, "root"]
    if on:
        cmd += ["netem"] + NETEM.split()
    print("    $ " + " ".join(cmd))
    subprocess.run(cmd, check=on)


def run_block(name, runs, opts, dry=False):
    for run in runs:
        run["id"] = run_id(run)
    ids = [run["id"] for run in runs]
    assert len(ids) == len(set(ids)), "duplicate run ids in block %s" % name
    random.Random("%d-%s" % (SEED, name)).shuffle(runs)
    todo = [run for run in runs if not is_done(run["id"])]
    print("=== %s: %d runs, %d already done, %d to go (~%.0f min) ===" % (
        name, len(runs), len(runs) - len(todo), len(todo), len(todo) * SECONDS_PER_RUN / 60.0))
    if opts.get("intro"):
        print(opts["intro"])
    if dry:
        for run in todo:
            print("   " + run["id"])
        return
    os.makedirs(LOG_DIR, exist_ok=True)
    # battery state changes how the car drives (and drifts during a block), so record it
    battery = input("Battery voltage now (ENTER = skip) > ").strip()
    with open(os.path.join(LOG_DIR, "plan_%s.json" % name), "w") as f:
        json.dump({"seed": SEED, "battery_v_start": battery, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "order": [run["id"] for run in runs]}, f, indent=1)
    if opts.get("netem"):
        if input("Apply netem now? [y/N] > ").strip().lower() != "y":
            return
        netem(True)
    try:
        for i, run in enumerate(todo, 1):
            while True:
                side = "LEFT " if run["rep"] % 2 else "RIGHT"   # by rep, so each cell is balanced
                run["start_side"] = side.strip()
                answer = input("\n[%d/%d] %s\n    Place robot on the %s 20 cm mark."
                               "  ENTER = launch, s = skip, q = quit > "
                               % (i, len(todo), run["id"], side)).strip().lower()
                if answer == "q":
                    return
                if answer == "s":
                    break
                code = launch(run)
                meta = read_meta(run["id"])
                if meta is None or meta.get("aborted"):
                    reason = (meta or {}).get("aborted") or "manager crashed (exit code %s), see above" % code
                    print("    !!! run failed: %s" % reason)
                    print("    -> fix it (e.g. restart pc_edge_brain.py), then ENTER to retry")
                    continue
                print_summary(meta)
                record(run["id"], opts, run.get("start_side", ""))
                break
    finally:
        if opts.get("netem"):
            netem(False)


def single(args):
    env = dict(COMMON)
    for arg in args:
        key, _, value = arg.partition("=")
        env[key.upper()] = value
    run = {"exp": "single", "rep": 1, "env": {k: str(v) for k, v in env.items()},
           "id": "single_" + time.strftime("%Y%m%d_%H%M%S")}
    os.makedirs(LOG_DIR, exist_ok=True)
    launch(run)
    print_summary(read_meta(run["id"]))


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in list(BLOCKS) + ["single"]:
        print(__doc__)
        sys.exit(1)
    if sys.argv[1] == "single":
        single(sys.argv[2:])
    else:
        block_runs, block_opts = BLOCKS[sys.argv[1]]()
        run_block(sys.argv[1], block_runs, block_opts, dry="--dry" in sys.argv)
    print("\nDone. Logs in %s" % os.path.abspath(LOG_DIR))
