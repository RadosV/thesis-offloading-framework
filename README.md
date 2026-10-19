# Latency-Aware Offloading Framework for a Low-Cost Robot Vehicle

Source code, data and analysis for the Master's thesis

**Design and Empirical Evaluation of a Dynamic Offloading Framework for Resilient Onboard-Cloud Robotic Navigation**

Rados Veselinovic (student number 7213661)
Master Embedded Systems Engineering, Dortmund University of Applied Sciences and Arts (Fachhochschule Dortmund), 2026

---

## What this is

A small RC car with a Raspberry Pi 3B+ drives to a visual target. Target detection runs either on the Pi,
on an edge server (a PC on the same WiFi), or in a hybrid mode in which both run in parallel and a
switching policy decides, 20 times per second, which answer the car steers by.

Two detection methods of very different cost are used: ArUco markers (about 11 ms per image on the Pi)
and a YOLOv5n neural network (about 1 s per image on the Pi, about 28 ms on the PC). Three switching
policies are compared: a fixed timeout (P0), a latency-aware policy (P1) and a cost-model policy (P2).

This repository contains everything needed to repeat the experiments and to recreate every table and
result figure of Chapter 6 of the thesis from the recorded data.

## Repository structure

| Folder | Content |
|---|---|
| `framework/` | Shared library: detection methods, switching policies, network conditions, camera, message format, resource logger |
| `raspberry-pi-scripts/` | Programs on the Pi: hybrid manager, motor controller, self-test, manual driving, ESC and servo test tools |
| `cloud/` | Edge brain, the program on the PC |
| `tests/` | Experiment runner, detection time benchmark, unit tests |
| `tools/` | Export of the YOLOv5n models to ONNX |
| `models/` | YOLOv5n models (ONNX) for 160, 224 and 320 px input |
| `targets/` | Printable targets: stop sign (A4) and ArUco marker, with the scripts that generate them |
| `data/` | All run logs of the experiments reported in the thesis |
| `analysis/` | `analyze_results.py` and its output (`analysis/output/`) |
| `figures/` | draw.io sources of the diagrams in Chapters 4 and 6 |

## Data

| Folder | Runs | Content | Used for |
|---|---|---|---|
| `data/e1_final` | 27 | Pi power, CPU load and uplink per configuration, vehicle stationary | RQ2 |
| `data/e2_final` | 95 | Added latency with YOLO, offset starts | RQ1, RQ2 |
| `data/e2_centre_control` | 75 | Added latency, centreline starts (pilot, motivates the course design) | Section 6.2 |
| `data/e3_final` | 135 | Switching policies with YOLO under delay spike and outage | RQ3, RQ4 |
| `data/e3aruco_final` | 70 | Switching policies with ArUco under delay spike and outage | RQ2, RQ3 |
| `data/e4_final` | 48 | Fixed timeout with τ = 0.2, 0.35 and 0.5 s under outage and jitter | RQ4 |
| `data/e5_final` | 20 | Real network impairment with Linux tc/netem | Validation |

Each run produces four files:

- `run_<id>.csv`: one row per control cycle (which side steers, command, target position, age of the server and onboard answers, detection times, added delay, policy state)
- `switch_<id>.csv`: one row per switch between Pi and server, with the policy's reason
- `res_<id>.csv`: CPU load, memory, network traffic and temperature of the Pi
- `meta_<id>.json`: settings and summary of the run (stop reason, arrival time, number of switches, average detection times)

Each block also has `plan_<block>.json` (seed, run order, battery voltage) and `runs_notes.csv`
(start side, manual pass/fail score, power meter reading, notes).

## Recreating the results

Requires Python 3.9 or newer.

```bash
pip install -r requirements-analysis.txt
python analysis/analyze_results.py
```

This writes `analysis/output/results.md` (Tables 6 and 10–15, the crossover latency with its bootstrap
confidence interval, all Fisher tests, switch reasons, E5 and start-side checks) and the result figures
`fig11`–`fig16` to `analysis/output/`.

## Running the system

### Hardware

Tamiya RC chassis with brushed motor, ESC (PWM) and steering servo (PWM), Raspberry Pi 3B+ on a separate
power bank, USB webcam, HC-SR04 ultrasonic sensor with a 1 kΩ / 2 kΩ voltage divider on the echo line,
and a PC as edge server on the same WiFi. The pin assignment is given in Appendix A.2 of the thesis.

### Software

Raspberry Pi (32-bit Raspberry Pi OS, OpenCV 4.5.5):

```bash
sudo apt install mosquitto pigpio python3-opencv python3-psutil
pip3 install -r requirements-pi.txt
```

PC:

```bash
pip install -r requirements-pc.txt
```

Copy `framework/`, `raspberry-pi-scripts/`, `tests/` and `models/` to the Pi, and `framework/`, `cloud/`
and `models/` to the PC, each with the same folder layout. The PC's IP address is set on the Pi with the
environment variable `PC_IP` (default `192.168.2.101`), and the Pi's address is passed to the edge brain
with `--broker`.

### Order of a session

1. Charge the drive battery and check its voltage.
2. On the Pi, start the PWM service and the MQTT broker: `bash raspberry-pi-scripts/start-scripts.sh`
3. Start the motor controller **before** switching the ESC on: `python3 raspberry-pi-scripts/motor_controller.py`.
   The ESC learns its neutral point from the first signal after power-on.
4. Switch the ESC on.
5. On the PC, start the edge brain: `python cloud/pc_edge_brain.py --broker <Pi IP>`
6. On the Pi, start a block, e.g. `python3 tests/run_experiments.py e3`. The runner shows each run, starts
   the hybrid manager with the right settings and records the outcome. `--dry` only prints the run plan.

Never start `pi_hybrid_manager_and_stream.py` by hand during an experiment. If the car does not move
although all programs look healthy, check the servo/ESC wiring first: `python3 raspberry-pi-scripts/car_selftest.py`
and `manual_drive.py` tell wiring faults from software faults.
