# Results recreated from the run logs

Runs per block: E1 27, E2 95, E2 pilot 75, E3 135, E3' 70, E4 48, E5 20

## Table 6: average detection time per image (mean of the per-run averages)

| Detection method | Raspberry Pi (ms) | PC (ms) | runs Pi / PC |
| --- | --- | --- | --- |
| aruco:id=0 | 11.1 | 1.9 | 53 / 66 |
| yolo:160 | 251.5 | – | 3 / 0 |
| yolo:320 | 955.8 | 28.2 | 185 / 277 |

## Table 10: E2, time to target and success with YOLO

| Condition | n | Arrived % [95 % CI] | Time to target s (mean ± CI) |
| --- | --- | --- | --- |
| CLOUD +0 ms | 10 | 100 [72–100] | 4.06 ± 0.21 |
| CLOUD +200 ms | 10 | 100 [72–100] | 4.14 ± 0.18 |
| CLOUD +400 ms | 10 | 100 [72–100] | 4.82 ± 0.25 |
| CLOUD +800 ms | 10 | 100 [72–100] | 5.62 ± 0.29 |
| **LOCAL** | 15 | 100 [80–100] | 6.17 ± 0.26 |
| CLOUD +1200 ms | 10 | 100 [72–100] | 6.35 ± 0.37 |
| CLOUD +1600 ms | 10 | 100 [72–100] | 7.50 ± 0.39 |
| CLOUD +2000 ms | 10 | 90 [60–98] | 7.78 ± 0.37 |
| CLOUD +2400 ms | 10 | 60 [31–83] | 8.71 ± 0.79 |

- Crossover latency L* = **1098 ms**, 95 % bootstrap CI 914–1273 ms
- Linear fit: +0.20 s time to target per 100 ms added latency
- Estimate from detection times (Table 6): 956 − 28 = **928 ms** (measured L* is 18 % higher)
- Estimate from logged answer ages in E2: onboard 1004 ms − server at +0 ms 70 ms = 934 ms
- E2 pilot (centreline start, three-step steering, stopped by hand): 75 runs, 85.3 % successful by manual score, turn commands in 5.8 % of all control cycles

## Table 11: E1, Raspberry Pi power, CPU load and uplink per configuration

| Configuration | Power (W) | CPU load (%) | Uplink (Mbit/s) |
| --- | --- | --- | --- |
| Idle | 1.8 | 3.2 | – |
| LOCAL ArUco | 3.0 | 13.8 | – |
| CLOUD ArUco | 3.0 | 13.1 | 0.38 |
| CLOUD ArUco, uncompressed frames | 3.2 | 11.0 | 9.15 |
| CLOUD YOLO-320 | 3.2 | 11.8 | 0.38 |
| HYBRID YOLO-320, cold standby | 3.0 | 12.1 | 0.38 |
| HYBRID YOLO-320, hot standby | 5.0 | 77.9 | 0.37 |
| LOCAL YOLO-160 | 4.8 | 81.0 | – |
| LOCAL YOLO-320 | 4.8 | 77.8 | – |

## Table 12: E3', ArUco under delay spike and outage

| Configuration | Disturbance | n | Arrived % [CI] | Time to target s | Switches per run |
| --- | --- | --- | --- | --- | --- |
| LOCAL | none | 10 | 100.0 [72–100] | 2.88 ± 0.12 | 0.0 |
| CLOUD | delay spike | 10 | 60.0 [31–83] | 4.72 ± 0.24 | 0.0 |
| P0 | delay spike | 10 | 100.0 [72–100] | 3.30 ± 0.20 | 3.0 |
| P2 | delay spike | 10 | 100.0 [72–100] | 2.89 ± 0.09 | 0.0 |
| CLOUD | outage | 10 | 70.0 [40–89] | 5.22 ± 0.18 | 0.0 |
| P0 | outage | 10 | 100.0 [72–100] | 3.30 ± 0.12 | 3.0 |
| P2 | outage | 10 | 100.0 [72–100] | 2.91 ± 0.11 | 0.0 |

## Table 13: E3, YOLO under delay spike and outage

| Configuration | Disturbance | n | Arrived % [CI] | Time to target s | Switches per run |
| --- | --- | --- | --- | --- | --- |
| LOCAL | none | 15 | 100.0 [80–100] | 6.26 ± 0.12 | 0.0 |
| CLOUD | delay spike | 15 | 60.0 [36–80] | 10.07 ± 0.41 | 0.0 |
| P0 | delay spike | 15 | 53.3 [30–75] | 10.91 ± 0.38 | 4.5 |
| P1 | delay spike | 15 | 93.3 [70–99] | 7.14 ± 0.14 | 4.5 |
| P2 | delay spike | 15 | 93.3 [70–99] | 6.70 ± 0.17 | 4.2 |
| CLOUD | outage | 15 | 60.0 [36–80] | 8.52 ± 1.58 | 0.0 |
| P0 | outage | 15 | 80.0 [55–93] | 7.27 ± 0.23 | 4.2 |
| P1 | outage | 15 | 93.3 [70–99] | 6.97 ± 0.11 | 4.1 |
| P2 | outage | 15 | 100.0 [80–100] | 6.89 ± 0.10 | 4.1 |

## Fisher's exact tests (success)

- E3 spike: P1 14/15 vs. P0 8/15, p = 0.035
- E3 spike: P2 14/15 vs. P0 8/15, p = 0.035
- E3 spike: P1 14/15 vs. CLOUD 9/15, p = 0.080
- E3 spike: P2 14/15 vs. CLOUD 9/15, p = 0.080
- E3 spike: P1 14/15 vs. P2 14/15, p = 1.000
- E3 outage: P2 15/15 vs. CLOUD 9/15, p = 0.017
- E3 outage: P1 14/15 vs. CLOUD 9/15, p = 0.080
- E3 outage: P0 12/15 vs. CLOUD 9/15, p = 0.427

## Switch reasons in E3 and E3' (switches after the run start)

- e3_final cost outage → CLOUD: cloud fresher × 38
- e3_final cost outage → LOCAL: silence × 23
- e3_final cost spike → CLOUD: cloud fresher × 39
- e3_final cost spike → LOCAL: local fresher × 1
- e3_final cost spike → LOCAL: silence × 23
- e3_final ewma outage → CLOUD: recovered × 38
- e3_final ewma outage → LOCAL: silence × 23
- e3_final ewma spike → CLOUD: recovered × 41
- e3_final ewma spike → LOCAL: silence × 26
- e3_final timeout outage → CLOUD: cloud alive × 24
- e3_final timeout outage → LOCAL: silence × 24
- e3_final timeout spike → CLOUD: cloud alive × 26
- e3_final timeout spike → LOCAL: silence × 26
- e3aruco_final timeout outage → CLOUD: cloud alive × 10
- e3aruco_final timeout outage → LOCAL: silence × 10
- e3aruco_final timeout spike → CLOUD: cloud alive × 10
- e3aruco_final timeout spike → LOCAL: silence × 10
- E3' P2: 1179 of 1179 control cycles steered by the Pi

## Table 14: E4, fixed timeout with different τ

| τ (s) | Disturbance | Arrived % [CI] | Switches mean / median / max | Outage detection delay s |
| --- | --- | --- | --- | --- |
| 0.2 | jitter | 62.5 [31–86] | 42.1 / 14 / 97 | – |
| 0.2 | outage | 62.5 [31–86] | 20.2 / 6 / 49 | -0.61 ± 0.16 |
| 0.35 | jitter | 87.5 [53–98] | 6.5 / 5 / 21 | – |
| 0.35 | outage | 75.0 [41–93] | 3.0 / 3 / 3 | +0.28 ± 0.04 |
| 0.5 | jitter | 100.0 [68–100] | 1.0 / 1 / 1 | – |
| 0.5 | outage | 75.0 [41–93] | 3.0 / 3 / 3 | +0.41 ± 0.02 |

- Fisher, jitter τ = 0.2 vs. 0.5 s: p = 0.20
- Gap between server answers (E2, +0 ms): median 0.30 s, mean 0.27 s

## E5: real network impairment (tc/netem)

- CLOUD: 10 | 100.0 [72–100] | 3.44 ± 0.11 | 0.0, server answers dropped: 0
- P2: 10 | 100.0 [72–100] | 3.43 ± 0.13 | 3.2, server answers dropped: 0

## Success by start side

- e2_final: LEFT 96 %, RIGHT 94 %
- e3_final: LEFT 86 %, RIGHT 76 %
- e3aruco_final: LEFT 91 %, RIGHT 89 %
- e4_final: LEFT 79 %, RIGHT 75 %
- e5_final: LEFT 100 %, RIGHT 100 %
