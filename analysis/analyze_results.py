#!/usr/bin/env python3
"""
analyze_results.py  -- recreates every table and result figure of Chapter 6 from the run logs.

    python analysis/analyze_results.py

Reads   data/<block>/meta_*.json, run_*.csv, switch_*.csv, runs_notes.csv
Writes  analysis/output/results.md            all tables and statistics (Tables 6, 10-15)
        analysis/output/fig11_crossover.png    E2: time to target vs. added latency
        analysis/output/fig12_resources.png    E1: Pi power and CPU load per configuration
        analysis/output/fig13_yolo_vs_aruco.png   E3/E3': relative time to target, switches
        analysis/output/fig14_policies.png     E3: success and time to target per policy
        analysis/output/fig15_example_runs.png E3: answer ages in one P0 and one P2 run
        analysis/output/fig16_timeout.png      E4: switches and outage detection delay vs. tau

Conventions (as in the thesis):
  success            stop_reason == "arrived" (arrival criterion fired before the 25 s limit)
  mean +- CI         mean +- 1.96 standard errors, over successful runs
  success CI         95 % Wilson score interval
  tests              two-sided Fisher's exact test
  crossover L*       linear interpolation between the two CLOUD latency levels whose mean
                     arrival times lie on either side of the LOCAL mean; 95 % CI from a
                     bootstrap with 20 000 resamples of the runs within each condition
  power (E1)         the meter shows Wh after 60 s, so W = Wh * 60; one reading of 0.6 Wh
                     (ten times the others, a transcription error) is excluded
"""
import glob
import json
import math
import os
import re
from math import comb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "analysis", "output")
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, YELLOW = "#2a78d6", "#eb6834", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": ["Calibri", "DejaVu Sans"], "font.size": 11, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "savefig.dpi": 200, "legend.frameon": False})


# ---------------------------------------------------------------- helpers
def load(block):
    runs = []
    for f in sorted(glob.glob(os.path.join(DATA, block, "meta_*.json"))):
        with open(f) as fh:
            m = json.load(fh)
        m["ok"] = m["stop_reason"] == "arrived"
        m["kind"] = kind(m)
        m["dist"] = (m.get("net_profile") or "stable").split(":")[0]
        runs.append(m)
    return runs


def kind(m):
    if m["strategy"] != "HYBRID":
        return m["strategy"]
    return {"timeout": "P0", "ewma": "P1", "cost": "P2"}[m["policy"].split(":")[0]]


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def ci(values):
    x = np.asarray(values, float)
    if len(x) < 2:
        return (x.mean() if len(x) else float("nan")), 0.0
    return x.mean(), 1.96 * x.std(ddof=1) / math.sqrt(len(x))


def fisher(k1, n1, k2, n2):
    """Two-sided Fisher's exact test for k1/n1 vs. k2/n2 successes."""
    K, N = k1 + k2, n1 + n2
    p = lambda x: comb(n1, x) * comb(n2, K - x) / comb(N, K)
    p0 = p(k1)
    return sum(p(x) for x in range(max(0, K - n2), min(n1, K) + 1) if p(x) <= p0 * (1 + 1e-9))


def cell(runs):
    k, n = sum(r["ok"] for r in runs), len(runs)
    lo, hi = wilson(k, n)
    t, h = ci([r["t_arrive"] for r in runs if r["ok"]])
    return dict(n=n, k=k, pct=100 * k / n, lo=lo, hi=hi, t=t, th=h,
                sw=np.mean([r["n_switches"] for r in runs]))


def fmt_cell(c):
    return "%d | %.1f [%.0f–%.0f] | %.2f ± %.2f | %.1f" % (c["n"], c["pct"], c["lo"], c["hi"], c["t"], c["th"], c["sw"])


md = []


def h(title):
    md.append("\n## " + title + "\n")


def row(*cols):
    md.append("| " + " | ".join(str(c) for c in cols) + " |")


# ---------------------------------------------------------------- load
E1, E2, E2C, E3, E3A, E4, E5 = (load(b) for b in
                                ("e1_final", "e2_final", "e2_centre_control", "e3_final", "e3aruco_final", "e4_final", "e5_final"))
ALL = E1 + E2 + E3 + E3A + E4 + E5

md.append("# Results recreated from the run logs\n")
md.append("Runs per block: " + ", ".join("%s %d" % (n, len(b)) for n, b in
                                          [("E1", E1), ("E2", E2), ("E2 pilot", E2C), ("E3", E3), ("E3'", E3A), ("E4", E4), ("E5", E5)]))

# ---------------------------------------------------------------- Table 6: detection times
h("Table 6: average detection time per image (mean of the per-run averages)")
row("Detection method", "Raspberry Pi (ms)", "PC (ms)", "runs Pi / PC")
row("---", "---", "---", "---")
det = {}
for m in ALL:
    spec = re.sub(r",.*", "", m.get("perception") or "")
    for side in ("local", "cloud"):
        v = (m.get(side) or {}).get("mean_infer_ms") if isinstance(m.get(side), dict) else None
        if v:
            det.setdefault((spec, side), []).append(v)
for spec in ("aruco:id=0", "yolo:160", "yolo:320"):
    pi, pc = det.get((spec, "local"), []), det.get((spec, "cloud"), [])
    row(spec, "%.1f" % np.mean(pi) if pi else "–", "%.1f" % np.mean(pc) if pc else "–", "%d / %d" % (len(pi), len(pc)))
T_LOCAL, T_REMOTE = np.mean(det[("yolo:320", "local")]), np.mean(det[("yolo:320", "cloud")])

# ---------------------------------------------------------------- Table 10 + L*: E2
h("Table 10: E2, time to target and success with YOLO")
row("Condition", "n", "Arrived % [95 % CI]", "Time to target s (mean ± CI)")
row("---", "---", "---", "---")
levels = sorted({r["inject_ms"] for r in E2 if r["strategy"] == "CLOUD"})
cloud = {L: [r for r in E2 if r["strategy"] == "CLOUD" and r["inject_ms"] == L] for L in levels}
local = [r for r in E2 if r["strategy"] == "LOCAL"]
lc = cell(local)
for L in levels:
    c = cell(cloud[L])
    if L == 1200:
        row("**LOCAL**", lc["n"], "%.0f [%.0f–%.0f]" % (lc["pct"], lc["lo"], lc["hi"]), "%.2f ± %.2f" % (lc["t"], lc["th"]))
    row("CLOUD +%d ms" % L, c["n"], "%.0f [%.0f–%.0f]" % (c["pct"], c["lo"], c["hi"]), "%.2f ± %.2f" % (c["t"], c["th"]))


def crossover(means, lmean):
    for a, b in zip(levels, levels[1:]):
        if means[a] <= lmean < means[b]:
            return a + (lmean - means[a]) / (means[b] - means[a]) * (b - a)
    return float("nan")


arr = {L: np.array([r["t_arrive"] for r in cloud[L] if r["ok"]]) for L in levels}
loc_t = np.array([r["t_arrive"] for r in local if r["ok"]])
Lstar = crossover({L: v.mean() for L, v in arr.items()}, loc_t.mean())
rng = np.random.default_rng(1)
boot = [crossover({L: rng.choice(v, len(v)).mean() for L, v in arr.items()}, rng.choice(loc_t, len(loc_t)).mean())
        for _ in range(20000)]
lo_b, hi_b = np.nanpercentile(boot, [2.5, 97.5])
x = np.concatenate([[L] * len(v) for L, v in arr.items()])
y = np.concatenate(list(arr.values()))
slope = np.polyfit(x, y, 1)[0] * 100
age_local = pd.concat(pd.read_csv(f) for f in glob.glob(os.path.join(DATA, "e2_final", "run_e2_LOCAL*.csv"))).local_stale_ms.mean()
age_cloud = pd.concat(pd.read_csv(f) for f in glob.glob(os.path.join(DATA, "e2_final", "run_e2_CLOUD_yolo320_stable_r*.csv"))).cloud_stale_ms.mean()
md.append("")
md.append("- Crossover latency L* = **%.0f ms**, 95 %% bootstrap CI %.0f–%.0f ms" % (Lstar, lo_b, hi_b))
md.append("- Linear fit: +%.2f s time to target per 100 ms added latency" % slope)
md.append("- Estimate from detection times (Table 6): %.0f − %.0f = **%.0f ms** (measured L* is %.0f %% higher)"
          % (T_LOCAL, T_REMOTE, T_LOCAL - T_REMOTE, 100 * (Lstar / (T_LOCAL - T_REMOTE) - 1)))
md.append("- Estimate from logged answer ages in E2: onboard %.0f ms − server at +0 ms %.0f ms = %.0f ms"
          % (age_local, age_cloud, age_local - age_cloud))
# The E2 pilot still used three-step steering (forward / left / right) and was stopped by hand,
# so success is the manual pass/fail score and turns are counted over all control cycles.
pilot_notes = pd.read_csv(os.path.join(DATA, "e2_centre_control", "runs_notes.csv"))
n_turn = n_cycles = 0
for f in glob.glob(os.path.join(DATA, "e2_centre_control", "run_*.csv")):
    cmd = pd.read_csv(f, usecols=["command"]).command.astype(str)
    n_turn += cmd.isin(["left", "right"]).sum()
    n_cycles += len(cmd)
md.append("- E2 pilot (centreline start, three-step steering, stopped by hand): %d runs, %.1f %% successful by manual score, "
          "turn commands in %.1f %% of all control cycles"
          % (len(pilot_notes), 100 * pilot_notes.manual_success.mean(), 100 * n_turn / n_cycles))

# ---------------------------------------------------------------- Table 11: E1
h("Table 11: E1, Raspberry Pi power, CPU load and uplink per configuration")
row("Configuration", "Power (W)", "CPU load (%)", "Uplink (Mbit/s)")
row("---", "---", "---", "---")
notes = pd.read_csv(os.path.join(DATA, "e1_final", "runs_notes.csv"))
watt = {r.run_id: (r.power_w * 60 if pd.notna(r.power_w) and r.power_w * 60 < 20 else np.nan) for r in notes.itertuples()}
E1_ORDER = [("e1_IDLE", "Idle"), ("e1_LOCAL_aruco", "LOCAL ArUco"), ("e1_CLOUD_aruco_stable", "CLOUD ArUco"),
            ("e1_CLOUD_aruco_stable_raw", "CLOUD ArUco, uncompressed frames"), ("e1_CLOUD_yolo320_stable", "CLOUD YOLO-320"),
            ("e1_HYBRID_timeout-tau0.5_cold_yolo320_stable", "HYBRID YOLO-320, cold standby"),
            ("e1_HYBRID_timeout-tau0.5_yolo320_stable", "HYBRID YOLO-320, hot standby"),
            ("e1_LOCAL_yolo160", "LOCAL YOLO-160"), ("e1_LOCAL_yolo320", "LOCAL YOLO-320")]
e1rows = []
for cid, label in E1_ORDER:
    g = [m for m in E1 if re.sub(r"_r\d+$", "", m["run_id"]) == cid]
    w = np.nanmean([watt.get(m["run_id"], np.nan) for m in g])
    cpu = np.mean([m["resources"]["cpu_pct"] for m in g])
    tx = np.mean([m["resources"]["tx_mbit_s"] for m in g])
    e1rows.append((label, w, cpu))
    row(label, "%.1f" % w, "%.1f" % cpu, "%.2f" % tx if tx > 0.01 else "–")

# ---------------------------------------------------------------- Tables 12, 13: E3', E3
def policy_table(runs, kinds, title):
    h(title)
    row("Configuration", "Disturbance", "n", "Arrived % [CI]", "Time to target s", "Switches per run")
    row("---", "---", "---", "---", "---", "---")
    out = {}
    out["LOCAL"] = cell([r for r in runs if r["kind"] == "LOCAL"])
    row("LOCAL", "none", fmt_cell(out["LOCAL"]))
    for dist in ("spike", "outage"):
        for k in kinds:
            c = cell([r for r in runs if r["kind"] == k and r["dist"] == dist])
            out[(k, dist)] = c
            row(k, "delay spike" if dist == "spike" else dist, fmt_cell(c))
    return out


C3A = policy_table(E3A, ["CLOUD", "P0", "P2"], "Table 12: E3', ArUco under delay spike and outage")
C3 = policy_table(E3, ["CLOUD", "P0", "P1", "P2"], "Table 13: E3, YOLO under delay spike and outage")

h("Fisher's exact tests (success)")
for a, b, dist in [("P1", "P0", "spike"), ("P2", "P0", "spike"), ("P1", "CLOUD", "spike"), ("P2", "CLOUD", "spike"),
                   ("P1", "P2", "spike"), ("P2", "CLOUD", "outage"), ("P1", "CLOUD", "outage"), ("P0", "CLOUD", "outage")]:
    x1, x2 = C3[(a, dist)], C3[(b, dist)]
    md.append("- E3 %s: %s %d/%d vs. %s %d/%d, p = %.3f" % (dist, a, x1["k"], x1["n"], b, x2["k"], x2["n"],
                                                            fisher(x1["k"], x1["n"], x2["k"], x2["n"])))

h("Switch reasons in E3 and E3' (switches after the run start)")
sw = []
for blk in ("e3_final", "e3aruco_final"):
    for f in glob.glob(os.path.join(DATA, blk, "switch_*.csv")):
        d = pd.read_csv(f)
        if len(d):
            d["block"] = blk
            sw.append(d)
sw = pd.concat(sw)
sw["policy"] = sw.run_id.str.extract(r"HYBRID_([a-z]+)-")[0]
sw["dist"] = sw.run_id.str.extract(r"_(spike|outage)")[0]
counts = sw[sw.t_rel > 0].groupby(["block", "policy", "dist", "to_brain", "reason"]).size()
for idx, v in counts.items():
    md.append("- %s %s %s → %s: %s × %d" % (idx[0], idx[1], idx[2], idx[3], idx[4], v))
aruco_p2 = pd.concat(pd.read_csv(f) for f in glob.glob(os.path.join(DATA, "e3aruco_final", "run_e3a_HYBRID_cost*.csv")))
md.append("- E3' P2: %d of %d control cycles steered by the Pi" % ((aruco_p2.brain == "LOCAL").sum(), len(aruco_p2)))

# ---------------------------------------------------------------- Table 14: E4
h("Table 14: E4, fixed timeout with different τ")
row("τ (s)", "Disturbance", "Arrived % [CI]", "Switches mean / median / max", "Outage detection delay s")
row("---", "---", "---", "---", "---")
for m in E4:
    m["tau"] = float(re.search(r"tau=([\d.]+)", m["policy"]).group(1))
delay = {}
for f in glob.glob(os.path.join(DATA, "e4_final", "switch_*outage*.csv")):
    d = pd.read_csv(f)
    tau = float(re.search(r"tau([\d.]+)_", f).group(1))
    first = d[(d.from_brain == "CLOUD") & (d.to_brain == "LOCAL")]
    if len(first):
        delay.setdefault(tau, []).append(first.t_rel.iloc[0] - 1.0)     # outage starts at t = 1.0 s
TAUS = [0.2, 0.35, 0.5]
e4sw = {}
for tau in TAUS:
    for dist in ("jitter", "outage"):
        g = [m for m in E4 if m["tau"] == tau and m["dist"] == dist]
        c = cell(g)
        s = [m["n_switches"] for m in g]
        e4sw[(tau, dist)] = s
        dd = "%+.2f ± %.2f" % ci(delay[tau]) if dist == "outage" else "–"
        row(tau, dist, "%.1f [%.0f–%.0f]" % (c["pct"], c["lo"], c["hi"]),
            "%.1f / %.0f / %d" % (np.mean(s), np.median(s), max(s)), dd)
j02 = [m["ok"] for m in E4 if m["tau"] == 0.2 and m["dist"] == "jitter"]
j05 = [m["ok"] for m in E4 if m["tau"] == 0.5 and m["dist"] == "jitter"]
md.append("")
md.append("- Fisher, jitter τ = 0.2 vs. 0.5 s: p = %.2f" % fisher(sum(j02), len(j02), sum(j05), len(j05)))
gaps = []
for f in glob.glob(os.path.join(DATA, "e2_final", "run_e2_CLOUD_yolo320_stable_r*.csv")):
    d = pd.read_csv(f)
    ch = d[d.cloud_fid.diff() != 0].dropna(subset=["cloud_fid"])
    gaps += list(ch.t.diff().dropna())
md.append("- Gap between server answers (E2, +0 ms): median %.2f s, mean %.2f s" % (np.median(gaps), np.mean(gaps)))

# ---------------------------------------------------------------- E5, start side
h("E5: real network impairment (tc/netem)")
for k in ("CLOUD", "P2"):
    c = cell([m for m in E5 if m["kind"] == k])
    dropped = sum((m.get("cloud") or {}).get("dropped", 0) for m in E5 if m["kind"] == k)
    md.append("- %s: %s, server answers dropped: %d" % (k, fmt_cell(c), dropped))

h("Success by start side")
for name, blk in [("e2_final", E2), ("e3_final", E3), ("e3aruco_final", E3A), ("e4_final", E4), ("e5_final", E5)]:
    n = pd.read_csv(os.path.join(DATA, name, "runs_notes.csv"))
    side = dict(zip(n.run_id, n.start_side))
    s = {}
    for m in blk:
        s.setdefault(side.get(m["run_id"]), []).append(m["ok"])
    md.append("- %s: " % name + ", ".join("%s %.0f %%" % (k, 100 * np.mean(v)) for k, v in sorted(s.items(), key=str)))

with open(os.path.join(OUT, "results.md"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(md) + "\n")

# ================================================================ figures
NAMES = {"CLOUD": "CLOUD", "P0": "P0 fixed\ntimeout", "P1": "P1 latency-\naware", "P2": "P2 cost\nmodel"}

# --- Figure 11: crossover
fig, ax = plt.subplots(figsize=(6.6, 3.8))
lm, lh = lc["t"], lc["th"]
ax.axhspan(lm - lh, lm + lh, color=ORANGE, alpha=.15, lw=0)
ax.axhline(lm, color=ORANGE, lw=2)
ax.text(60, lm + 0.12, "onboard YOLO (LOCAL): %.2f s" % lm, color=INK, fontsize=10)
cc = [cell(cloud[L]) for L in levels]
ax.errorbar(levels, [c["t"] for c in cc], yerr=[c["th"] for c in cc], color=BLUE, lw=2, marker="o", ms=6, capsize=3,
            mec="white", mew=1, label="offloaded YOLO (CLOUD)")
ax.axvline(Lstar, color=INK2, ls="--", lw=1)
ax.text(Lstar + 30, 4.2, "measured L* ≈ %d ms" % (round(Lstar / 100) * 100), fontsize=10, color=INK)
est = T_LOCAL - T_REMOTE
ax.axvline(est, color=INK2, ls=":", lw=1)
ax.text(est - 30, 8.9, "estimated ≈ %d ms" % (round(est / 10) * 10), fontsize=10, color=INK2, ha="right")
for L, c in zip(levels, cc):
    if c["pct"] < 100:
        ax.text(L, c["t"] + c["th"] + 0.15, "%.0f %% arrived" % c["pct"], ha="center", fontsize=9.5, color=INK2)
ax.set_xlabel("Added latency on the server's answers (ms)")
ax.set_ylabel("Time to target (s), mean ± 95 % CI")
ax.set_xticks(levels); ax.set_ylim(3.5, 9.9); ax.legend(loc="upper left", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig11_crossover.png")); plt.close(fig)

# --- Figure 12: resources
FIG12 = [r for r in e1rows if r[0] != "CLOUD ArUco, uncompressed frames"]
short = {"HYBRID YOLO-320, cold standby": "HYBRID cold YOLO-320", "HYBRID YOLO-320, hot standby": "HYBRID hot YOLO-320"}
lab = [short.get(r[0], r[0]) for r in FIG12]
fig, axs = plt.subplots(1, 2, figsize=(6.5, 3.4), sharey=True)
yy = np.arange(len(lab))[::-1]
for ax, vals, t, f in [(axs[0], [r[1] for r in FIG12], "Raspberry Pi power (W)", "%.1f"),
                       (axs[1], [r[2] for r in FIG12], "Raspberry Pi CPU load (%)", "%.0f")]:
    cols = [INK2 if l == "Idle" else (ORANGE if ("LOCAL YOLO" in l or "hot" in l) else BLUE) for l in lab]
    ax.barh(yy, vals, color=cols, height=.62)
    for a, v in zip(yy, vals):
        ax.text(v + max(vals) * .02, a, f % v, va="center", fontsize=8.5, color=INK)
    ax.set_xlabel(t); ax.grid(axis="y", visible=False); ax.set_xlim(0, max(vals) * 1.2)
axs[0].set_yticks(yy); axs[0].set_yticklabels(lab)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig12_resources.png")); plt.close(fig)

# --- Figure 13: YOLO vs. ArUco
w = .36
fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.6))
for j, (C, col) in enumerate([(C3, BLUE), (C3A, ORANGE)]):
    xx = np.arange(3) + (j - .5) * w
    rel = [100 * C[(k, "spike")]["t"] / C["LOCAL"]["t"] for k in ("CLOUD", "P0", "P2")]
    nsw = [C[(k, "spike")]["sw"] for k in ("CLOUD", "P0", "P2")]
    axs[0].bar(xx, rel, w * .92, color=col)
    axs[1].bar(xx, nsw, w * .92, color=col)
    for a, v in zip(xx, nsw):
        axs[1].text(a, v + .1, "%.1f" % v, ha="center", fontsize=9.5)
axs[0].axhline(100, color=INK2, ls="--", lw=1)
axs[0].set_ylabel("Time to target, % of LOCAL"); axs[1].set_ylabel("Switches per run")
for ax in axs:
    ax.set_xticks(range(3)); ax.set_xticklabels([NAMES[k] for k in ("CLOUD", "P0", "P2")]); ax.grid(axis="x", visible=False)
axs[0].set_ylim(0, 200); axs[1].set_ylim(0, 5.5)
fig.legend(handles=[Patch(color=BLUE, label="YOLO (E3)"), Patch(color=ORANGE, label="ArUco (E3′)"),
                    Line2D([], [], color=INK2, ls="--", lw=1, label="onboard only (LOCAL) = 100 %")],
           loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(os.path.join(OUT, "fig13_yolo_vs_aruco.png")); plt.close(fig)

# --- Figure 14: policies
K = ["CLOUD", "P0", "P1", "P2"]
x4 = np.arange(4)
fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.7))
for j, (dist, col) in enumerate([("spike", BLUE), ("outage", ORANGE)]):
    cs = [C3[(k, dist)] for k in K]
    xx = x4 + (j - .5) * w
    axs[0].bar(xx, [c["pct"] for c in cs], w * .92, color=col)
    axs[0].errorbar(xx, [c["pct"] for c in cs], yerr=[[c["pct"] - c["lo"] for c in cs], [c["hi"] - c["pct"] for c in cs]],
                    fmt="none", ecolor=INK, capsize=2, lw=.8)
    axs[1].bar(xx, [c["t"] for c in cs], w * .92, color=col)
    axs[1].errorbar(xx, [c["t"] for c in cs], yerr=[c["th"] for c in cs], fmt="none", ecolor=INK, capsize=2, lw=.8)
axs[0].axhline(100, color=INK2, ls="--", lw=1); axs[1].axhline(C3["LOCAL"]["t"], color=INK2, ls="--", lw=1)
axs[0].set_ylabel("Runs that arrived (%), 95 % Wilson CI"); axs[1].set_ylabel("Time to target (s), mean ± 95 % CI")
for ax in axs:
    ax.set_xticks(x4); ax.set_xticklabels([NAMES[k] for k in K], fontsize=10); ax.grid(axis="x", visible=False)
axs[0].set_ylim(0, 110)
fig.legend(handles=[Patch(color=BLUE, label="delay spike"), Patch(color=ORANGE, label="outage"),
                    Line2D([], [], color=INK2, ls="--", lw=1,
                           label="onboard only (LOCAL): 100 %% arrived, %.2f s" % C3["LOCAL"]["t"])],
           loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(os.path.join(OUT, "fig14_policies.png")); plt.close(fig)

# --- Figure 15: example runs
fig, axs = plt.subplots(2, 1, figsize=(6.6, 4.8), sharex=True)
for ax, pat, name in [(axs[0], "timeout-tau0.5", "P0 fixed timeout"), (axs[1], "cost-margin30", "P2 cost model")]:
    f = sorted(glob.glob(os.path.join(DATA, "e3_final", "run_e3_HYBRID_%s_yolo320_spike*_r1.csv" % pat)))[0]
    with open(f.replace("run_", "meta_").replace(".csv", ".json")) as fh:
        m = json.load(fh)
    d = pd.read_csv(f)
    d = d[d.t_rel <= m["t_arrive"] + 0.05]
    ax.axvspan(1, 3.5, color=YELLOW, alpha=.22, lw=0)
    ax.plot(d.t_rel, d.cloud_stale_ms, color=BLUE, lw=1.6)
    ax.plot(d.t_rel, d.local_stale_ms, color=ORANGE, lw=1.6)
    ax.fill_between(d.t_rel.values, -80, -330, where=(d.brain == "LOCAL").values, color=INK2, step="post", lw=0)
    ax.axvline(m["t_arrive"], color=INK, ls="--", lw=1)
    ax.text(m["t_arrive"] - 0.12, 2950, "arrived %.1f s" % m["t_arrive"], ha="right", va="top", fontsize=10, color=INK)
    ax.set_ylim(-380, 3300); ax.set_ylabel("answer age (ms)"); ax.set_title(name, fontsize=11, loc="left", color=INK)
axs[1].set_xlim(-0.3, 12); axs[1].set_xlabel("time since run start (s)")
fig.legend(handles=[Line2D([], [], color=BLUE, lw=1.6, label="age of server answer"), Patch(color=INK2, label="Pi steers"),
                    Line2D([], [], color=ORANGE, lw=1.6, label="age of onboard answer"),
                    Patch(color=YELLOW, alpha=.35, label="delay spike (+1500 ms)")],
           loc="lower center", ncol=2, fontsize=10, bbox_to_anchor=(0.5, 0.0), columnspacing=2.5)
fig.tight_layout(rect=(0, 0.09, 1, 1)); fig.savefig(os.path.join(OUT, "fig15_example_runs.png")); plt.close(fig)

# --- Figure 16: timeout
off = .012
fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.3))
for j, (dist, col) in enumerate([("jitter", BLUE), ("outage", ORANGE)]):
    dx = (j - .5) * 2 * off
    for tau in TAUS:
        v = e4sw[(tau, dist)]
        axs[0].scatter(np.full(len(v), tau) + dx, v, s=14, color=col, alpha=.5, lw=0)
    axs[0].plot([t + dx for t in TAUS], [np.mean(e4sw[(t, dist)]) for t in TAUS], color=col, lw=2, marker="o", ms=6,
                mec="white", label=dist)
axs[0].set_yscale("log"); axs[0].set_yticks([1, 2, 5, 10, 20, 50, 100])
axs[0].set_yticklabels(["1", "2", "5", "10", "20", "50", "100"]); axs[0].minorticks_off()
axs[0].set_xticks(TAUS); axs[0].set_xlabel("timeout τ (s)"); axs[0].set_ylabel("Switches per run (dots = single runs)")
axs[0].legend(fontsize=10)
mm = [ci(delay[t])[0] for t in TAUS]
ss = [ci(delay[t])[1] for t in TAUS]
axs[1].plot([0.15, .55], [.15, .55], color=INK2, ls="--", lw=1)
axs[1].text(.41, .53, "delay = τ", fontsize=10, color=INK2, ha="right")
axs[1].axhline(0, color=INK, lw=.8)
axs[1].errorbar(TAUS, mm, yerr=ss, color=ORANGE, marker="o", lw=2, capsize=3, ms=6, mec="white")
for t, v in zip(TAUS, mm):
    axs[1].text(t + .015, v - .07, "%+.2f s" % v, fontsize=10, va="center")
axs[1].set_xticks(TAUS); axs[1].set_xlabel("timeout τ (s)"); axs[1].set_ylabel("Outage detection delay (s), mean ± 95 % CI")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig16_timeout.png")); plt.close(fig)

print("Wrote", os.path.join(OUT, "results.md"), "and 6 figures.")
