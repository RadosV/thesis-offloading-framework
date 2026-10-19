"""
resources.py  -- CPU / memory / network / temperature sampler
=============================================================
Background thread writing one CSV row every `period` s:
  t, cpu_pct (whole machine), proc_cpu_pct (this process, can exceed 100 on
  multi-core), rss_mb, mem_pct, net_tx_bytes, net_rx_bytes (wlan0 if present),
  temp_c (SoC, Raspberry Pi only)
summary() gives the per-run means and the uplink/downlink rate in Mbit/s.
Needs psutil (Pi: sudo apt install python3-psutil); without it logging is off.
Power is NOT measured here - use the inline USB meter (see run_experiments e1).
"""

import csv
import subprocess
import threading
import time

try:
    import psutil
except ImportError:
    psutil = None


def read_temp_c():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except (OSError, ValueError):
        return None


def read_throttled():
    """Raspberry Pi under-voltage / throttling flags, e.g. '0x0'; None elsewhere."""
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, universal_newlines=True, timeout=1.0).stdout
        return out.strip().split("=")[-1] or None
    except (OSError, subprocess.SubprocessError):
        return None


class ResourceSampler(threading.Thread):
    COLUMNS = ["t", "cpu_pct", "proc_cpu_pct", "rss_mb", "mem_pct",
               "net_tx_bytes", "net_rx_bytes", "temp_c"]

    def __init__(self, path, period=0.2, iface=None):
        super().__init__(daemon=True)
        self.path, self.period, self.iface = path, float(period), iface
        self.enabled = psutil is not None
        self.throttled_start = read_throttled()
        self._stop_evt = threading.Event()
        self._n, self._cpu, self._proc, self._temp_max = 0, 0.0, 0.0, None
        self._first = self._last = None           # (t, tx, rx)
        if not self.enabled:
            print("[resources] psutil not installed - resource logging disabled")

    def _net(self):
        per_nic = psutil.net_io_counters(pernic=True)
        if self.iface is None:
            self.iface = "wlan0" if "wlan0" in per_nic else ""
        c = per_nic.get(self.iface) if self.iface else psutil.net_io_counters()
        return (c.bytes_sent, c.bytes_recv) if c else (0, 0)

    def run(self):
        if not self.enabled:
            return
        proc = psutil.Process()
        proc.cpu_percent(None)
        psutil.cpu_percent(None)
        with open(self.path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(self.COLUMNS)
            while not self._stop_evt.wait(self.period):
                now = time.time()
                tx, rx = self._net()
                cpu, pcpu = psutil.cpu_percent(None), proc.cpu_percent(None)
                temp = read_temp_c()
                w.writerow(["%.3f" % now, cpu, pcpu, "%.1f" % (proc.memory_info().rss / 2 ** 20),
                            psutil.virtual_memory().percent, tx, rx, "" if temp is None else temp])
                self._n += 1
                self._cpu += cpu
                self._proc += pcpu
                if temp is not None:
                    self._temp_max = temp if self._temp_max is None else max(self._temp_max, temp)
                if self._first is None:
                    self._first = (now, tx, rx)
                self._last = (now, tx, rx)

    def stop(self):
        self._stop_evt.set()

    def summary(self):
        s = {"enabled": self.enabled, "samples": self._n,
             "throttled_start": self.throttled_start, "throttled_end": read_throttled()}
        if self._n:
            s.update(cpu_pct=round(self._cpu / self._n, 1), proc_cpu_pct=round(self._proc / self._n, 1),
                     temp_max_c=self._temp_max)
        if self._first and self._last and self._last[0] > self._first[0]:
            dt = self._last[0] - self._first[0]
            s.update(tx_mbit_s=round((self._last[1] - self._first[1]) * 8 / dt / 1e6, 3),
                     rx_mbit_s=round((self._last[2] - self._first[2]) * 8 / dt / 1e6, 3))
        return s
