"""
netprofile.py  -- reproducible network impairment (software, on the Pi)
=======================================================================
Every cloud decision that reaches the Pi gets a release time
    arrival + delay(t)
where t is the time since the run started; inside an outage window the
decision is dropped. The injected delay therefore adds to the whole
perception-decision loop exactly like extra network latency, and - because the
manager measures cloud silence and decision age on RELEASE - the arbitration
policy sees it. It does NOT model bandwidth limits or loss on the video uplink
(use real tc/netem for that, experiment E5).

Profiles (NET_PROFILE env var; INJECT_MS is added as a constant base delay):
  stable                          no impairment
  const:ms=100                    constant extra delay
  spike:ms=400,start=3,end=6      extra delay only inside the window
  ramp:ms=600,start=2,end=12      delay grows linearly from 0 to ms
  outage:start=3,end=5            all cloud decisions dropped inside the window
  jitter:mean=150,sd=80           random delay per decision, N(mean, sd) >= 0
"""

import heapq
import random
import threading
import time

from .spec import parse_spec


class NetProfile:
    DEFAULTS = {
        "stable": {},
        "const": {"ms": 0.0},
        "spike": {"ms": 400.0, "start": 3.0, "end": 6.0},
        "ramp": {"ms": 600.0, "start": 2.0, "end": 12.0},
        "outage": {"start": 3.0, "end": 5.0},
        "jitter": {"mean": 150.0, "sd": 80.0},
    }

    def __init__(self, name="stable", base_ms=0.0, seed=0, **params):
        if name not in self.DEFAULTS:
            raise ValueError("unknown network profile %r (use %s)" % (name, " | ".join(self.DEFAULTS)))
        if "" in params:
            params["ms"] = params.pop("")
        self.name, self.base = name, float(base_ms)
        self.p = dict(self.DEFAULTS[name], **params)
        self._rng = random.Random(seed)

    def nominal(self, t):
        """(delay_ms, dropping) at run time t; t=None = warm-up before the run clock starts."""
        n, p, b = self.name, self.p, self.base
        if t is None or n == "stable":
            return b, False
        if n == "const":
            return b + p["ms"], False
        if n == "spike":
            return b + (p["ms"] if p["start"] <= t < p["end"] else 0.0), False
        if n == "ramp":
            f = min(max((t - p["start"]) / (p["end"] - p["start"]), 0.0), 1.0)
            return b + f * p["ms"], False
        if n == "outage":
            return b, p["start"] <= t < p["end"]
        return b + p["mean"], False                   # jitter: the mean

    def sample(self, t):
        """Delay in ms for one decision arriving at run time t, or None = drop."""
        delay, drop = self.nominal(t)
        if drop:
            return None
        if self.name == "jitter" and t is not None:
            delay = self.base + max(0.0, self._rng.gauss(self.p["mean"], self.p["sd"]))
        return delay


def make_profile(spec="stable", base_ms=0.0, seed=0):
    name, params = parse_spec(spec)
    return NetProfile(name, base_ms, seed, **params)


class ImpairedChannel:
    """Thread-safe delay/drop buffer between the MQTT callback and the control loop."""

    def __init__(self, profile):
        self.profile = profile
        self.t0 = None
        self.n_arrived = self.n_dropped = 0
        self._heap, self._seq = [], 0
        self._lock = threading.Lock()

    def start(self, t0):
        """Start the run clock (the profile's time windows are relative to it)."""
        self.t0 = t0

    def _t(self, now):
        return None if self.t0 is None else now - self.t0

    def push(self, msg, now=None):
        now = time.time() if now is None else now
        delay = self.profile.sample(self._t(now))
        with self._lock:
            self.n_arrived += 1
            if delay is None:
                self.n_dropped += 1
                return
            heapq.heappush(self._heap, (now + delay / 1000.0, self._seq, msg))
            self._seq += 1

    def pop_due(self, now=None):
        now = time.time() if now is None else now
        out = []
        with self._lock:
            while self._heap and self._heap[0][0] <= now:
                out.append(heapq.heappop(self._heap)[2])
        return out

    def nominal(self, now):
        return self.profile.nominal(self._t(now))
