"""
policies.py  -- framework extension point 2: arbitration policies
=================================================================
Called once per control cycle by the hybrid manager. A policy sees
  * how stale the latest cloud and local decisions are (age of information:
    now minus the capture time of the frame a decision is based on, in ms),
  * how long the cloud has been silent (time since the last released cloud
    decision, in s),
and returns which brain drives: "CLOUD" or "LOCAL". `reason` explains the last
switch and `state()` is written to the per-cycle log.

Policies (POLICY env var):
  timeout:tau=0.5              v1 baseline: cloud while silence < tau
  ewma:hi=300,lo=200,k=5       latency-aware: EWMA of cloud age with hysteresis
  cost:margin=30,k=5           cost model (literature): offload iff the cloud
                               delivers fresher decisions than local execution,
                               outage detection with an RFC 6298 adaptive timeout
"""

from .spec import parse_spec


def _fmt(x):
    return "" if x is None else "%.1f" % x


class Ewma:
    def __init__(self, alpha):
        self.alpha, self.value = float(alpha), None

    def update(self, x):
        if x is not None:
            self.value = x if self.value is None else self.alpha * x + (1 - self.alpha) * self.value
        return self.value


class Policy:
    name = "base"

    def __init__(self):
        self.reason = ""

    def on_cloud_message(self, now):
        """Called for every cloud decision released to the manager."""

    def decide(self, now, cloud_stale_ms, local_stale_ms, silence_s, current):
        raise NotImplementedError

    def state(self):
        return ""


class FixedTimeout(Policy):
    """v1 baseline: use the cloud while its last decision arrived less than tau s ago.
    Detects outages, but is blind to a constant added latency."""
    name = "timeout"

    def __init__(self, tau=0.5):
        super().__init__()
        self.tau = float(tau)

    def decide(self, now, cloud_stale_ms, local_stale_ms, silence_s, current):
        if silence_s < self.tau:
            self.reason = "cloud alive"
            return "CLOUD"
        self.reason = "silence"
        return "LOCAL"


class EwmaHysteresis(Policy):
    """Latency-aware: leave the cloud when the EWMA of its decision age exceeds
    `hi` (meant to be the crossover latency measured in RQ1) or it is silent for
    `tau`; return only after the EWMA stays below `lo` for `k` cycles."""
    name = "ewma"

    def __init__(self, hi=300, lo=200, alpha=0.2, tau=0.5, k=5):
        super().__init__()
        self.hi, self.lo, self.tau, self.k = float(hi), float(lo), float(tau), int(k)
        self.ewma = Ewma(alpha)
        self._good = 0

    def decide(self, now, cloud_stale_ms, local_stale_ms, silence_s, current):
        e = self.ewma.update(cloud_stale_ms)
        if e is None or silence_s >= self.tau:
            self._good = 0
            self.reason = "silence"
            return "LOCAL"
        if current == "CLOUD":
            if e > self.hi:
                self._good = 0
                self.reason = "latency"
                return "LOCAL"
            return "CLOUD"
        self._good = self._good + 1 if e < self.lo else 0
        if self._good >= self.k:
            self.reason = "recovered"
            return "CLOUD"
        return "LOCAL"

    def state(self):
        return "ewma=%s" % _fmt(self.ewma.value)


class CostModel(Policy):
    """Classic offloading criterion (Mach & Becvar 2017; MAUI, Cuervo et al. 2010):
    offload iff the remote path delivers fresher decisions than local execution,
        EWMA(cloud age) + margin < EWMA(local age),
    switching only after the preference held for k cycles (hysteresis).
    Outages are detected with an adaptive timeout on the gaps between cloud
    decisions, computed like TCP's retransmission timeout (RFC 6298):
        RTO = SRTT + 4 * RTTVAR, clamped to [rto_min, rto_max] seconds.
    `local_prior` (ms) is used while no local age is known (cold standby)."""
    name = "cost"

    def __init__(self, margin=30, alpha=0.2, k=5, rto_min=0.15, rto_max=1.0, local_prior=None):
        super().__init__()
        self.margin, self.k = float(margin), int(k)
        self.rto_min, self.rto_max = float(rto_min), float(rto_max)
        self.local_prior = None if local_prior is None else float(local_prior)
        self.cloud, self.local = Ewma(alpha), Ewma(alpha)
        self.srtt = self.rttvar = None
        self._last_msg = None
        self._pending = 0

    def on_cloud_message(self, now):
        if self._last_msg is not None:
            gap = now - self._last_msg
            if gap <= self.rto_max:          # an outage gap is not a sample (cf. Karn)
                if self.srtt is None:
                    self.srtt, self.rttvar = gap, gap / 2.0
                else:
                    self.rttvar = 0.75 * self.rttvar + 0.25 * abs(self.srtt - gap)
                    self.srtt = 0.875 * self.srtt + 0.125 * gap
        self._last_msg = now

    def rto(self):
        if self.srtt is None:
            return self.rto_max
        return min(max(self.srtt + 4.0 * self.rttvar, self.rto_min), self.rto_max)

    def decide(self, now, cloud_stale_ms, local_stale_ms, silence_s, current):
        c = self.cloud.update(cloud_stale_ms)
        l = self.local.update(local_stale_ms)
        if l is None:
            l = self.local_prior
        if c is None or silence_s >= self.rto():
            self._pending = 0
            self.reason = "silence"
            return "LOCAL"
        if l is None:
            want, why = "CLOUD", "no local estimate"
        elif c + self.margin < l:
            want, why = "CLOUD", "cloud fresher"
        else:
            want, why = "LOCAL", "local fresher"
        if want == current:
            self._pending = 0
            return current
        self._pending += 1
        if self._pending >= self.k:
            self._pending = 0
            self.reason = why
            return want
        return current

    def state(self):
        return "c=%s;l=%s;rto=%.3f" % (_fmt(self.cloud.value), _fmt(self.local.value), self.rto())


POLICIES = {cls.name: cls for cls in (FixedTimeout, EwmaHysteresis, CostModel)}


def make_policy(spec="timeout:tau=0.5"):
    name, params = parse_spec(spec)
    if name not in POLICIES:
        raise ValueError("unknown policy %r (use %s)" % (spec, " | ".join(POLICIES)))
    params.pop("", None)
    return POLICIES[name](**params)
