"""
spec.py  -- 'name:key=value,...' strings
========================================
Every experiment knob (perception module, arbitration policy, network profile)
fits into one environment variable, e.g.

  yolo:320               -> ("yolo",    {"": 320.0})
  timeout:tau=0.35       -> ("timeout", {"tau": 0.35})
  ewma:hi=300,lo=200     -> ("ewma",    {"hi": 300.0, "lo": 200.0})
"""


def _value(text):
    try:
        return float(text)
    except ValueError:
        return text


def parse_spec(spec):
    """Split a spec into (name, params). A bare value is stored under key ''."""
    name, _, rest = str(spec).strip().partition(":")
    params = {}
    for part in rest.split(","):
        part = part.strip()
        if not part:
            continue
        key, eq, value = part.partition("=")
        if not eq:
            key, value = "", key
        params[key.strip()] = _value(value.strip())
    return name.strip().lower(), params


def label(spec):
    """Short, filename-safe label: 'timeout:tau=0.5' -> 'timeout-tau0.5', 'yolo:320' -> 'yolo320'."""
    name, params = parse_spec(spec)
    out = name
    for key, value in params.items():
        text = "%g" % value if isinstance(value, float) else str(value)
        out += text if key == "" else "-%s%s" % (key, text)
    return out
