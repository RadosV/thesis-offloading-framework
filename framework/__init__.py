"""
Latency-aware offloading framework (v2).

Shared building blocks used by the Pi hybrid manager, the edge brain, the
experiment runner and the benchmarks:

  perception  extension point 1: what is computed (ArUco, YOLO, ...)
  policies    extension point 2: where it is computed (arbitration policies)
  protocol    wire formats Pi <-> edge server (frame header, cloud reply)
  netprofile  reproducible network impairment (delay / outage schedules)
  camera      newest-frame camera grabber (camera, video, image, synthetic)
  resources   CPU / memory / network / temperature sampler
  spec        'name:key=value' strings for all of the above
"""
