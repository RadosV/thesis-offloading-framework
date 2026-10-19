"""
Printable stop sign = navigation target for the YOLO workload
(COCO class 11 'stop sign'; the pretrained YOLOv8n recognises printed signs).

    python stop-sign-generated.py      -> stop_sign_A4.pdf (print at 100 %) + stop_sign_A4.png (preview)

The sign is 180 mm across the flats on A4. For a bigger target print it on A3
with "fit to page". Check detection with: bench_inference.py --live yolo:320
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

A4_MM = (210.0, 297.0)
SIGN_MM = 180.0                 # across the flats
BORDER = 0.045                  # white rim, fraction of the size
RED = (0.76, 0.11, 0.15)
HERE = os.path.dirname(os.path.abspath(__file__))


def octagon(cx, cy, across_flats):
    r = across_flats / 2.0 / np.cos(np.pi / 8)            # circumradius
    angles = np.pi / 8 + np.arange(8) * np.pi / 4        # flat top and bottom
    return np.column_stack([cx + r * np.cos(angles), cy + r * np.sin(angles)])


def main():
    fig = plt.figure(figsize=(A4_MM[0] / 25.4, A4_MM[1] / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, A4_MM[0])
    ax.set_ylim(0, A4_MM[1])
    ax.set_aspect("equal")
    ax.axis("off")

    cx, cy = A4_MM[0] / 2, A4_MM[1] / 2 + 10
    ax.add_patch(Polygon(octagon(cx, cy, SIGN_MM), closed=True, facecolor="white",
                         edgecolor=(0.6, 0.6, 0.6), linewidth=0.5))
    ax.add_patch(Polygon(octagon(cx, cy, SIGN_MM * (1 - 2 * BORDER)), closed=True, facecolor=RED))

    # scale the text so that "STOP" spans ~72 % of the sign width
    text = ax.text(cx, cy, "STOP", color="white", fontweight="bold", fontfamily="DejaVu Sans",
                   ha="center", va="center_baseline", fontsize=100)
    renderer = fig.canvas.get_renderer()
    width_px = text.get_window_extent(renderer).width
    target_px = 0.72 * SIGN_MM / A4_MM[0] * fig.get_figwidth() * fig.dpi
    text.set_fontsize(100 * target_px / width_px)

    ax.text(cx, 18, "Navigation target (YOLO, COCO class 11 'stop sign') - print at 100 % scale",
            ha="center", va="center", fontsize=7, color=(0.5, 0.5, 0.5))
    fig.savefig(os.path.join(HERE, "stop_sign_A4.pdf"))
    fig.savefig(os.path.join(HERE, "stop_sign_A4.png"), dpi=100)
    print("-> stop_sign_A4.pdf / stop_sign_A4.png in", HERE)


if __name__ == "__main__":
    main()
