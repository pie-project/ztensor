#!/usr/bin/env python3
"""Generate SVG benchmark charts for the README.

Uses the same data as the website's BenchmarkCharts.jsx.
Output goes to website/static/charts/.

Usage:
    python scripts/generate_readme_charts.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "website", "static", "charts")

# ── Palette (matches website light theme) ──────────────────────────────

ZTENSOR_BLUE = "#2563EB"
BASELINE_GRAY = "#CBD5E1"
TEXT_COLOR = "#334155"
MUTED_COLOR = "#94A3B8"
GRID_COLOR = "#F1F5F9"

FORMAT_COLORS = {
    "ztensor":     "#2563EB",
    "safetensors": "#DC2626",
    "pickle":      "#16A34A",
    "npz":         "#F59E0B",
    "gguf":        "#9333EA",
    "onnx":        "#EA580C",
    "hdf5":        "#D97706",
}


def _style_ax(ax, ylabel):
    ax.set_ylabel(ylabel, fontsize=10, color=TEXT_COLOR, labelpad=8)
    ax.tick_params(axis="both", labelsize=9, colors=MUTED_COLOR, length=0)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(MUTED_COLOR)
    ax.set_ylim(bottom=0)


# ── Chart 1: Cross-format read ─────────────────────────────────────────

CROSS_FORMAT_DATA = [
    # (label, ztensor GB/s, native GB/s | None)
    (".safetensors", 2.27, 1.48),
    (".pt",          2.26, 1.44),
    (".npz",         2.35, 1.15),
    (".gguf",        2.29, 2.34),
    (".onnx",        2.28, 0.79),
    (".h5",          2.41, 1.48),
]


def draw_cross_format_read(path):
    labels = [d[0] for d in CROSS_FORMAT_DATA]
    zt_vals = [d[1] for d in CROSS_FORMAT_DATA]
    native_vals = [d[2] for d in CROSS_FORMAT_DATA]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 3.5))
    bars_zt = ax.bar(x - width / 2, zt_vals, width, label="ztensor",
                     color=ZTENSOR_BLUE, edgecolor="white", linewidth=0.5,
                     zorder=3)
    bars_native = ax.bar(x + width / 2, native_vals, width,
                         label="reference impl.",
                         color=BASELINE_GRAY, edgecolor="white", linewidth=0.5,
                         zorder=3)


    _style_ax(ax, "Read Throughput (GB/s)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, color=TEXT_COLOR)
    ax.legend(fontsize=9, frameon=False, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, 1.0))
    ax.set_ylim(0, max(zt_vals + native_vals) * 1.18)

    fig.tight_layout()
    fig.savefig(path, format="svg", transparent=True)
    plt.close(fig)
    print(f"  ✓ {path}")


# ── Chart 2: Write throughput by distribution ──────────────────────────

WRITE_DIST_DATA = [
    # name, large, mixed, small
    ("ztensor",     3.60, 3.65, 1.43),
    ("safetensors", 1.72, 1.75, 1.30),
    ("pickle",      3.58, 3.65, 1.76),
    ("npz",         2.39, 2.38, 0.50),
    ("gguf",        3.81, 3.90, 1.01),
    ("onnx",        0.29, 0.29, 0.34),
    ("hdf5",        3.68, 3.67, 0.27),
]

DIST_LABELS = ["Large", "Mixed", "Small"]


def draw_write_throughput(path):
    names = [d[0] for d in WRITE_DIST_DATA]
    large = [d[1] for d in WRITE_DIST_DATA]
    mixed = [d[2] for d in WRITE_DIST_DATA]
    small = [d[3] for d in WRITE_DIST_DATA]

    x = np.arange(len(names))
    width = 0.25

    # Color: blue shades for ztensor row, gray shades for everything else
    def bar_colors(vals, shade):
        """shade: 0=large(darkest), 1=mixed, 2=small(lightest)"""
        blues = ["#1D4ED8", "#3B82F6", "#93C5FD"]
        grays = ["#94A3B8", "#CBD5E1", "#E2E8F0"]
        return [blues[shade] if names[i] == "ztensor" else grays[shade]
                for i in range(len(vals))]

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.bar(x - width, large, width, label="Large",
           color=bar_colors(large, 0), edgecolor="white", linewidth=0.5,
           zorder=3)
    ax.bar(x, mixed, width, label="Mixed",
           color=bar_colors(mixed, 1), edgecolor="white", linewidth=0.5,
           zorder=3)
    ax.bar(x + width, small, width, label="Small",
           color=bar_colors(small, 2), edgecolor="white", linewidth=0.5,
           zorder=3)

    _style_ax(ax, "Write Throughput (GB/s)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10, color=TEXT_COLOR)

    # Custom legend with blue shades
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1D4ED8", label="Large"),
        Patch(facecolor="#3B82F6", label="Mixed"),
        Patch(facecolor="#93C5FD", label="Small"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, frameon=False,
              loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    ax.set_ylim(0, max(large + mixed + small) * 1.12)

    fig.tight_layout()
    fig.savefig(path, format="svg", transparent=True)
    plt.close(fig)
    print(f"  ✓ {path}")


# ── Main ───────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Generating README charts…")
    draw_cross_format_read(os.path.join(OUTPUT_DIR, "cross_format_read.svg"))
    draw_write_throughput(os.path.join(OUTPUT_DIR, "write_throughput.svg"))
    print("Done.")


if __name__ == "__main__":
    main()
