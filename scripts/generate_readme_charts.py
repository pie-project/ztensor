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
    # (label, zt_zerocopy, zt_copy, native_copy, native_zerocopy | None)
    # Llama 3.2 1B shapes (~2.8 GB), median of 5 runs, 2 warmup
    (".safetensors", 2.19, 1.46, 1.33, 1.35),
    (".pt",          2.04, 1.33, 0.89, None),
    (".npz",         2.11, 1.41, 1.04, None),
    (".gguf",        2.11, 1.38, 1.39, 2.15),
    (".onnx",        2.07, 1.29, 0.76, None),
    (".h5",          1.96, 1.30, 1.35, None),
]

ZTENSOR_DARK = "#1D4ED8"   # ztensor (default, zero-copy)
ZTENSOR_LIGHT = "#93C5FD"  # ztensor (zc off)
NATIVE_DARK = "#64748B"    # native zero-copy
NATIVE_LIGHT = "#CBD5E1"   # native copy


def draw_cross_format_read(path):
    labels = [d[0] for d in CROSS_FORMAT_DATA]
    zt_zc = [d[1] for d in CROSS_FORMAT_DATA]
    zt_cp = [d[2] for d in CROSS_FORMAT_DATA]
    nat_cp = [d[3] for d in CROSS_FORMAT_DATA]
    nat_zc = [d[4] if d[4] is not None else 0 for d in CROSS_FORMAT_DATA]
    has_nat_zc = [d[4] is not None for d in CROSS_FORMAT_DATA]

    x = np.arange(len(labels))
    width = 0.18

    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.bar(x - 1.5 * width, zt_zc, width, label="ztensor",
           color=ZTENSOR_DARK, edgecolor="white", linewidth=0.5, zorder=3)
    ax.bar(x - 0.5 * width, zt_cp, width, label="ztensor (zc off)",
           color=ZTENSOR_LIGHT, edgecolor="white", linewidth=0.5, zorder=3)

    # Native zero-copy: only draw where available
    nat_zc_colors = [NATIVE_DARK if h else "none" for h in has_nat_zc]
    ax.bar(x + 0.5 * width, nat_zc, width, label="ref. zero-copy",
           color=nat_zc_colors, edgecolor=["white" if h else "none" for h in has_nat_zc],
           linewidth=0.5, zorder=3)

    ax.bar(x + 1.5 * width, nat_cp, width, label="ref. copy",
           color=NATIVE_LIGHT, edgecolor="white", linewidth=0.5, zorder=3)

    _style_ax(ax, "Read Throughput (GB/s)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, color=TEXT_COLOR)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=ZTENSOR_DARK, label="ztensor"),
        Patch(facecolor=ZTENSOR_LIGHT, label="ztensor (zc off)"),
        Patch(facecolor=NATIVE_DARK, label="ref. zero-copy"),
        Patch(facecolor=NATIVE_LIGHT, label="ref. copy"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, frameon=False,
              loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0))
    all_vals = zt_zc + zt_cp + nat_cp + [v for v in nat_zc if v > 0]
    ax.set_ylim(0, max(all_vals) * 1.15)

    fig.tight_layout()
    fig.savefig(path, format="svg", transparent=True)
    plt.close(fig)
    print(f"  ✓ {path}")


# ── Chart 2: Write throughput by distribution ──────────────────────────

WRITE_DIST_DATA = [
    # name, large, mixed, small
    # 512MB, median of 5 runs, 2 warmup
    ("ztensor",     3.62, 3.65, 1.42),
    ("safetensors", 1.72, 1.77, 1.48),
    ("pickle",      3.62, 3.68, 2.00),
    ("npz",         2.40, 2.40, 0.51),
    ("gguf",        3.85, 3.86, 1.06),
    ("onnx",        0.28, 0.29, 0.32),
    ("hdf5",        3.67, 3.69, 0.27),
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
