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

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "website", "static", "charts"
)

# ── Palette (matches website light theme) ──────────────────────────────

ZTENSOR_BLUE = "#2563EB"
BASELINE_GRAY = "#CBD5E1"
TEXT_COLOR = "#334155"
MUTED_COLOR = "#94A3B8"
GRID_COLOR = "#F1F5F9"

FORMAT_COLORS = {
    "ztensor": "#2563EB",
    "safetensors": "#DC2626",
    "pickle": "#16A34A",
    "npz": "#F59E0B",
    "gguf": "#9333EA",
    "onnx": "#EA580C",
    "hdf5": "#D97706",
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
    # Llama 3.2 1B shapes (~2.8 GB), median of 5 runs, 1 warmup.
    # .onnx measured at 1 GB: protobuf caps a message at 2 GB.
    (".zt", 2.27, 0.96, None, None),
    (".safetensors", 2.47, 1.00, 1.57, 1.59),
    (".pt", 2.29, 0.83, 1.60, None),
    (".npz", 2.33, 0.94, 0.80, None),
    (".gguf", 2.37, 0.92, 1.57, 2.52),
    (".onnx", 2.30, 0.82, 0.81, None),
    (".h5", 2.36, 0.95, 1.47, None),
]

ZTENSOR_DARK = "#1D4ED8"  # ztensor (default, zero-copy)
ZTENSOR_LIGHT = "#93C5FD"  # ztensor (zc off)
NATIVE_DARK = "#64748B"  # native zero-copy
NATIVE_LIGHT = "#CBD5E1"  # native copy


def draw_cross_format_read(path):
    labels = [d[0] for d in CROSS_FORMAT_DATA]
    zt_zc = [d[1] for d in CROSS_FORMAT_DATA]
    zt_cp = [d[2] for d in CROSS_FORMAT_DATA]
    nat_cp = [d[3] if d[3] is not None else 0 for d in CROSS_FORMAT_DATA]
    has_nat_cp = [d[3] is not None for d in CROSS_FORMAT_DATA]
    nat_zc = [d[4] if d[4] is not None else 0 for d in CROSS_FORMAT_DATA]
    has_nat_zc = [d[4] is not None for d in CROSS_FORMAT_DATA]

    x = np.arange(len(labels))
    width = 0.18

    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.bar(
        x - 1.5 * width,
        zt_zc,
        width,
        label="ztensor",
        color=ZTENSOR_DARK,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    ax.bar(
        x - 0.5 * width,
        zt_cp,
        width,
        label="ztensor (zc off)",
        color=ZTENSOR_LIGHT,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )

    # Native zero-copy: only draw where available
    nat_zc_colors = [NATIVE_DARK if h else "none" for h in has_nat_zc]
    ax.bar(
        x + 0.5 * width,
        nat_zc,
        width,
        label="ref. zero-copy",
        color=nat_zc_colors,
        edgecolor=["white" if h else "none" for h in has_nat_zc],
        linewidth=0.5,
        zorder=3,
    )

    # Native copy: only draw where available
    nat_cp_colors = [NATIVE_LIGHT if h else "none" for h in has_nat_cp]
    ax.bar(
        x + 1.5 * width,
        nat_cp,
        width,
        label="ref. copy",
        color=nat_cp_colors,
        edgecolor=["white" if h else "none" for h in has_nat_cp],
        linewidth=0.5,
        zorder=3,
    )

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
    ax.legend(
        handles=legend_elements,
        fontsize=8,
        frameon=False,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.0),
    )
    all_vals = (
        zt_zc + zt_cp + [v for v in nat_cp if v > 0] + [v for v in nat_zc if v > 0]
    )
    ax.set_ylim(0, max(all_vals) * 1.15)

    fig.tight_layout()
    fig.savefig(path, format="svg", transparent=True)
    plt.close(fig)
    print(f"  ✓ {path}")


# ── Chart 2: Write throughput by distribution ──────────────────────────

WRITE_DIST_DATA = [
    # name, large, mixed, small
    # 512MB, median of 5 runs, 1 warmup.
    # ztensor writes canonical form here (64 KiB placement); on `small`
    # that is 51k tiny tensors each rounded to a page, so the file is 6.4x
    # the payload and the write pays for every byte of it. At the 4 KiB
    # floor the same workload writes 1.32 GB/s into 1.21x. See the
    # alignment table in the benchmarks page.
    ("ztensor", 3.29, 3.62, 0.80),
    ("safetensors", 5.18, 6.27, 2.62),
    ("pickle", 5.91, 6.03, 2.86),
    ("npz", 1.10, 1.15, 0.54),
    ("gguf", 4.78, 6.25, 1.30),
    ("onnx", 0.29, 0.30, 0.35),
    ("hdf5", 6.13, 5.96, 0.28),
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
        return [
            blues[shade] if names[i] == "ztensor" else grays[shade]
            for i in range(len(vals))
        ]

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.bar(
        x - width,
        large,
        width,
        label="Large",
        color=bar_colors(large, 0),
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    ax.bar(
        x,
        mixed,
        width,
        label="Mixed",
        color=bar_colors(mixed, 1),
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    ax.bar(
        x + width,
        small,
        width,
        label="Small",
        color=bar_colors(small, 2),
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )

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
    ax.legend(
        handles=legend_elements,
        fontsize=9,
        frameon=False,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.0),
    )
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
