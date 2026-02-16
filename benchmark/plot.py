import os
import numpy as np

# --- PLOT STYLE CONSTANTS ---

COLORS = {
    'ztensor': '#2563EB',
    'ztensor_zstd': '#3B82F6',
    'ztensor_zerocopy': '#60A5FA',
    'safetensors': '#DC2626',
    'pickle': '#16A34A',
    'hdf5': '#9333EA',
    'gguf': '#EA580C',
    'npz': '#0D9488',
    'onnx': '#7C3AED',
    'zt_read_st': '#F59E0B',
    'zt_read_pt': '#10B981',
    'zt_read_gguf': '#F97316',
    'zt_read_npz': '#14B8A6',
    'zt_read_onnx': '#A78BFA',
}

MARKERS = {
    'ztensor': 'o',
    'ztensor_zstd': 'D',
    'ztensor_zerocopy': '*',
    'safetensors': 's',
    'pickle': '^',
    'hdf5': 'v',
    'gguf': 'P',
    'npz': 'h',
    'onnx': 'd',
    'zt_read_st': 'D',
    'zt_read_pt': '^',
    'zt_read_gguf': 'P',
    'zt_read_npz': 'h',
    'zt_read_onnx': 'd',
}

LABELS = {
    'ztensor': 'ztensor',
    'ztensor_zstd': 'ztensor (zstd-3)',
    'ztensor_zerocopy': 'ztensor (zero-copy)',
    'safetensors': 'safetensors',
    'pickle': 'pickle',
    'hdf5': 'hdf5',
    'gguf': 'gguf',
    'npz': 'npz',
    'onnx': 'onnx',
    'zt_read_st': 'ztensor\u2192safetensors',
    'zt_read_pt': 'ztensor\u2192pytorch',
    'zt_read_gguf': 'ztensor\u2192gguf',
    'zt_read_npz': 'ztensor\u2192npz',
    'zt_read_onnx': 'ztensor\u2192onnx',
}


def _setup_plot_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
    })


def draw_plot(csv_path: str = "bench_out/sweep_results.csv", output_dir: str = "bench_out/plots"):
    """
    Creates benchmark plots from sweep results CSV.
    Generates individual plots + a clean website summary figure.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    # Backward compatibility: add default columns if missing
    if 'DataStyle' not in df.columns:
        df['DataStyle'] = 'random'
    if 'Scenario' not in df.columns:
        df['Scenario'] = 'full_load'

    _setup_plot_style()

    # === 1. Line plots per distribution (full_load, random only) ===
    base = df[(df['DataStyle'] == 'random') & (df['Scenario'] == 'full_load')]
    distributions = base['Distribution'].unique()
    formats = base['Format'].unique()

    for dist in distributions:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        subset = base[base['Distribution'] == dist]

        # Write throughput
        ax = axes[0]
        for fmt in formats:
            data = subset[subset['Format'] == fmt].sort_values('SizeMB')
            if data.empty:
                continue
            ax.plot(
                data['SizeMB'], data['WriteGBs'],
                marker=MARKERS.get(fmt, 'o'), markersize=8, linewidth=2.5,
                color=COLORS.get(fmt, '#666'),
                label=LABELS.get(fmt, fmt), alpha=0.9,
            )
        ax.set_xlabel('Data Size (MB)')
        ax.set_ylabel('Write Throughput (GB/s)')
        ax.set_title(f'Write Performance \u2014 {dist.title()} Tensors')
        ax.legend(loc='best', framealpha=0.9)
        ax.set_ylim(bottom=0)
        ax.set_facecolor('#FAFAFA')

        # Read throughput
        ax = axes[1]
        for fmt in formats:
            data = subset[subset['Format'] == fmt].sort_values('SizeMB')
            if data.empty:
                continue
            read_throughput = (data['SizeMB'] / 1024) / data['ReadSeconds']
            ax.plot(
                data['SizeMB'], read_throughput,
                marker=MARKERS.get(fmt, 'o'), markersize=8, linewidth=2.5,
                color=COLORS.get(fmt, '#666'),
                label=LABELS.get(fmt, fmt), alpha=0.9,
            )
        ax.set_xlabel('Data Size (MB)')
        ax.set_ylabel('Read Throughput (GB/s)')
        ax.set_title(f'Read Performance \u2014 {dist.title()} Tensors')
        ax.legend(loc='best', framealpha=0.9)
        ax.set_ylim(bottom=0)
        ax.set_facecolor('#FAFAFA')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/throughput_{dist}.png")
        plt.savefig(f"{output_dir}/throughput_{dist}.pdf")
        plt.close()

        # Read-only single panel
        fig, ax = plt.subplots(figsize=(8, 5))
        for fmt in formats:
            data = subset[subset['Format'] == fmt].sort_values('SizeMB')
            if data.empty:
                continue
            read_throughput = (data['SizeMB'] / 1024) / data['ReadSeconds']
            linewidth = 3.0 if fmt.startswith('ztensor') and fmt != 'ztensor_zstd' else 2.0
            ax.plot(
                data['SizeMB'], read_throughput,
                marker=MARKERS.get(fmt, 'o'), markersize=8, linewidth=linewidth,
                color=COLORS.get(fmt, '#666'),
                label=LABELS.get(fmt, fmt), alpha=0.9,
            )
        ax.set_xlabel('Data Size (MB)')
        ax.set_ylabel('Read Throughput (GB/s)')
        ax.set_title(f'Read Throughput \u2014 {dist.title()} Tensors')
        ax.legend(loc='best', framealpha=0.9)
        ax.set_ylim(bottom=0)
        ax.set_facecolor('#FAFAFA')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/read_{dist}.png")
        plt.savefig(f"{output_dir}/read_{dist}.pdf")
        plt.close()

    # === 2. Website summary figure (unchanged: mixed, full_load, random) ===
    _draw_website_figure(df, output_dir)

    # === 3. Read hero + write summary for restructured docs ===
    _draw_read_hero(df, output_dir)
    _draw_write_summary(df, output_dir)

    # === 4. Feature-specific plots ===
    _draw_compression_comparison(df, output_dir)
    _draw_zerocopy_comparison(df, output_dir)
    _draw_selective_comparison(df, output_dir)
    _draw_crossformat_comparison(df, output_dir)

    print(f"Plots saved to {output_dir}/")


def _draw_website_figure(df, output_dir):
    """Generate the clean summary figure used on the website."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Use full_load, random, mixed distribution
    subset = df[(df.get('Scenario', 'full_load') == 'full_load') &
                (df.get('DataStyle', 'random') == 'random') &
                (df['Distribution'] == 'mixed')]

    if 'Scenario' in df.columns:
        subset = df[(df['Scenario'] == 'full_load') &
                    (df['DataStyle'] == 'random') &
                    (df['Distribution'] == 'mixed')]
    else:
        subset = df[df['Distribution'] == 'mixed']

    if subset.empty:
        print("Warning: No 'mixed' distribution data for website figure")
        return

    show_formats = [f for f in ['ztensor', 'ztensor_zstd', 'safetensors', 'pickle', 'hdf5', 'gguf']
                    if f in subset['Format'].unique()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: Read throughput
    ax = axes[0]
    for fmt in show_formats:
        data = subset[subset['Format'] == fmt].sort_values('SizeMB')
        if data.empty:
            continue
        read_tp = (data['SizeMB'] / 1024) / data['ReadSeconds']
        linewidth = 3.0 if fmt.startswith('ztensor') else 2.0
        ax.plot(
            data['SizeMB'], read_tp,
            marker=MARKERS.get(fmt, 'o'), markersize=9, linewidth=linewidth,
            color=COLORS.get(fmt, '#666'),
            label=LABELS.get(fmt, fmt), alpha=0.9,
        )

    ax.set_xlabel('Size (MB)')
    ax.set_ylabel('Read Throughput (GB/s)')
    ax.set_title('Read Performance (Mixed)')
    ax.legend(loc='best', framealpha=0.9, fontsize=10)
    ax.set_ylim(bottom=0)
    ax.set_facecolor('#FAFAFA')

    # Right: Write throughput
    ax = axes[1]
    for fmt in show_formats:
        data = subset[subset['Format'] == fmt].sort_values('SizeMB')
        if data.empty:
            continue
        linewidth = 3.0 if fmt.startswith('ztensor') else 2.0
        ax.plot(
            data['SizeMB'], data['WriteGBs'],
            marker=MARKERS.get(fmt, 'o'), markersize=9, linewidth=linewidth,
            color=COLORS.get(fmt, '#666'),
            label=LABELS.get(fmt, fmt), alpha=0.9,
        )

    ax.set_xlabel('Size (MB)')
    ax.set_ylabel('Write Throughput (GB/s)')
    ax.set_title('Write Performance (Mixed)')
    ax.legend(loc='best', framealpha=0.9, fontsize=10)
    ax.set_ylim(bottom=0)
    ax.set_facecolor('#FAFAFA')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/benchmark.png")
    plt.savefig(f"{output_dir}/benchmark.pdf")
    plt.close()


def _draw_read_hero(df, output_dir):
    """Large single-panel read throughput plot for the hero section."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    subset = df[(df.get('Scenario', 'full_load') == 'full_load') &
                (df.get('DataStyle', 'random') == 'random') &
                (df['Distribution'] == 'mixed')]
    if 'Scenario' in df.columns:
        subset = df[(df['Scenario'] == 'full_load') &
                    (df['DataStyle'] == 'random') &
                    (df['Distribution'] == 'mixed')]

    if subset.empty:
        return

    show_formats = [f for f in ['ztensor', 'ztensor_zstd', 'safetensors', 'pickle', 'hdf5', 'gguf']
                    if f in subset['Format'].unique()]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for fmt in show_formats:
        data = subset[subset['Format'] == fmt].sort_values('SizeMB')
        if data.empty:
            continue
        read_tp = (data['SizeMB'] / 1024) / data['ReadSeconds']
        linewidth = 3.5 if fmt == 'ztensor' else (2.5 if fmt == 'ztensor_zstd' else 2.0)
        ax.plot(
            data['SizeMB'], read_tp,
            marker=MARKERS.get(fmt, 'o'), markersize=10, linewidth=linewidth,
            color=COLORS.get(fmt, '#666'),
            label=LABELS.get(fmt, fmt), alpha=0.9,
        )

    ax.set_xlabel('Data Size (MB)')
    ax.set_ylabel('Read Throughput (GB/s)')
    ax.set_title('Read Throughput \u2014 Mixed Tensors')
    ax.legend(loc='best', framealpha=0.9, fontsize=11)
    ax.set_ylim(bottom=0)
    ax.set_facecolor('#FAFAFA')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/read_hero.png")
    plt.savefig(f"{output_dir}/read_hero.pdf")
    plt.close()


def _draw_write_summary(df, output_dir):
    """Single-panel write throughput plot for the consolidated write section."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    subset = df[(df.get('Scenario', 'full_load') == 'full_load') &
                (df.get('DataStyle', 'random') == 'random') &
                (df['Distribution'] == 'mixed')]
    if 'Scenario' in df.columns:
        subset = df[(df['Scenario'] == 'full_load') &
                    (df['DataStyle'] == 'random') &
                    (df['Distribution'] == 'mixed')]

    if subset.empty:
        return

    show_formats = [f for f in ['ztensor', 'ztensor_zstd', 'safetensors', 'pickle', 'hdf5', 'gguf']
                    if f in subset['Format'].unique()]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for fmt in show_formats:
        data = subset[subset['Format'] == fmt].sort_values('SizeMB')
        if data.empty:
            continue
        linewidth = 3.5 if fmt == 'ztensor' else (2.5 if fmt == 'ztensor_zstd' else 2.0)
        ax.plot(
            data['SizeMB'], data['WriteGBs'],
            marker=MARKERS.get(fmt, 'o'), markersize=10, linewidth=linewidth,
            color=COLORS.get(fmt, '#666'),
            label=LABELS.get(fmt, fmt), alpha=0.9,
        )

    ax.set_xlabel('Data Size (MB)')
    ax.set_ylabel('Write Throughput (GB/s)')
    ax.set_title('Write Throughput \u2014 Mixed Tensors')
    ax.legend(loc='best', framealpha=0.9, fontsize=11)
    ax.set_ylim(bottom=0)
    ax.set_facecolor('#FAFAFA')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/write_summary.png")
    plt.savefig(f"{output_dir}/write_summary.pdf")
    plt.close()


def _draw_compression_comparison(df, output_dir):
    """Bar chart: file size for random vs structured data at 1024MB mixed."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if 'DataStyle' not in df.columns:
        return

    full_load = df[df['Scenario'] == 'full_load']

    # Get random and structured data at the largest overlapping size, mixed distribution
    random_data = full_load[(full_load['DataStyle'] == 'random') &
                            (full_load['Distribution'] == 'mixed')]
    structured_data = full_load[(full_load['DataStyle'] == 'structured') &
                                (full_load['Distribution'] == 'mixed')]

    if random_data.empty or structured_data.empty:
        return

    # Find largest common size
    common_sizes = set(random_data['SizeMB'].unique()) & set(structured_data['SizeMB'].unique())
    if not common_sizes:
        return
    target_size = max(common_sizes)

    random_row = random_data[random_data['SizeMB'] == target_size]
    struct_row = structured_data[structured_data['SizeMB'] == target_size]

    show_formats = ['ztensor', 'ztensor_zstd', 'safetensors', 'pickle', 'hdf5', 'gguf']
    formats_present = [f for f in show_formats
                       if f in random_row['Format'].values and f in struct_row['Format'].values]

    if not formats_present:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(formats_present))
    width = 0.35

    random_sizes = []
    struct_sizes = []
    for fmt in formats_present:
        r = random_row[random_row['Format'] == fmt]
        s = struct_row[struct_row['Format'] == fmt]
        random_sizes.append(r['FileSizeGB'].values[0] * 1024 if not r.empty else 0)
        struct_sizes.append(s['FileSizeGB'].values[0] * 1024 if not s.empty else 0)

    bars1 = ax.bar(x - width/2, random_sizes, width, label='Random data',
                   color='#94A3B8', alpha=0.8)
    bars2 = ax.bar(x + width/2, struct_sizes, width, label='Structured data',
                   color='#2563EB', alpha=0.8)

    ax.set_xlabel('Format')
    ax.set_ylabel('File Size (MB)')
    ax.set_title(f'Compression: Random vs Structured Data ({target_size}MB Mixed)')
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(f, f) for f in formats_present], rotation=15, ha='right')
    ax.legend()
    ax.set_facecolor('#FAFAFA')

    # Add size labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/compression_comparison.png")
    plt.savefig(f"{output_dir}/compression_comparison.pdf")
    plt.close()


def _draw_zerocopy_comparison(df, output_dir):
    """Bar chart: read throughput for copy=True vs copy=False (zero-copy)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if 'Scenario' not in df.columns:
        return

    full_load = df[(df['Scenario'] == 'full_load') & (df['DataStyle'] == 'random') &
                   (df['Distribution'] == 'mixed') & (df['Format'] == 'ztensor')]
    zero_copy = df[(df['Scenario'] == 'zero_copy') & (df['DataStyle'] == 'random') &
                   (df['Distribution'] == 'mixed') & (df['Format'] == 'ztensor')]

    if full_load.empty or zero_copy.empty:
        return

    common_sizes = sorted(set(full_load['SizeMB'].unique()) & set(zero_copy['SizeMB'].unique()))
    if not common_sizes:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(common_sizes))
    width = 0.35

    copy_tp = []
    zerocopy_tp = []
    for size in common_sizes:
        fl = full_load[full_load['SizeMB'] == size]
        zc = zero_copy[zero_copy['SizeMB'] == size]
        copy_tp.append((size / 1024) / fl['ReadSeconds'].values[0] if not fl.empty else 0)
        zerocopy_tp.append((size / 1024) / zc['ReadSeconds'].values[0] if not zc.empty else 0)

    bars1 = ax.bar(x - width/2, copy_tp, width, label='ztensor (copy=True)',
                   color='#2563EB', alpha=0.8)
    bars2 = ax.bar(x + width/2, zerocopy_tp, width, label='ztensor (zero-copy)',
                   color='#60A5FA', alpha=0.8)

    ax.set_xlabel('Data Size (MB)')
    ax.set_ylabel('Read Throughput (GB/s)')
    ax.set_title('Zero-Copy Read Performance')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s}MB' for s in common_sizes])
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.set_facecolor('#FAFAFA')

    # Add throughput labels
    for bar in list(bars1) + list(bars2):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/zerocopy_comparison.png")
    plt.savefig(f"{output_dir}/zerocopy_comparison.pdf")
    plt.close()


def _draw_selective_comparison(df, output_dir):
    """Bar chart: time to load 10% of tensors per format."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if 'Scenario' not in df.columns:
        return

    selective = df[(df['Scenario'] == 'selective') & (df['Distribution'] == 'mixed')]

    if selective.empty:
        return

    # Use the largest available size
    target_size = selective['SizeMB'].max()
    subset = selective[selective['SizeMB'] == target_size]

    if subset.empty:
        return

    show_formats = ['ztensor', 'safetensors', 'pickle', 'hdf5', 'gguf']
    formats_present = [f for f in show_formats if f in subset['Format'].values]

    if not formats_present:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(formats_present))

    latencies = []
    bar_colors = []
    for fmt in formats_present:
        row = subset[subset['Format'] == fmt]
        latencies.append(row['ReadSeconds'].values[0] if not row.empty else 0)
        bar_colors.append(COLORS.get(fmt, '#666'))

    bars = ax.bar(x, latencies, color=bar_colors, alpha=0.85)

    ax.set_xlabel('Format')
    ax.set_ylabel('Read Latency (seconds)')
    ax.set_title(f'Selective Loading: 10% of Tensors ({target_size}MB Mixed)')
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(f, f) for f in formats_present])
    ax.set_facecolor('#FAFAFA')

    # Add latency labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}s', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/selective_loading.png")
    plt.savefig(f"{output_dir}/selective_loading.pdf")
    plt.close()


def _draw_crossformat_comparison(df, output_dir):
    """Grouped bar chart: ztensor reading each format vs native reader."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if 'Scenario' not in df.columns:
        return

    base = df[(df['Scenario'] == 'full_load') & (df['DataStyle'] == 'random') &
              (df['Distribution'] == 'mixed')]

    if base.empty:
        return

    # Find the best size with cross-format data
    cross_fmts = {'zt_read_st', 'zt_read_pt', 'zt_read_gguf', 'zt_read_npz', 'zt_read_onnx'}
    has_cross = base[base['Format'].isin(cross_fmts)]
    if has_cross.empty:
        return

    target_size = 512
    available = has_cross['SizeMB'].unique()
    if target_size not in available:
        target_size = max(available)

    subset = base[base['SizeMB'] == target_size]

    # Build paired data: (format_label, ztensor_throughput, native_throughput)
    pairs = []

    # .zt (native only, no "native reader" comparison)
    zt_row = subset[subset['Format'] == 'ztensor']
    if not zt_row.empty:
        zt_tp = (target_size / 1024) / zt_row['ReadSeconds'].values[0]
        pairs.append(('.zt', zt_tp, None))

    # .safetensors
    zt_st = subset[subset['Format'] == 'zt_read_st']
    native_st = subset[subset['Format'] == 'safetensors']
    if not zt_st.empty:
        zt_tp = (target_size / 1024) / zt_st['ReadSeconds'].values[0]
        native_tp = (target_size / 1024) / native_st['ReadSeconds'].values[0] if not native_st.empty else None
        pairs.append(('.safetensors', zt_tp, native_tp))

    # .pt
    zt_pt = subset[subset['Format'] == 'zt_read_pt']
    native_pt = subset[subset['Format'] == 'pickle']
    if not zt_pt.empty:
        zt_tp = (target_size / 1024) / zt_pt['ReadSeconds'].values[0]
        native_tp = (target_size / 1024) / native_pt['ReadSeconds'].values[0] if not native_pt.empty else None
        pairs.append(('.pt', zt_tp, native_tp))

    # .gguf
    zt_gguf = subset[subset['Format'] == 'zt_read_gguf']
    native_gguf = subset[subset['Format'] == 'gguf']
    if not zt_gguf.empty:
        zt_tp = (target_size / 1024) / zt_gguf['ReadSeconds'].values[0]
        native_tp = (target_size / 1024) / native_gguf['ReadSeconds'].values[0] if not native_gguf.empty else None
        pairs.append(('.gguf', zt_tp, native_tp))

    # .npz
    zt_npz = subset[subset['Format'] == 'zt_read_npz']
    native_npz = subset[subset['Format'] == 'npz']
    if not zt_npz.empty:
        zt_tp = (target_size / 1024) / zt_npz['ReadSeconds'].values[0]
        native_tp = (target_size / 1024) / native_npz['ReadSeconds'].values[0] if not native_npz.empty else None
        pairs.append(('.npz', zt_tp, native_tp))

    # .onnx
    zt_onnx = subset[subset['Format'] == 'zt_read_onnx']
    native_onnx = subset[subset['Format'] == 'onnx']
    if not zt_onnx.empty:
        zt_tp = (target_size / 1024) / zt_onnx['ReadSeconds'].values[0]
        native_tp = (target_size / 1024) / native_onnx['ReadSeconds'].values[0] if not native_onnx.empty else None
        pairs.append(('.onnx', zt_tp, native_tp))

    if not pairs:
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(pairs))
    width = 0.35

    zt_vals = [p[1] for p in pairs]
    native_vals = [p[2] if p[2] is not None else 0 for p in pairs]
    labels = [p[0] for p in pairs]

    bars1 = ax.bar(x - width/2, zt_vals, width, label='ztensor', color='#2563EB', alpha=0.9)
    bars2 = ax.bar(x + width/2, native_vals, width, label='Native reader', color='#94A3B8', alpha=0.8)

    # Hide the native bar for .zt (no comparison)
    for i, p in enumerate(pairs):
        if p[2] is None:
            bars2[i].set_alpha(0)

    ax.set_xlabel('Source Format')
    ax.set_ylabel('Read Throughput (GB/s)')
    ax.set_title(f'Cross-Format Reading \u2014 {target_size}MB Mixed')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='best', framealpha=0.9)
    ax.set_ylim(bottom=0)
    ax.set_facecolor('#FAFAFA')

    # Add throughput labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    for i, bar in enumerate(bars2):
        if pairs[i][2] is not None:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/crossformat_comparison.png")
    plt.savefig(f"{output_dir}/crossformat_comparison.pdf")
    plt.close()
