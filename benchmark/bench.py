import os
import sys
import time
import numpy as np
import argparse
import threading
from typing import Dict, Union

# Note: h5py, gguf, ztensor, torch are imported lazily in benchmark functions

# Global backend setting: 'numpy' or 'torch'
BACKEND = 'numpy'

# --- UTILITIES ---

def clear_cache_robust():
    """
    Clears file system cache without requiring sudo.
    Allocates a buffer larger than RAM (or a significant chunk) and reads random data.
    """
    # Quick "sudo" attempt for Linux if user runs as root
    if sys.platform == "linux" and os.geteuid() == 0:
        os.system('echo 3 > /proc/sys/vm/drop_caches')
        return

    # Fallback: Read a 4GB temp file to evict pages (adjust based on your RAM)
    # This acts as a "Cache Buster"
    try:
        temp_filename = "cache_buster.tmp"
        # Write only if doesn't exist to save time
        if not os.path.exists(temp_filename):
            with open(temp_filename, "wb") as f:
                f.write(os.urandom(1024 * 1024 * 100)) # 100MB chunk for demo, increase for real bench
        
        with open(temp_filename, "rb") as f:
            while f.read(1024 * 1024 * 100): pass
    except:
        pass


import pandas as pd
import matplotlib.pyplot as plt

def generate_tensor_dict(total_size_mb: int, distribution: str = "mixed") -> Dict[str, Union[np.ndarray, 'torch.Tensor']]:
    """
    Generates tensors summing to total_size_mb.
    distribution:
      - 'mixed': Realistic mix (some large weights, some small biases/metadata)
      - 'small': Many small tensors (1KB - 100KB). Stresses metadata/parsing.
      - 'large': Few large tensors (10MB - 100MB). Stresses raw BW.
    
    Uses global BACKEND setting to determine tensor type ('numpy' or 'torch').
    """
    print(f"[{distribution.upper()}] Generating {total_size_mb}MB of synthetic data (backend={BACKEND})...")
    tensors = {}
    remaining_bytes = total_size_mb * 1024 * 1024
    i = 0
    
    while remaining_bytes > 0:
        if distribution == "mixed":
            if remaining_bytes > 100 * 1024 * 1024:
                shape = (5000, 5000) 
            elif remaining_bytes > 10 * 1024 * 1024:
                 shape = (1000, 2500) 
            else:
                elems = remaining_bytes // 4
                shape = (elems,)
                
        elif distribution == "large":
            # Target ~50MB chunks
            target_bytes = 50 * 1024 * 1024
            if remaining_bytes < target_bytes:
                target_bytes = remaining_bytes
            elems = target_bytes // 4
            shape = (elems,)

        elif distribution == "small":
            # Target ~10KB chunks
            target_bytes = 10 * 1024 
            if remaining_bytes < target_bytes:
                target_bytes = remaining_bytes
            elems = target_bytes // 4
            shape = (elems,)
        
        if np.prod(shape) == 0: break
        
        # Generate as numpy first, then convert if needed
        t_np = np.random.randn(*shape).astype(np.float32)
        
        if BACKEND == 'torch':
            import torch
            t = torch.from_numpy(t_np)
            remaining_bytes -= t.numel() * t.element_size()
        else:
            t = t_np
            remaining_bytes -= t.nbytes
        
        tensors[f"layer_{i}.weight"] = t
        i += 1
    return tensors

# --- BENCHMARK FUNCTIONS ---

def benchmark_write(format_name, tensors, filepath):
    import ztensor
    import h5py
    import gguf
    
    start = time.perf_counter()
    
    if format_name.startswith("ztensor"):
        compress = False
        if format_name == "ztensor_zstd":
            compress = True # Level 3
        elif format_name == "ztensor_fast":
             compress = 1 # Level 1
        elif format_name == "ztensor_max":
             compress = 19 # Level 19

        with ztensor.Writer(filepath) as w:
            for name, t in tensors.items():
                w.add_tensor(name, t, compress=compress)

    elif format_name == "safetensors":
        if BACKEND == 'torch':
            import safetensors.torch
            safetensors.torch.save_file(tensors, filepath)
        else:
            import safetensors.numpy
            safetensors.numpy.save_file(tensors, filepath)
        
    elif format_name == "pickle":
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(tensors, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    elif format_name == "hdf5":
        with h5py.File(filepath, "w") as f:
            for name, t in tensors.items():
                # Convert torch tensors to numpy for hdf5
                data = t.numpy() if BACKEND == 'torch' else t
                f.create_dataset(name, data=data)
                
    elif format_name == "gguf":
        gw = gguf.GGUFWriter(filepath, "benchmark_model")
        for name, t in tensors.items():
            # gguf requires numpy arrays
            data = t.numpy() if BACKEND == 'torch' else t
            gw.add_tensor(name, data)
        gw.write_header_to_file()
        gw.write_kv_data_to_file()
        gw.write_tensors_to_file()
        gw.close()
    
    end = time.perf_counter()
    size_gb = os.path.getsize(filepath) / (1024**3)
    duration = end - start
    return size_gb / duration, size_gb 

def benchmark_read(format_name, filepath):
    import ztensor
    import h5py
    import gguf
    
    start = time.perf_counter()
    loaded_tensors = {}

    if format_name.startswith("ztensor"):
        if BACKEND == 'torch':
            import ztensor.torch
            loaded_tensors = ztensor.torch.load_file(filepath)
        else:
            import ztensor.numpy
            loaded_tensors = ztensor.numpy.load_file(filepath)

    elif format_name == "safetensors":
        if BACKEND == 'torch':
            import safetensors.torch
            loaded_tensors = safetensors.torch.load_file(filepath)
        else:
            import safetensors.numpy
            loaded_tensors = safetensors.numpy.load_file(filepath)

    elif format_name == "pickle":
        import pickle
        with open(filepath, 'rb') as f:
            loaded_tensors = pickle.load(f)

    elif format_name == "hdf5":
        with h5py.File(filepath, "r") as f:
            if BACKEND == 'torch':
                import torch
                for k in f.keys():
                    loaded_tensors[k] = torch.from_numpy(f[k][:])
            else:
                for k in f.keys():
                    loaded_tensors[k] = f[k][:]

    elif format_name == "gguf":
        reader = gguf.GGUFReader(filepath)
        if BACKEND == 'torch':
            import torch
            for tensor in reader.tensors:
                loaded_tensors[tensor.name] = torch.from_numpy(np.array(tensor.data, copy=True))
        else:
            for tensor in reader.tensors:
                loaded_tensors[tensor.name] = np.array(tensor.data, copy=True)

    # FORCE ACTUAL LOAD
    for t in loaded_tensors.values():
        _ = t.sum()
            
    end = time.perf_counter()
    return end - start

# --- SWEEP & PLOT ---

def run_sweep():
    # Sweep configuration
    sizes = [128, 512, 1024, 2048] # MB
    distributions = ["mixed", "large"]
    formats = ["safetensors", "pickle", "hdf5", "ztensor", "ztensor_zstd", "ztensor_fast", "ztensor_max", "gguf"] 
    
    results = []
    
    os.makedirs("bench_out", exist_ok=True)
    os.makedirs("bench_out/plots", exist_ok=True)

    print(f"Starting Sweep...")
    print(f"Sizes: {sizes}")
    print(f"Dists: {distributions}")
    
    for dist in distributions:
        for size_mb in sizes:
            # Generate data once per size/dist combo
            tensors = generate_tensor_dict(size_mb, dist)
            tensor_count = len(tensors)
            
            for fmt in formats:
                filepath = f"bench_out/sweep.{fmt}"
                if fmt == "pickle": filepath = "bench_out/sweep.pt"
                if fmt == "hdf5": filepath = "bench_out/sweep.h5"
                
                # Write
                try:
                    w_speed, size_gb = benchmark_write(fmt, tensors, filepath)
                except Exception as e:
                    print(f"FAIL Write {fmt} {size_mb} {dist}: {e}")
                    continue
                
                # Read
                clear_cache_robust()
                try:
                    r_lat = benchmark_read(fmt, filepath)
                except Exception as e:
                    print(f"FAIL Read {fmt} {size_mb} {dist}: {e}")
                    r_lat = 0
                
                # Cleanup
                if os.path.exists(filepath): os.remove(filepath)
                
                # Record
                print(f"Result: {fmt:<12} | {size_mb}MB ({dist}) | W: {w_speed:.2f}GB/s | R: {r_lat:.3f}s")
                results.append({
                    "Format": fmt,
                    "SizeMB": size_mb,
                    "Distribution": dist,
                    "WriteGBs": w_speed,
                    "ReadSeconds": r_lat,
                    "TensorCount": tensor_count
                })

    # Save CSV
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv("bench_out/sweep_results.csv", index=False)
    print("Sweep complete. Results saved to bench_out/sweep_results.csv")
    plot_results(df)

def plot_results(df):
    """Quick plots after sweep (calls draw_plot)."""
    draw_plot("bench_out/sweep_results.csv")


def draw_plot(csv_path: str = "bench_out/sweep_results.csv", output_dir: str = "bench_out/plots"):
    """
    Creates publication-grade benchmark plots from sweep results CSV.
    
    Args:
        csv_path: Path to the sweep_results.csv file
        output_dir: Directory to save plots
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    # Load data
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Publication Style Setup ---
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
    
    # Color palette - vibrant, colorblind-friendly
    COLORS = {
        'ztensor': '#2563EB',      # Blue (primary)
        'safetensors': '#DC2626',  # Red
        'pickle': '#16A34A',       # Green
        'hdf5': '#9333EA',         # Purple
        'gguf': '#EA580C',         # Orange
        'ztensor_zstd': '#3B82F6', # Lighter Blue
        'ztensor_fast': '#60A5FA', # Even Lighter Blue
        'ztensor_max': '#1E40AF',  # Darker Blue
    }
    
    MARKERS = {
        'ztensor': 'o',
        'safetensors': 's',
        'pickle': '^',
        'hdf5': 'D',
        'gguf': 'v',
    }
    
    distributions = df['Distribution'].unique()
    formats = df['Format'].unique()
    
    # === 1. Write Throughput Plots ===
    for dist in distributions:
        fig, ax = plt.subplots(figsize=(8, 5))
        subset = df[df['Distribution'] == dist]
        
        for fmt in formats:
            data = subset[subset['Format'] == fmt].sort_values('SizeMB')
            ax.plot(
                data['SizeMB'], data['WriteGBs'],
                marker=MARKERS.get(fmt, 'o'),
                markersize=8,
                linewidth=2.5,
                color=COLORS.get(fmt, '#666'),
                label=fmt,
                alpha=0.9
            )
        
        ax.set_xlabel('File Size (MB)')
        ax.set_ylabel('Write Throughput (GB/s)')
        ax.set_title(f'Write Performance — {dist.title()} Tensors')
        ax.legend(loc='best', framealpha=0.9)
        ax.set_ylim(bottom=0)
        
        # Add subtle background
        ax.set_facecolor('#FAFAFA')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/write_throughput_{dist}.png")
        plt.savefig(f"{output_dir}/write_throughput_{dist}.pdf")  # Vector for papers
        plt.close()
    
    # === 2. Read Throughput Plots (convert latency to throughput) ===
    for dist in distributions:
        fig, ax = plt.subplots(figsize=(8, 5))
        subset = df[df['Distribution'] == dist]
        
        for fmt in formats:
            data = subset[subset['Format'] == fmt].sort_values('SizeMB')
            # Convert read latency to throughput: GB/s = (SizeMB / 1024) / ReadSeconds
            read_throughput = (data['SizeMB'] / 1024) / data['ReadSeconds']
            ax.plot(
                data['SizeMB'], read_throughput,
                marker=MARKERS.get(fmt, 'o'),
                markersize=8,
                linewidth=2.5,
                color=COLORS.get(fmt, '#666'),
                label=fmt,
                alpha=0.9
            )
        
        ax.set_xlabel('File Size (MB)')
        ax.set_ylabel('Read Throughput (GB/s)')
        ax.set_title(f'Read Performance — {dist.title()} Tensors')
        ax.legend(loc='best', framealpha=0.9)
        ax.set_ylim(bottom=0)
        ax.set_facecolor('#FAFAFA')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/read_throughput_{dist}.png")
        plt.savefig(f"{output_dir}/read_throughput_{dist}.pdf")
        plt.close()
    
    # === 3. Combined Bar Chart (largest size, all distributions) ===
    max_size = df['SizeMB'].max()
    subset = df[df['SizeMB'] == max_size]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Group data by distribution
    for idx, (metric, title, ylabel) in enumerate([
        ('WriteGBs', 'Write Throughput', 'Throughput (GB/s)'),
        ('ReadThroughput', 'Read Throughput', 'Throughput (GB/s)')
    ]):
        ax = axes[idx]
        
        x = np.arange(len(distributions))
        width = 0.15
        multiplier = 0
        
        for fmt in formats:
            values = []
            for dist in distributions:
                row = subset[(subset['Format'] == fmt) & (subset['Distribution'] == dist)]
                if metric == 'WriteGBs':
                    val = row['WriteGBs'].values[0] if len(row) > 0 else 0
                else:
                    val = (row['SizeMB'].values[0] / 1024) / row['ReadSeconds'].values[0] if len(row) > 0 else 0
                values.append(val)
            
            offset = width * multiplier
            bars = ax.bar(
                x + offset, values, width,
                label=fmt,
                color=COLORS.get(fmt, '#666'),
                alpha=0.9,
                edgecolor='white',
                linewidth=0.5
            )
            multiplier += 1
        
        ax.set_xlabel('Tensor Distribution')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{title} @ {max_size}MB')
        ax.set_xticks(x + width * (len(formats) - 1) / 2)
        ax.set_xticklabels([d.title() for d in distributions])
        ax.legend(loc='upper right', framealpha=0.9)
        ax.set_ylim(bottom=0)
        ax.set_facecolor('#FAFAFA')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_bar_{max_size}mb.png")
    plt.savefig(f"{output_dir}/comparison_bar_{max_size}mb.pdf")
    plt.close()
    
    # === 4. Heatmap for quick overview ===
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    for idx, (metric, title) in enumerate([('WriteGBs', 'Write GB/s'), ('ReadThroughput', 'Read GB/s')]):
        ax = axes[idx]
        
        # Create pivot table
        pivot_data = []
        for fmt in formats:
            row = []
            for size in sorted(df['SizeMB'].unique()):
                # Average across distributions
                subset = df[(df['Format'] == fmt) & (df['SizeMB'] == size)]
                if metric == 'WriteGBs':
                    val = subset['WriteGBs'].mean()
                else:
                    val = ((subset['SizeMB'] / 1024) / subset['ReadSeconds']).mean()
                row.append(val)
            pivot_data.append(row)
        
        pivot_data = np.array(pivot_data)
        
        im = ax.imshow(pivot_data, cmap='RdYlGn', aspect='auto')
        
        ax.set_xticks(np.arange(len(df['SizeMB'].unique())))
        ax.set_yticks(np.arange(len(formats)))
        ax.set_xticklabels([f"{s}MB" for s in sorted(df['SizeMB'].unique())])
        ax.set_yticklabels(formats)
        ax.set_xlabel('File Size')
        ax.set_title(title)
        
        # Add value annotations
        for i in range(len(formats)):
            for j in range(len(df['SizeMB'].unique())):
                text = ax.text(j, i, f'{pivot_data[i, j]:.1f}',
                             ha='center', va='center', color='black', fontsize=9)
        
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/heatmap_overview.png")
    plt.savefig(f"{output_dir}/heatmap_overview.pdf")
    plt.close()
    
    print(f"✓ Publication-grade plots saved to {output_dir}/")

# --- MAIN LOOP ---

def main():
    global BACKEND
    
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?", choices=["run", "sweep", "plot"], default="run", 
                       help="Run mode: single 'run', matrix 'sweep', or 'plot' to regenerate charts")
    parser.add_argument("--size", type=str, choices=["small", "large"], default="small")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--csv", type=str, default="bench_out/sweep_results.csv", help="CSV path for plot mode")
    parser.add_argument("--backend", type=str, choices=["numpy", "torch"], default="numpy",
                       help="Tensor backend to use: 'numpy' (default) or 'torch'")
    args = parser.parse_args()
    
    BACKEND = args.backend
    
    if args.mode == "sweep":
        run_sweep()
        return
    
    if args.mode == "plot":
        draw_plot(args.csv)
        return

    # Default single run behavior
    size_mb = 100 if args.size == "small" else 1024 
    tensors = generate_tensor_dict(size_mb, "mixed")
    
    formats = ["safetensors", "pickle", "hdf5", "gguf", "ztensor", "ztensor_zstd", "ztensor_fast", "ztensor_max"]

    os.makedirs("bench_out", exist_ok=True)
    
    # Headers
    print(f"\n{'Format':<15} | {'Write (GB/s)':<12} | {'Read (s)':<12} | {'Size (MB)':<12}")
    print("-" * 65)

    for fmt in formats:
        filepath = f"bench_out/test.{fmt}"
        if fmt == "pickle": filepath = "bench_out/test.pt"
        if fmt == "hdf5": filepath = "bench_out/test.h5"
        
        # 1. WRITE TEST
        try:
            write_speed, size_gb = benchmark_write(fmt, tensors, filepath)
            file_size_mb = size_gb * 1024
        except Exception as e:
            print(f"{fmt:<15} | Write Failed: {e}")
            continue

        # 2. COLD READ TEST
        clear_cache_robust()
        
        try:
            cold_lat = benchmark_read(fmt, filepath)
        except Exception as e:
            print(f"Read failed for {fmt}: {e}")
            cold_lat = 0
        
        # DISPLAY
        print(f"{fmt:<15} | {write_speed:<12.2f} | {cold_lat:<12.4f} | {file_size_mb:<12.2f}")

        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == "__main__":
    main()