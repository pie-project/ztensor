import os
import sys
import time
import numpy as np
import h5py
import gguf
import argparse
import threading
from typing import Dict
import ztensor

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

def generate_tensor_dict(total_size_mb: int, distribution: str = "mixed") -> Dict[str, np.ndarray]:
    """
    Generates tensors summing to total_size_mb.
    distribution:
      - 'mixed': Realistic mix (some large weights, some small biases/metadata)
      - 'small': Many small tensors (1KB - 100KB). Stresses metadata/parsing.
      - 'large': Few large tensors (10MB - 100MB). Stresses raw BW.
    """
    print(f"[{distribution.upper()}] Generating {total_size_mb}MB of synthetic data...")
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
            
        t = np.random.randn(*shape).astype(np.float32)
        tensors[f"layer_{i}.weight"] = t
        remaining_bytes -= t.nbytes
        i += 1
    return tensors

# --- BENCHMARK FUNCTIONS ---

def benchmark_write(format_name, tensors, filepath):
    start = time.perf_counter()
    
    if format_name == "ztensor" or format_name == "ztensor_zstd":
        compress = (format_name == "ztensor_zstd")
        with ztensor.Writer(filepath) as w:
            for name, t in tensors.items():
                w.add_tensor(name, t, compress=compress)

    elif format_name == "safetensors":
        import safetensors.numpy
        safetensors.numpy.save_file(tensors, filepath)
        
    elif format_name == "pickle":
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(tensors, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    elif format_name == "hdf5":
        with h5py.File(filepath, "w") as f:
            for name, t in tensors.items():
                f.create_dataset(name, data=t)
                
    elif format_name == "gguf":
        gw = gguf.GGUFWriter(filepath, "benchmark_model")
        for name, t in tensors.items():
            gw.add_tensor(name, t)
        gw.write_header_to_file()
        gw.write_kv_data_to_file()
        gw.write_tensors_to_file()
        gw.close()
    
    end = time.perf_counter()
    size_gb = os.path.getsize(filepath) / (1024**3)
    duration = end - start
    return size_gb / duration, size_gb 

def benchmark_read(format_name, filepath):
    start = time.perf_counter()
    loaded_tensors = {}

    if format_name == "ztensor" or format_name == "ztensor_zstd":
        with ztensor.Reader(filepath) as r:
            for name in r.tensor_names:
                loaded_tensors[name] = r.read_tensor(name, to='numpy')

    elif format_name == "safetensors":
        import safetensors.numpy
        loaded_tensors = safetensors.numpy.load_file(filepath)

    elif format_name == "pickle":
        import pickle
        with open(filepath, 'rb') as f:
            loaded_tensors = pickle.load(f)

    elif format_name == "hdf5":
        with h5py.File(filepath, "r") as f:
            for k in f.keys():
                loaded_tensors[k] = f[k][:]

    elif format_name == "gguf":
        reader = gguf.GGUFReader(filepath)
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
    distributions = ["mixed", "small", "large"]
    formats = ["safetensors", "pickle", "hdf5", "ztensor", "gguf"] 
    
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
    df = pd.DataFrame(results)
    df.to_csv("bench_out/sweep_results.csv", index=False)
    print("Sweep complete. Results saved to bench_out/sweep_results.csv")
    plot_results(df)

def plot_results(df):
    distributions = df['Distribution'].unique()
    
    # 1. Throughput vs Size (per distribution)
    for dist in distributions:
        subset = df[df['Distribution'] == dist]
        plt.figure(figsize=(10, 6))
        for fmt in subset['Format'].unique():
            data = subset[subset['Format'] == fmt]
            plt.plot(data['SizeMB'], data['WriteGBs'], marker='o', label=fmt)
        
        plt.title(f"Write Throughput vs File Size ({dist.title()})")
        plt.xlabel("Total Size (MB)")
        plt.ylabel("Throughput (GB/s)")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"bench_out/plots/write_throughput_{dist}.png")
        plt.close()
        
    # 2. Read Latency vs Size (per distribution)
    for dist in distributions:
        subset = df[df['Distribution'] == dist]
        plt.figure(figsize=(10, 6))
        for fmt in subset['Format'].unique():
            data = subset[subset['Format'] == fmt]
            plt.plot(data['SizeMB'], data['ReadSeconds'], marker='o', label=fmt)
            
        plt.title(f"Read Latency vs File Size ({dist.title()})")
        plt.xlabel("Total Size (MB)")
        plt.ylabel("Latency (s)")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"bench_out/plots/read_latency_{dist}.png")
        plt.close()

    print("Plots generated in bench_out/plots/")

# --- MAIN LOOP ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?", choices=["run", "sweep"], default="run", help="Run mode: single 'run' or matrix 'sweep'")
    parser.add_argument("--size", type=str, choices=["small", "large"], default="small")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    
    if args.mode == "sweep":
        run_sweep()
        return

    # Default single run behavior
    size_mb = 100 if args.size == "small" else 1024 
    tensors = generate_tensor_dict(size_mb, "mixed")
    
    formats = ["safetensors", "pickle", "hdf5", "gguf", "ztensor", "ztensor_zstd"]

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