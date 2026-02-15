import os
import sys
import time
import numpy as np
import argparse
import statistics
from typing import Dict, List, Optional, Union

# Note: h5py, gguf, ztensor, torch are imported lazily in benchmark functions

# Global backend setting: 'numpy' or 'torch'
BACKEND = 'numpy'

# Benchmark configuration
WARMUP_RUNS = 1
BENCH_RUNS = 3

# --- UTILITIES ---

def drop_file_cache(filepath: str):
    """
    Drops the page cache for a specific file using posix_fadvise(DONTNEED).
    Falls back to sync + /proc/sys/vm/drop_caches if root, or a large read buster.
    """
    # First, sync to flush dirty pages
    os.sync()

    # Method 1: posix_fadvise (Linux, targeted and effective)
    if hasattr(os, 'posix_fadvise'):
        try:
            fd = os.open(filepath, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
            return
        except OSError:
            pass

    # Method 2: drop_caches if root (Linux)
    if sys.platform == "linux" and os.geteuid() == 0:
        os.system('echo 3 > /proc/sys/vm/drop_caches')
        return

    # Method 3: Fallback - read a large buster file to evict pages
    try:
        buster = "cache_buster.tmp"
        buster_size = 2 * 1024 * 1024 * 1024  # 2GB
        if not os.path.exists(buster) or os.path.getsize(buster) < buster_size:
            with open(buster, "wb") as f:
                for _ in range(2048):
                    f.write(os.urandom(1024 * 1024))
        with open(buster, "rb") as f:
            while f.read(4 * 1024 * 1024):
                pass
    except Exception:
        pass


def generate_tensor_dict(
    total_size_mb: int,
    distribution: str = "mixed",
    data_style: str = "random",
) -> Dict[str, Union[np.ndarray, 'torch.Tensor']]:
    """
    Generates tensors summing to total_size_mb.
    distribution:
      - 'mixed': Realistic mix (some large weights, some small biases/metadata)
      - 'small': Many small tensors (1KB - 100KB). Stresses metadata/parsing.
      - 'large': Few large tensors (10MB - 100MB). Stresses raw BW.
    data_style:
      - 'random': Uniform random float32 (incompressible).
      - 'structured': Normal(0, 0.02) with block-sparsity (compressible, mimics real weights).
      - 'mixed_dtype': Realistic dtype mix (70% fp32, 20% fp16, 10% int8).

    Uses global BACKEND setting to determine tensor type ('numpy' or 'torch').
    """
    style_tag = f", {data_style}" if data_style != "random" else ""
    print(f"  [{distribution.upper()}] Generating {total_size_mb}MB of synthetic data "
          f"(backend={BACKEND}{style_tag})...")
    tensors = {}
    remaining_bytes = total_size_mb * 1024 * 1024
    i = 0

    while remaining_bytes > 0:
        # Determine dtype and element size for this tensor
        if data_style == "mixed_dtype":
            if i % 10 < 7:
                dtype = np.float32
                elem_size = 4
            elif i % 10 < 9:
                dtype = np.float16
                elem_size = 2
            else:
                dtype = np.int8
                elem_size = 1
        else:
            dtype = np.float32
            elem_size = 4

        # Determine shape based on distribution
        if distribution == "mixed":
            if remaining_bytes > 100 * 1024 * 1024:
                shape = (5000, 5000)
            elif remaining_bytes > 10 * 1024 * 1024:
                shape = (1000, 2500)
            else:
                elems = remaining_bytes // elem_size
                shape = (elems,)

        elif distribution == "large":
            target_bytes = 50 * 1024 * 1024
            if remaining_bytes < target_bytes:
                target_bytes = remaining_bytes
            elems = target_bytes // elem_size
            shape = (elems,)

        elif distribution == "small":
            target_bytes = 10 * 1024
            if remaining_bytes < target_bytes:
                target_bytes = remaining_bytes
            elems = target_bytes // elem_size
            shape = (elems,)

        elif distribution == "llama-1b":
            # Llama 3.2 1B architecture: hidden=2048, layers=16, kv_heads=8,
            # intermediate=8192, vocab=128256.
            # Total size: ~2.5 GB in fp16. Ignores total_size_mb.
            llama_tensors = _generate_llama_1b(data_style)
            return llama_tensors

        if np.prod(shape) == 0:
            break

        # Generate data based on data_style
        if data_style == "random":
            t_np = np.random.randn(*shape).astype(np.float32)
        elif data_style == "structured":
            t_np = np.random.normal(0.0, 0.02, size=shape).astype(np.float32)
            # Add block-sparsity: zero out ~20% of rows for realistic structure
            if len(shape) == 2 and shape[0] > 10:
                zero_rows = np.random.choice(shape[0], size=shape[0] // 5, replace=False)
                t_np[zero_rows] = 0.0
        elif data_style == "mixed_dtype":
            if dtype == np.int8:
                t_np = np.random.randint(-128, 127, size=shape, dtype=np.int8)
            else:
                t_np = np.random.normal(0.0, 0.02, size=shape).astype(dtype)

        if BACKEND == 'torch':
            import torch
            if dtype == np.int8:
                t = torch.from_numpy(t_np.copy())
            else:
                t = torch.from_numpy(t_np)
            remaining_bytes -= t.numel() * t.element_size()
        else:
            t = t_np
            remaining_bytes -= t.nbytes

        tensors[f"layer_{i}.weight"] = t
        i += 1
    return tensors


def _generate_llama_1b(data_style: str = "random") -> dict:
    """
    Generate tensors matching Llama 3.2 1B architecture shapes.
    hidden=2048, layers=16, heads=32, kv_heads=8, intermediate=8192, vocab=128256.
    All float16 (matching real HF checkpoints). Random data.
    ~1.24B params, ~2.5 GB in fp16.
    """
    H = 2048        # hidden_size
    KV = 512        # num_kv_heads * head_dim = 8 * 64
    I = 8192        # intermediate_size
    V = 128256      # vocab_size
    N = 16          # num_hidden_layers

    dtype = np.float16
    tensors = {}

    def _make(shape):
        if data_style == "structured":
            t = np.random.normal(0.0, 0.02, size=shape).astype(dtype)
            if len(shape) == 2 and shape[0] > 10:
                zero_rows = np.random.choice(shape[0], size=shape[0] // 5, replace=False)
                t[zero_rows] = 0.0
            return t
        return np.random.randn(*shape).astype(dtype)

    # Embedding (tied with lm_head in Llama 3.2 1B, but we include both)
    tensors["model.embed_tokens.weight"] = _make((V, H))
    tensors["model.norm.weight"] = _make((H,))
    tensors["lm_head.weight"] = _make((V, H))

    # Per-layer weights (16 layers)
    for layer in range(N):
        p = f"model.layers.{layer}"
        # Self-attention (GQA: q_proj full size, k/v_proj reduced for 8 KV heads)
        tensors[f"{p}.self_attn.q_proj.weight"] = _make((H, H))
        tensors[f"{p}.self_attn.k_proj.weight"] = _make((KV, H))
        tensors[f"{p}.self_attn.v_proj.weight"] = _make((KV, H))
        tensors[f"{p}.self_attn.o_proj.weight"] = _make((H, H))
        # MLP
        tensors[f"{p}.mlp.gate_proj.weight"] = _make((I, H))
        tensors[f"{p}.mlp.up_proj.weight"] = _make((I, H))
        tensors[f"{p}.mlp.down_proj.weight"] = _make((H, I))
        # Layer norms
        tensors[f"{p}.input_layernorm.weight"] = _make((H,))
        tensors[f"{p}.post_attention_layernorm.weight"] = _make((H,))

    total_bytes = sum(t.nbytes for t in tensors.values())
    total_params = sum(t.size for t in tensors.values())
    print(f"  [LLAMA-1B] Generated {len(tensors)} tensors, "
          f"{total_params/1e9:.2f}B params, {total_bytes/1e9:.2f} GB ({data_style})")

    return tensors

# --- FORMAT HELPERS ---

# Formats that write + read with their own native library
NATIVE_FORMATS = ["ztensor", "ztensor_zstd", "safetensors", "pickle", "hdf5", "gguf", "npz", "onnx"]

# Zstd compression levels to benchmark
ZSTD_FORMATS = ["ztensor_zstd3", "ztensor_zstd13", "ztensor_zstd22"]

# Zero-copy variant (separate from NATIVE_FORMATS so default benchmarks are unchanged)
ZEROCOPY_FORMATS = ["ztensor_zerocopy"]

# Cross-format read: write with native lib, read via ztensor
CROSS_READ_FORMATS = ["zt_read_st", "zt_read_pt", "zt_read_gguf", "zt_read_npz", "zt_read_onnx", "zt_read_h5"]

ALL_FORMATS = NATIVE_FORMATS + ZEROCOPY_FORMATS + CROSS_READ_FORMATS

def filepath_for_format(fmt, base_dir="bench_out", prefix="test"):
    """Return the appropriate filepath for a format."""
    if fmt == "pickle":
        return f"{base_dir}/{prefix}.pickle"
    elif fmt == "hdf5":
        return f"{base_dir}/{prefix}.h5"
    elif fmt == "zt_read_st":
        return f"{base_dir}/{prefix}.safetensors"
    elif fmt == "zt_read_pt":
        return f"{base_dir}/{prefix}_pt.pt"
    elif fmt == "zt_read_gguf":
        return f"{base_dir}/{prefix}.gguf"
    elif fmt == "zt_read_npz":
        return f"{base_dir}/{prefix}.npz"
    elif fmt == "zt_read_onnx":
        return f"{base_dir}/{prefix}.onnx"
    elif fmt == "zt_read_h5":
        return f"{base_dir}/{prefix}.h5"
    elif fmt == "ztensor_zstd":
        return f"{base_dir}/{prefix}_zstd.zt"
    elif fmt.startswith("ztensor_zstd"):
        level = fmt.replace("ztensor_zstd", "")
        return f"{base_dir}/{prefix}_zstd{level}.zt"
    elif fmt == "ztensor_zerocopy":
        return f"{base_dir}/{prefix}_zerocopy.zt"
    elif fmt.startswith("ztensor"):
        return f"{base_dir}/{prefix}.zt"
    else:
        return f"{base_dir}/{prefix}.{fmt}"


# --- BENCHMARK FUNCTIONS ---

def benchmark_write(format_name, tensors, filepath):
    """Write tensors in the given format. Returns (throughput_GBs, file_size_GB)."""

    start = time.perf_counter()

    if format_name in ("ztensor", "ztensor_zerocopy"):
        if BACKEND == 'torch':
            import ztensor.torch
            ztensor.torch.save_file(tensors, filepath)
        else:
            import ztensor.numpy
            ztensor.numpy.save_file(tensors, filepath)

    elif format_name == "ztensor_zstd" or format_name.startswith("ztensor_zstd"):
        level = 3
        if format_name != "ztensor_zstd":
            level = int(format_name.replace("ztensor_zstd", ""))
        if BACKEND == 'torch':
            import ztensor.torch
            ztensor.torch.save_file(tensors, filepath, compression=level)
        else:
            import ztensor.numpy
            ztensor.numpy.save_file(tensors, filepath, compression=level)

    elif format_name in ("safetensors", "zt_read_st"):
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

    elif format_name == "zt_read_pt":
        import torch
        if BACKEND == 'torch':
            state_dict = dict(tensors)
        else:
            state_dict = {k: torch.from_numpy(v) for k, v in tensors.items()}
        torch.save(state_dict, filepath)

    elif format_name == "zt_read_gguf":
        import gguf
        gw = gguf.GGUFWriter(filepath, "benchmark_model")
        for name, t in tensors.items():
            data = t.numpy() if BACKEND == 'torch' else t
            gw.add_tensor(name, data)
        gw.write_header_to_file()
        gw.write_kv_data_to_file()
        gw.write_tensors_to_file()
        gw.close()

    elif format_name in ("hdf5", "zt_read_h5"):
        import h5py
        with h5py.File(filepath, "w") as f:
            for name, t in tensors.items():
                data = t.numpy() if BACKEND == 'torch' else t
                f.create_dataset(name, data=data)

    elif format_name == "gguf":
        import gguf
        gw = gguf.GGUFWriter(filepath, "benchmark_model")
        for name, t in tensors.items():
            data = t.numpy() if BACKEND == 'torch' else t
            gw.add_tensor(name, data)
        gw.write_header_to_file()
        gw.write_kv_data_to_file()
        gw.write_tensors_to_file()
        gw.close()

    elif format_name in ("npz", "zt_read_npz"):
        np_tensors = {k: (v.numpy() if BACKEND == 'torch' else v) for k, v in tensors.items()}
        np.savez(filepath, **np_tensors)

    elif format_name in ("onnx", "zt_read_onnx"):
        import onnx
        from onnx import numpy_helper, helper
        initializers = []
        for name, t in tensors.items():
            data = t.numpy() if BACKEND == 'torch' else t
            initializers.append(numpy_helper.from_array(data, name=name))
        graph = helper.make_graph([], 'benchmark', [], [], initializer=initializers)
        model = helper.make_model(graph)
        onnx.save(model, filepath)

    end = time.perf_counter()
    size_gb = os.path.getsize(filepath) / (1024**3)
    duration = end - start
    return size_gb / duration if duration > 0 else 0, size_gb


def benchmark_read(format_name, filepath):
    """Read tensors in the given format. Returns latency in seconds."""

    start = time.perf_counter()
    loaded_tensors = {}

    if format_name == "ztensor_zerocopy":
        # Zero-copy: mmap-backed, no memcpy
        if BACKEND == 'torch':
            import ztensor.torch
            loaded_tensors = ztensor.torch.load_file(filepath, copy=False)
        else:
            import ztensor.numpy
            loaded_tensors = ztensor.numpy.load_file(filepath, copy=False)

    elif format_name.startswith("ztensor") or format_name.startswith("zt_read_"):
        if BACKEND == 'torch':
            import ztensor.torch
            loaded_tensors = ztensor.torch.load_file(filepath)
        else:
            import ztensor.numpy
            loaded_tensors = ztensor.numpy.load_file(filepath)

    elif format_name == "safetensors":
        if BACKEND == 'torch':
            from safetensors.torch import load_file as st_load
            loaded_tensors = st_load(filepath)
        else:
            from safetensors.numpy import load_file as st_load
            loaded_tensors = st_load(filepath)

    elif format_name == "pickle":
        import pickle
        with open(filepath, 'rb') as f:
            loaded_tensors = pickle.load(f)

    elif format_name == "hdf5":
        import h5py
        with h5py.File(filepath, "r") as f:
            if BACKEND == 'torch':
                import torch
                for k in f.keys():
                    loaded_tensors[k] = torch.from_numpy(f[k][:])
            else:
                for k in f.keys():
                    loaded_tensors[k] = f[k][:]

    elif format_name == "gguf":
        # GGUFReader uses np.memmap internally; tensor.data is already mmap-backed
        import gguf
        reader = gguf.GGUFReader(filepath)
        if BACKEND == 'torch':
            import torch
            for tensor in reader.tensors:
                # torch.from_numpy needs writable array; mmap is read-only
                loaded_tensors[tensor.name] = torch.from_numpy(tensor.data.copy())
        else:
            for tensor in reader.tensors:
                loaded_tensors[tensor.name] = tensor.data  # zero-copy mmap view

    elif format_name == "npz":
        data = np.load(filepath)
        if BACKEND == 'torch':
            import torch
            for k in data.files:
                loaded_tensors[k] = torch.from_numpy(data[k])
        else:
            for k in data.files:
                loaded_tensors[k] = data[k]

    elif format_name == "onnx":
        import onnx
        from onnx import numpy_helper
        model = onnx.load(filepath)
        if BACKEND == 'torch':
            import torch
            for init in model.graph.initializer:
                loaded_tensors[init.name] = torch.from_numpy(numpy_helper.to_array(init).copy())
        else:
            for init in model.graph.initializer:
                loaded_tensors[init.name] = numpy_helper.to_array(init)

    # Force actual data access (faults mmap pages for zero-copy formats).
    # For few large tensors, use view-as-uint8 to avoid dtype-dependent compute
    # overhead (e.g., float16 sum is 12x slower due to numpy float32 promotion).
    # For many small tensors, use plain sum() to avoid per-tensor view overhead.
    if len(loaded_tensors) > 1000:
        for t in loaded_tensors.values():
            _ = t.sum()
    else:
        for t in loaded_tensors.values():
            if hasattr(t, 'numpy'):
                _ = t.view(torch.uint8).sum()
            else:
                _ = t.view(np.uint8).sum()

    end = time.perf_counter()
    return end - start


def benchmark_read_selective(format_name, filepath, fraction=0.1):
    """Read only a fraction of tensors from a file. Returns latency in seconds."""

    start = time.perf_counter()
    loaded_tensors = {}

    if format_name.startswith("ztensor") or format_name.startswith("zt_"):
        import ztensor
        reader = ztensor.Reader(filepath)
        all_keys = reader.keys()
        step = max(1, int(1.0 / fraction))
        selected = all_keys[::step]
        loaded_tensors = dict(reader.read_tensors(selected))

    elif format_name == "safetensors":
        from safetensors import safe_open
        framework = "pt" if BACKEND == 'torch' else "numpy"
        with safe_open(filepath, framework=framework) as f:
            all_keys = list(f.keys())
            step = max(1, int(1.0 / fraction))
            selected = all_keys[::step]
            for k in selected:
                loaded_tensors[k] = f.get_tensor(k)

    elif format_name == "pickle":
        # Pickle cannot do selective loading -- must load everything
        import pickle
        with open(filepath, 'rb') as f:
            all_tensors = pickle.load(f)
        all_keys = list(all_tensors.keys())
        step = max(1, int(1.0 / fraction))
        selected = all_keys[::step]
        loaded_tensors = {k: all_tensors[k] for k in selected}

    elif format_name == "hdf5":
        import h5py
        with h5py.File(filepath, "r") as f:
            all_keys = list(f.keys())
            step = max(1, int(1.0 / fraction))
            selected = all_keys[::step]
            if BACKEND == 'torch':
                import torch
                for k in selected:
                    loaded_tensors[k] = torch.from_numpy(f[k][:])
            else:
                for k in selected:
                    loaded_tensors[k] = f[k][:]

    elif format_name == "gguf":
        import gguf
        reader = gguf.GGUFReader(filepath)
        all_tensors = list(reader.tensors)
        step = max(1, int(1.0 / fraction))
        selected = all_tensors[::step]
        if BACKEND == 'torch':
            import torch
            for tensor in selected:
                loaded_tensors[tensor.name] = torch.from_numpy(tensor.data.copy())
        else:
            for tensor in selected:
                loaded_tensors[tensor.name] = tensor.data  # mmap-backed view

    # Force data access
    for t in loaded_tensors.values():
        _ = t.sum()

    end = time.perf_counter()
    return end - start


def run_single_benchmark(
    fmt, tensors, filepath,
    runs=BENCH_RUNS, warmup=WARMUP_RUNS,
    scenario="full_load",
):
    """
    Run write + read benchmark for a single format with warmup and multiple runs.
    scenario: "full_load" (default), "zero_copy", "selective"
    Returns dict with median write throughput, median read latency, and file size.
    """
    write_speeds = []
    read_latencies = []
    file_size_gb = 0

    # For zero_copy scenario, use ztensor_zerocopy read path
    read_fmt = "ztensor_zerocopy" if (scenario == "zero_copy" and fmt == "ztensor") else fmt

    for i in range(warmup + runs):
        is_warmup = i < warmup

        # Write
        try:
            w_speed, size_gb = benchmark_write(fmt, tensors, filepath)
            file_size_gb = size_gb
        except Exception as e:
            print(f"  FAIL Write {fmt}: {e}")
            return None

        if not is_warmup:
            write_speeds.append(w_speed)

        # Read (with cache dropping)
        drop_file_cache(filepath)

        try:
            if scenario == "selective":
                r_lat = benchmark_read_selective(fmt, filepath, fraction=0.1)
            elif scenario == "zero_copy":
                r_lat = benchmark_read(read_fmt, filepath)
            else:
                r_lat = benchmark_read(fmt, filepath)
        except Exception as e:
            print(f"  FAIL Read {fmt} ({scenario}): {e}")
            r_lat = float('inf')

        if not is_warmup:
            read_latencies.append(r_lat)

        # Clean up between runs
        if os.path.exists(filepath):
            os.remove(filepath)

    return {
        "write_median": statistics.median(write_speeds),
        "read_median": statistics.median(read_latencies),
        "write_all": write_speeds,
        "read_all": read_latencies,
        "file_size_gb": file_size_gb,
    }


# --- SWEEP & PLOT ---

def run_sweep():
    import pandas as pd

    sizes = [128, 512, 1024, 2048]  # MB
    distributions = ["mixed", "small", "large"]
    formats = NATIVE_FORMATS  # Only native formats in main sweep

    results = []

    os.makedirs("bench_out", exist_ok=True)
    os.makedirs("bench_out/plots", exist_ok=True)

    # Define sweep matrix: (data_style, scenario, distributions, sizes, formats)
    sweep_configs = [
        # Original sweep: random data, full load, all distributions and sizes
        {
            "data_style": "random",
            "scenario": "full_load",
            "distributions": distributions,
            "sizes": sizes,
            "formats": formats,
        },
        # Structured data: shows realistic compression ratios
        {
            "data_style": "structured",
            "scenario": "full_load",
            "distributions": ["mixed"],
            "sizes": [512, 1024, 2048],
            "formats": formats,
        },
        # Zero-copy: ztensor's killer feature for inference
        {
            "data_style": "random",
            "scenario": "zero_copy",
            "distributions": ["mixed"],
            "sizes": [512, 1024, 2048],
            "formats": ["ztensor"],  # only ztensor supports zero-copy
        },
        # Selective loading: load 10% of tensors
        {
            "data_style": "random",
            "scenario": "selective",
            "distributions": ["mixed"],
            "sizes": [512, 1024, 2048],
            "formats": formats,
        },
        # Cross-format: ztensor reading other formats
        {
            "data_style": "random",
            "scenario": "full_load",
            "distributions": ["mixed"],
            "sizes": sizes,
            "formats": CROSS_READ_FORMATS,
        },
        # Realistic workload: Llama 1B shapes (fixed size, ignores sizes param)
        {
            "data_style": "random",
            "scenario": "full_load",
            "distributions": ["llama-1b"],
            "sizes": [0],  # ignored, llama-1b generates its own fixed size
            "formats": formats,
        },
        # Zstd compression level sweep
        {
            "data_style": "random",
            "scenario": "full_load",
            "distributions": ["mixed"],
            "sizes": [512],
            "formats": ["ztensor"] + ZSTD_FORMATS,
        },
        {
            "data_style": "structured",
            "scenario": "full_load",
            "distributions": ["mixed"],
            "sizes": [512],
            "formats": ["ztensor"] + ZSTD_FORMATS,
        },
    ]

    print(f"=== Benchmark Sweep ===")
    print(f"Runs per data point: {BENCH_RUNS} (+ {WARMUP_RUNS} warmup)")
    print()

    for config in sweep_configs:
        data_style = config["data_style"]
        scenario = config["scenario"]

        print(f"--- {scenario} / {data_style} ---")
        print(f"  Sizes: {config['sizes']}, Dists: {config['distributions']}, "
              f"Formats: {config['formats']}")
        print()

        for dist in config["distributions"]:
            for size_mb in config["sizes"]:
                tensors = generate_tensor_dict(size_mb, dist, data_style)
                tensor_count = len(tensors)

                for fmt in config["formats"]:
                    filepath = filepath_for_format(fmt, "bench_out", "sweep")

                    result = run_single_benchmark(
                        fmt, tensors, filepath, scenario=scenario,
                    )

                    if result is None:
                        continue

                    # Clean up
                    if os.path.exists(filepath):
                        os.remove(filepath)

                    w_speed = result["write_median"]
                    r_lat = result["read_median"]
                    size_gb = result["file_size_gb"]

                    print(f"  {fmt:<16} | {size_mb:>4}MB ({dist:<5}) | "
                          f"W: {w_speed:.2f} GB/s | R: {r_lat:.4f}s | "
                          f"File: {size_gb*1024:.1f}MB | Tensors: {tensor_count} "
                          f"| {scenario}/{data_style}")

                    results.append({
                        "Format": fmt,
                        "SizeMB": size_mb,
                        "Distribution": dist,
                        "DataStyle": data_style,
                        "Scenario": scenario,
                        "WriteGBs": w_speed,
                        "ReadSeconds": r_lat,
                        "FileSizeGB": size_gb,
                        "TensorCount": tensor_count,
                    })

                print()

    # Save CSV
    df = pd.DataFrame(results)
    df.to_csv("bench_out/sweep_results.csv", index=False)
    print("Sweep complete. Results saved to bench_out/sweep_results.csv")
    draw_plot("bench_out/sweep_results.csv")


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


# --- MAIN ---

def main():
    global BACKEND, WARMUP_RUNS, BENCH_RUNS

    parser = argparse.ArgumentParser(
        description="Benchmark tensor serialization formats (ztensor, safetensors, pytorch, pickle, hdf5, gguf)")
    parser.add_argument("mode", nargs="?", choices=["run", "sweep", "plot"], default="run",
                        help="Run mode: single 'run', matrix 'sweep', or 'plot' to regenerate charts")
    parser.add_argument("--size", type=int, default=512, help="Data size in MB for single run (default: 512)")
    parser.add_argument("--runs", type=int, default=3, help="Number of timed runs per data point (default: 3)")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup runs (default: 1)")
    parser.add_argument("--csv", type=str, default="bench_out/sweep_results.csv", help="CSV path for plot mode")
    parser.add_argument("--backend", type=str, choices=["numpy", "torch"], default="numpy",
                        help="Tensor backend: 'numpy' (default) or 'torch'")
    parser.add_argument("--formats", type=str, nargs="+", default=None,
                        help="Specific formats to benchmark (default: all native)")
    parser.add_argument("--data-style", type=str, choices=["random", "structured", "mixed_dtype"],
                        default="random", help="Data generation style (default: random)")
    parser.add_argument("--scenario", type=str, choices=["full_load", "zero_copy", "selective"],
                        default="full_load", help="Benchmark scenario (default: full_load)")
    parser.add_argument("--dist", type=str, choices=["mixed", "small", "large", "llama-1b"],
                        default="mixed", help="Tensor distribution (default: mixed)")
    args = parser.parse_args()

    BACKEND = args.backend
    BENCH_RUNS = args.runs
    WARMUP_RUNS = args.warmup

    if args.mode == "sweep":
        run_sweep()
        return

    if args.mode == "plot":
        draw_plot(args.csv)
        return

    # Default: single run mode
    size_mb = args.size
    data_style = args.data_style
    scenario = args.scenario
    distribution = args.dist
    tensors = generate_tensor_dict(size_mb, distribution, data_style)
    formats = args.formats or NATIVE_FORMATS

    os.makedirs("bench_out", exist_ok=True)

    scenario_label = f", {scenario}" if scenario != "full_load" else ""
    style_label = f", {data_style}" if data_style != "random" else ""
    size_label = f"{size_mb}MB " if distribution != "llama-1b" else ""
    print(f"\n=== Single Run: {size_label}{distribution}{style_label}{scenario_label}, "
          f"{BENCH_RUNS} runs + {WARMUP_RUNS} warmup ===\n")
    print(f"{'Format':<20} | {'Write (GB/s)':>12} | {'Read (s)':>12} | {'Read (GB/s)':>12} | {'File (MB)':>10}")
    print("-" * 80)

    for fmt in formats:
        filepath = filepath_for_format(fmt)

        result = run_single_benchmark(fmt, tensors, filepath, scenario=scenario)

        if result is None:
            continue

        w_speed = result["write_median"]
        r_lat = result["read_median"]
        file_mb = result["file_size_gb"] * 1024
        r_tp = (file_mb / 1024) / r_lat if r_lat > 0 else 0

        print(f"{fmt:<20} | {w_speed:>12.2f} | {r_lat:>12.4f} | {r_tp:>12.2f} | {file_mb:>10.1f}")

        if os.path.exists(filepath):
            os.remove(filepath)

    print()


if __name__ == "__main__":
    main()
