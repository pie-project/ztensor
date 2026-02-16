import gc
import os
import sys
import time
import statistics
import numpy as np

import config


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


def benchmark_write(format_name, tensors, filepath):
    """Write tensors in the given format. Returns (throughput_GBs, file_size_GB, data_size_GB)."""
    data_size_gb = sum(
        (t.numpy().nbytes if hasattr(t, 'numpy') else t.nbytes) for t in tensors.values()
    ) / (1024**3)

    start = time.perf_counter()

    if format_name in ("ztensor", "ztensor_zerocopy"):
        if config.BACKEND == 'torch':
            import ztensor.torch
            ztensor.torch.save_file(tensors, filepath)
        else:
            import ztensor.numpy
            ztensor.numpy.save_file(tensors, filepath)

    elif format_name == "ztensor_zstd" or format_name.startswith("ztensor_zstd"):
        level = 3
        if format_name != "ztensor_zstd":
            level = int(format_name.replace("ztensor_zstd", ""))
        if config.BACKEND == 'torch':
            import ztensor.torch
            ztensor.torch.save_file(tensors, filepath, compression=level)
        else:
            import ztensor.numpy
            ztensor.numpy.save_file(tensors, filepath, compression=level)

    elif format_name in ("safetensors", "zt_read_st"):
        if config.BACKEND == 'torch':
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
        if config.BACKEND == 'torch':
            state_dict = dict(tensors)
        else:
            state_dict = {k: torch.from_numpy(v) for k, v in tensors.items()}
        torch.save(state_dict, filepath)

    elif format_name == "zt_read_gguf":
        import gguf
        gw = gguf.GGUFWriter(filepath, "benchmark_model")
        for name, t in tensors.items():
            data = t.numpy() if config.BACKEND == 'torch' else t
            gw.add_tensor(name, data)
        gw.write_header_to_file()
        gw.write_kv_data_to_file()
        gw.write_tensors_to_file()
        gw.close()

    elif format_name in ("hdf5", "zt_read_h5"):
        import h5py
        with h5py.File(filepath, "w") as f:
            for name, t in tensors.items():
                data = t.numpy() if config.BACKEND == 'torch' else t
                f.create_dataset(name, data=data)

    elif format_name == "gguf":
        import gguf
        gw = gguf.GGUFWriter(filepath, "benchmark_model")
        for name, t in tensors.items():
            data = t.numpy() if config.BACKEND == 'torch' else t
            gw.add_tensor(name, data)
        gw.write_header_to_file()
        gw.write_kv_data_to_file()
        gw.write_tensors_to_file()
        gw.close()

    elif format_name in ("npz", "zt_read_npz"):
        np_tensors = {k: (v.numpy() if config.BACKEND == 'torch' else v) for k, v in tensors.items()}
        np.savez(filepath, **np_tensors)

    elif format_name in ("onnx", "zt_read_onnx"):
        import onnx
        from onnx import numpy_helper, helper
        initializers = []
        for name, t in tensors.items():
            data = t.numpy() if config.BACKEND == 'torch' else t
            initializers.append(numpy_helper.from_array(data, name=name))
        graph = helper.make_graph([], 'benchmark', [], [], initializer=initializers)
        model = helper.make_model(graph)
        onnx.save(model, filepath)

    end = time.perf_counter()
    file_size_gb = os.path.getsize(filepath) / (1024**3)
    duration = end - start
    throughput = data_size_gb / duration if duration > 0 else 0
    return throughput, file_size_gb, data_size_gb


def benchmark_read(format_name, filepath, fastest=False):
    """Read tensors in the given format. Returns latency in seconds.

    If fastest=True, each format uses its most performant read path
    (e.g., mmap/zero-copy when available).
    """

    start = time.perf_counter()
    loaded_tensors = {}

    if format_name == "ztensor_zerocopy":
        # Zero-copy: mmap-backed, no memcpy
        if config.BACKEND == 'torch':
            import ztensor.torch
            loaded_tensors = ztensor.torch.load_file(filepath, copy=False)
        else:
            import ztensor.numpy
            loaded_tensors = ztensor.numpy.load_file(filepath, copy=False)

    elif format_name.startswith("ztensor") or format_name.startswith("zt_read_"):
        copy = not fastest
        if config.BACKEND == 'torch':
            import ztensor.torch
            loaded_tensors = ztensor.torch.load_file(filepath, copy=copy)
        else:
            import ztensor.numpy
            loaded_tensors = ztensor.numpy.load_file(filepath, copy=copy)

    elif format_name == "safetensors":
        if fastest:
            # safe_open uses mmap internally; get_tensor returns views via np.frombuffer
            from safetensors import safe_open
            framework = "pt" if config.BACKEND == 'torch' else "numpy"
            with safe_open(filepath, framework=framework) as f:
                for k in f.keys():
                    loaded_tensors[k] = f.get_tensor(k)
        else:
            if config.BACKEND == 'torch':
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
            if config.BACKEND == 'torch':
                import torch
                for k in f.keys():
                    loaded_tensors[k] = torch.from_numpy(f[k][:])
            else:
                for k in f.keys():
                    loaded_tensors[k] = f[k][:]

    elif format_name == "gguf":
        import gguf
        reader = gguf.GGUFReader(filepath)
        if fastest:
            # Return mmap-backed views directly (fastest, zero-copy)
            if config.BACKEND == 'torch':
                import torch
                for tensor in reader.tensors:
                    loaded_tensors[tensor.name] = torch.from_numpy(tensor.data.copy())
            else:
                for tensor in reader.tensors:
                    loaded_tensors[tensor.name] = tensor.data
        else:
            if config.BACKEND == 'torch':
                import torch
                for tensor in reader.tensors:
                    loaded_tensors[tensor.name] = torch.from_numpy(tensor.data.copy())
            else:
                for tensor in reader.tensors:
                    loaded_tensors[tensor.name] = np.array(tensor.data)  # force copy for fair comparison

    elif format_name == "npz":
        data = np.load(filepath)
        if config.BACKEND == 'torch':
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
        if config.BACKEND == 'torch':
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
        framework = "pt" if config.BACKEND == 'torch' else "numpy"
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
            if config.BACKEND == 'torch':
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
        if config.BACKEND == 'torch':
            import torch
            for tensor in selected:
                loaded_tensors[tensor.name] = torch.from_numpy(tensor.data.copy())
        else:
            for tensor in selected:
                loaded_tensors[tensor.name] = np.array(tensor.data)  # force copy for fair comparison

    elif format_name == "npz":
        data = np.load(filepath)
        all_keys = list(data.files)
        step = max(1, int(1.0 / fraction))
        selected = all_keys[::step]
        if config.BACKEND == 'torch':
            import torch
            for k in selected:
                loaded_tensors[k] = torch.from_numpy(data[k])
        else:
            for k in selected:
                loaded_tensors[k] = data[k]

    elif format_name == "onnx":
        import onnx
        from onnx import numpy_helper
        model = onnx.load(filepath)
        all_inits = list(model.graph.initializer)
        step = max(1, int(1.0 / fraction))
        selected = all_inits[::step]
        if config.BACKEND == 'torch':
            import torch
            for init in selected:
                loaded_tensors[init.name] = torch.from_numpy(numpy_helper.to_array(init).copy())
        else:
            for init in selected:
                loaded_tensors[init.name] = numpy_helper.to_array(init)

    if not loaded_tensors:
        print(f"  WARNING: No tensors loaded for {format_name} in selective mode (unhandled format?)")
        return float('inf')

    # Force data access
    for t in loaded_tensors.values():
        _ = t.sum()

    end = time.perf_counter()
    return end - start


def run_single_benchmark(
    fmt, tensors, filepath,
    runs=None, warmup=None,
    scenario="full_load",
):
    """
    Run write + read benchmark for a single format with warmup and multiple runs.
    scenario: "full_load" (default), "zero_copy", "selective", "fastest"
    Returns dict with median write throughput, median read latency, and file size.
    """
    if runs is None:
        runs = config.BENCH_RUNS
    if warmup is None:
        warmup = config.WARMUP_RUNS

    write_speeds = []
    read_latencies = []
    file_size_gb = 0
    data_size_gb = 0

    # For zero_copy scenario, use ztensor_zerocopy read path
    read_fmt = "ztensor_zerocopy" if (scenario == "zero_copy" and fmt == "ztensor") else fmt

    for i in range(warmup + runs):
        is_warmup = i < warmup

        # Write
        try:
            w_speed, file_size_gb_val, data_size_gb_val = benchmark_write(fmt, tensors, filepath)
            file_size_gb = file_size_gb_val
            data_size_gb = data_size_gb_val
        except Exception as e:
            print(f"  FAIL Write {fmt}: {e}")
            return None

        if not is_warmup:
            write_speeds.append(w_speed)

        # Free previous iteration's tensors before reading
        gc.collect()

        # Read (with cache dropping)
        drop_file_cache(filepath)

        try:
            if scenario == "selective":
                r_lat = benchmark_read_selective(fmt, filepath, fraction=0.1)
            elif scenario == "zero_copy":
                r_lat = benchmark_read(read_fmt, filepath)
            elif scenario == "fastest":
                r_lat = benchmark_read(fmt, filepath, fastest=True)
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
        "data_size_gb": data_size_gb,
    }
