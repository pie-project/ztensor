# Runtime globals — set by main() via `import config; config.BACKEND = ...`
# Other modules must access as `config.BACKEND`, NOT `from config import BACKEND`.

BACKEND = "numpy"
WARMUP_RUNS = 1
BENCH_RUNS = 3

# --- Format constants ---

# Formats that write + read with their own native library
NATIVE_FORMATS = [
    "ztensor",
    "ztensor_zstd",
    "safetensors",
    "pickle",
    "hdf5",
    "gguf",
    "npz",
    "onnx",
]

# Zstd compression levels to benchmark
ZSTD_FORMATS = ["ztensor_zstd3", "ztensor_zstd13", "ztensor_zstd22"]

# Zero-copy variant (separate from NATIVE_FORMATS so default benchmarks are unchanged)
ZEROCOPY_FORMATS = ["ztensor_zerocopy"]

# Cross-format read: write with native lib, read via ztensor
CROSS_READ_FORMATS = [
    "zt_read_st",
    "zt_read_pt",
    "zt_read_gguf",
    "zt_read_npz",
    "zt_read_onnx",
    "zt_read_h5",
]

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
