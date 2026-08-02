import os

import config
from datagen import generate_tensor_dict
from runners import run_single_benchmark
from plot import draw_plot


def run_sweep():
    import pandas as pd

    sizes = [128, 512, 1024, 2048]  # MB
    distributions = ["mixed", "small", "large"]
    formats = config.NATIVE_FORMATS  # Only native formats in main sweep

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
            "formats": config.CROSS_READ_FORMATS,
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
            "formats": ["ztensor"] + config.ZSTD_FORMATS,
        },
        {
            "data_style": "structured",
            "scenario": "full_load",
            "distributions": ["mixed"],
            "sizes": [512],
            "formats": ["ztensor"] + config.ZSTD_FORMATS,
        },
    ]

    print(f"=== Benchmark Sweep ===")
    print(f"Runs per data point: {config.BENCH_RUNS} (+ {config.WARMUP_RUNS} warmup)")
    print()

    for cfg in sweep_configs:
        data_style = cfg["data_style"]
        scenario = cfg["scenario"]

        print(f"--- {scenario} / {data_style} ---")
        print(
            f"  Sizes: {cfg['sizes']}, Dists: {cfg['distributions']}, "
            f"Formats: {cfg['formats']}"
        )
        print()

        for dist in cfg["distributions"]:
            for size_mb in cfg["sizes"]:
                tensors = generate_tensor_dict(size_mb, dist, data_style)
                tensor_count = len(tensors)

                for fmt in cfg["formats"]:
                    filepath = config.filepath_for_format(fmt, "bench_out", "sweep")

                    result = run_single_benchmark(
                        fmt,
                        tensors,
                        filepath,
                        scenario=scenario,
                    )

                    if result is None:
                        continue

                    # Clean up
                    if os.path.exists(filepath):
                        os.remove(filepath)

                    w_speed = result["write_median"]
                    r_lat = result["read_median"]
                    size_gb = result["file_size_gb"]

                    print(
                        f"  {fmt:<16} | {size_mb:>4}MB ({dist:<5}) | "
                        f"W: {w_speed:.2f} GB/s | R: {r_lat:.4f}s | "
                        f"File: {size_gb*1024:.1f}MB | Tensors: {tensor_count} "
                        f"| {scenario}/{data_style}"
                    )

                    results.append(
                        {
                            "Format": fmt,
                            "SizeMB": size_mb,
                            "Distribution": dist,
                            "DataStyle": data_style,
                            "Scenario": scenario,
                            "WriteGBs": w_speed,
                            "ReadSeconds": r_lat,
                            "FileSizeGB": size_gb,
                            "DataSizeGB": result["data_size_gb"],
                            "TensorCount": tensor_count,
                        }
                    )

                print()

    # Save CSV
    df = pd.DataFrame(results)
    df.to_csv("bench_out/sweep_results.csv", index=False)
    print("Sweep complete. Results saved to bench_out/sweep_results.csv")
    draw_plot("bench_out/sweep_results.csv")
