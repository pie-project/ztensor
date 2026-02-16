import os
import argparse

import config
from datagen import generate_tensor_dict
from runners import run_single_benchmark
from sweep import run_sweep
from plot import draw_plot


def main():
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
    parser.add_argument("--scenario", type=str, choices=["full_load", "zero_copy", "selective", "fastest"],
                        default="full_load", help="Benchmark scenario (default: full_load)")
    parser.add_argument("--dist", type=str, choices=["mixed", "small", "large", "llama-1b"],
                        default="mixed", help="Tensor distribution (default: mixed)")
    args = parser.parse_args()

    config.BACKEND = args.backend
    config.BENCH_RUNS = args.runs
    config.WARMUP_RUNS = args.warmup

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
    formats = args.formats or config.NATIVE_FORMATS

    os.makedirs("bench_out", exist_ok=True)

    scenario_label = f", {scenario}" if scenario != "full_load" else ""
    style_label = f", {data_style}" if data_style != "random" else ""
    size_label = f"{size_mb}MB " if distribution != "llama-1b" else ""
    print(f"\n=== Single Run: {size_label}{distribution}{style_label}{scenario_label}, "
          f"{config.BENCH_RUNS} runs + {config.WARMUP_RUNS} warmup ===\n")
    print(f"{'Format':<20} | {'Write (GB/s)':>12} | {'Read (s)':>12} | {'Read (GB/s)':>12} | {'File (MB)':>10}")
    print("-" * 80)

    for fmt in formats:
        filepath = config.filepath_for_format(fmt)

        result = run_single_benchmark(fmt, tensors, filepath, scenario=scenario)

        if result is None:
            continue

        w_speed = result["write_median"]
        r_lat = result["read_median"]
        file_mb = result["file_size_gb"] * 1024
        data_size_gb = result["data_size_gb"]
        r_tp = data_size_gb / r_lat if r_lat > 0 else 0

        print(f"{fmt:<20} | {w_speed:>12.2f} | {r_lat:>12.4f} | {r_tp:>12.2f} | {file_mb:>10.1f}")

        if os.path.exists(filepath):
            os.remove(filepath)

    print()


if __name__ == "__main__":
    main()
