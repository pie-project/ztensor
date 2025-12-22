import time
import numpy as np
from ztensor import Reader, Writer
import os

def create_test_file(filename="model.zt"):
    """Create a test file with some random tensors if it doesn't exist."""
    print(f"Generating test file '{filename}'...")
    with Writer(filename) as writer:
        # 100MB float32 tensor
        writer.add_tensor("large_matrix", np.random.randn(5000, 5000).astype(np.float32))
        # Small tensor
        writer.add_tensor("bias", np.random.randn(5000).astype(np.float32))
        # Int tensor
        writer.add_tensor("counts", np.arange(1000, dtype=np.int64))

def main():
    filename = "model.zt"
    
    # Create file if it doesn't exist to make the test runnable out of the box
    if not os.path.exists(filename):
        create_test_file(filename)
    
    print("-" * 50)
    print(f"Testing ztensor with '{filename}'")
    print("-" * 50)

    # 1. Open and List
    t0 = time.perf_counter()
    with Reader(filename) as reader:
        t_open = time.perf_counter() - t0
        print(f"File opened in: {t_open * 1000:.2f} ms")
        print(f"Total tensors:  {len(reader)}")
        
        print("\nAvailable Tensors:")
        for meta in reader:
            print(f"  - {meta.name:<15} {str(meta.shape):<15} {meta.dtype_str:<10} {meta.size / 1024 / 1024:.2f} MB")

        # 2. Load and Benchmark
        print("\nPerformance Metrics:")
        total_bytes = 0
        t_start_load = time.perf_counter()
        
        for meta in reader:
            t_tensor_start = time.perf_counter()
            data = reader[meta.name]
            t_tensor_end = time.perf_counter()
            
            # Verify it's actually loaded (force read if lazy)
            # ztensor loads eagerly on access usually, but good to touch data
            _ = data.shape 
            
            mb = meta.size / (1024 * 1024)
            dur = t_tensor_end - t_tensor_start
            throughput = mb / dur if dur > 0 else 0
            
            print(f"  Loaded {meta.name:<15} ({mb:>6.2f} MB) in {dur:>6.4f}s ({throughput:>7.2f} MB/s)")
            total_bytes += meta.size

        t_total = time.perf_counter() - t_start_load
        
    total_mb = total_bytes / (1024 * 1024)
    avg_throughput = total_mb / t_total if t_total > 0 else 0
    
    print("-" * 50)
    print(f"Summary:")
    print(f"  Total Data:   {total_mb:.2f} MB")
    print(f"  Total Time:   {t_total:.4f}s")
    print(f"  Avg Speed:    {avg_throughput:.2f} MB/s")
    print("-" * 50)

if __name__ == "__main__":
    main()
