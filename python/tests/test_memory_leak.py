import os
import psutil
import numpy as np
import ztensor
import time
import gc

def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def test_memory_stability():
    filename = "leak_test.zt"
    
    # 1. Create a moderately sized file
    shape = (1000, 1000) # 1M float32 = 4MB
    data = np.random.randn(*shape).astype(np.float32)
    
    with ztensor.Writer(filename) as w:
        w.add_tensor("data", data, compress=False)
        w.add_tensor("compressed", data, compress=True) # Forces copy path

    print(f"Initial Memory: {get_process_memory():.2f} MB")
    
    # Warmup
    for _ in range(100):
        with ztensor.Reader(filename) as r:
            _ = r.read_tensor("data")
            _ = r.read_tensor("compressed")
            _ = r.read_tensors(["data", "compressed"])
    
    gc.collect()
    start_mem = get_process_memory()
    print(f"After Warmup: {start_mem:.2f} MB")
    
    iterations = 5000
    for i in range(iterations):
        # 1. Zero-copy read
        reader = ztensor.Reader(filename)
        t1 = reader.read_tensor("data")
        # Check that it is zero-copy: has _reader_ref
        assert hasattr(t1, '_reader_ref')
        
        # 2. Copy read (compressed)
        t2 = reader.read_tensor("compressed")
        # Check that it is NOT zero-copy: no _reader_ref
        assert not hasattr(t2, '_reader_ref')
        
        # 3. Batch read (mixed)
        # Note: 'compressed' will be copied, 'data' will be zero-copy
        tensors = reader.read_tensors(["data", "compressed"])
        assert hasattr(tensors[0], '_reader_ref')
        assert not hasattr(tensors[1], '_reader_ref')
        
        # Explicit delete to trigger GC/ref-counting
        del t1
        del t2
        del tensors
        del reader
        
        if i % 1000 == 0:
            gc.collect()
            curr_mem = get_process_memory()
            print(f"Iter {i}: {curr_mem:.2f} MB (Delta: {curr_mem - start_mem:.2f} MB)")
            
    gc.collect()
    end_mem = get_process_memory()
    print(f"Final Memory: {end_mem:.2f} MB")
    print(f"Total Growth: {end_mem - start_mem:.2f} MB")
    
    # Allow some small fluctuations (fragmentation, python internal), but not linear growth
    # 5000 iters * (struct overhead) should not be visible if leaks are plugged.
    if (end_mem - start_mem) > 10.0: # 10MB tolerance
        raise RuntimeError("Memory leak detected! Growth > 10MB")
    else:
        print("[PASS] No significant memory leak detected.")

    if os.path.exists(filename):
        os.remove(filename)

if __name__ == "__main__":
    test_memory_stability()
