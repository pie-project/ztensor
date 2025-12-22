import os
import numpy as np
import ztensor
import pytest

def test_zerocopy_read():
    filename = "test_zerocopy.zt"
    
    try:
        # 1. Create a large tensor
        shape = (1000, 1000)
        data = np.random.randn(*shape).astype(np.float32)
        
        # 2. Write to file
        with ztensor.Writer(filename) as w:
            w.add_tensor("data", data, compress=False)
            
        # 3. Read back with Reader
        with ztensor.Reader(filename) as reader:
            retrieved = reader.read_tensor("data")
            
            # 4. Verify Correctness
            assert np.allclose(retrieved, data), "Data mismatch!"
            
            # 5. Verify Zero-Copy
            # If zero-copy, OWNDATA should be False
            assert not retrieved.flags['OWNDATA'], "Array should not own data (zero-copy)"
            
            # Check base object
            assert retrieved.base is not None, "Array base should be set (view)"
            
            # Verify internal reference to reader to prevent UAF
            assert hasattr(retrieved, '_reader_ref'), "Array should keep a reference to reader"
            assert retrieved._reader_ref is reader, "Reader reference mismatch"

            print("\n[PASS] Single zero-copy verification successful!")

            # 6. Verify Batch Zero-Copy
            results = reader.read_tensors(["data"])
            assert len(results) == 1
            batch_retrieved = results[0]
            
            assert np.allclose(batch_retrieved, data), "Batch Data mismatch!"
            assert not batch_retrieved.flags['OWNDATA'], "Batch Array should not own data (zero-copy)"
            assert hasattr(batch_retrieved, '_reader_ref'), "Batch Array should keep a reference to reader"
            assert batch_retrieved._reader_ref is reader, "Batch Reader reference mismatch"
            
            print("\n[PASS] Batch zero-copy verification successful!")
            print(f"Array flags: {batch_retrieved.flags}")
            
    finally:
        if os.path.exists(filename):
            os.remove(filename)

if __name__ == "__main__":
    test_zerocopy_read()
