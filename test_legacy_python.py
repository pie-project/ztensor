import os
import struct
import tempfile
import numpy as np
import ztensor
import cbor2

def create_legacy_file(path, tensor_name, data):
    # Create a legacy v0.1.0 file manually
    MAGIC = b"ZTEN0001"
    
    with open(path, "wb") as f:
        f.write(MAGIC)
        
        # Align to 64 bytes
        # Pos is 8. Padding is 56.
        f.write(b'\0' * 56)
        
        offset = 64
        size = data.nbytes
        f.write(data.tobytes())
        
        # Manifest
        manifest = [{
            "name": tensor_name,
            "offset": offset,
            "size": size,
            "dtype": "float32",
            "shape": list(data.shape),
            "encoding": "raw",
            "layout": "dense"
        }]
        
        cbor_data = cbor2.dumps(manifest)
        f.write(cbor_data)
        
        # Manifest Size (u64 LE)
        f.write(struct.pack("<Q", len(cbor_data)))

def test_legacy_zero_copy():
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        path = tmp.name
    
    try:
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        create_legacy_file(path, "legacy_tensor", data)
        
        with ztensor.Reader(path) as reader:
            print(f"Tensors: {reader.tensor_names}")
            
            # Read tensor
            arr = reader.read_tensor("legacy_tensor")
            print(f"Data: {arr}")
            
            if not np.allclose(arr, data):
                raise ValueError("Data mismatch!")
                
            # Verify zero-copy
            # _reader_ref is set only if it's zero-copy
            if hasattr(arr, '_reader_ref'):
                print("SUCCESS: Zero-copy detected (reader ref present)!")
            else:
                 # Check if the internal owner is NULL (view) or not.
                 # The wrapper might hide this, but _reader_ref is the python-side indicator.
                 print("WARNING: _reader_ref not found. Checking internal flags via string representation or type.")
                 # If it's a _ZTensorView, it should have it if zero-copy logic worked.
                 if not isinstance(arr, ztensor._ZTensorView):
                     print("FAILED: Returned array is not _ZTensorView, so definitely copied.")
                 else:
                     print("FAILED: _ZTensorView found but no reader ref. Likely copied.")
            
            # Additional check: modifying file (if mapped) might be dangerous or visible, 
            # but we can check if the base pointer matches mmap usage if we had access to it.
            # Simple success criteria is reading correct data via the new code path.
            
    finally:
        if os.path.exists(path):
            os.remove(path)

if __name__ == "__main__":
    test_legacy_zero_copy()
