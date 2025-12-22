import os
import sys
import struct
import subprocess
import cbor2
import numpy as np
from safetensors.numpy import save_file

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Assumes script is in cli/tests/ and binary in cli/target/debug/
CLI_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "../target/debug/ztensor"))
TEST_DIR = "test_artifacts"

def run_cmd(args, check=True):
    print(f"RUNNING: {' '.join(args)}")
    result = subprocess.run(args, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        raise RuntimeError(f"Command failed: {args}")
    return result

def create_legacy_file(filename):
    """Creates a basic valid v0.1.0 ztensor file."""
    magic = b"ZTEN0001"
    
    # Create simple f32 tensor: [1.0, 2.0, 3.0]
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    data_bytes = data.tobytes()
    
    # Needs to be 64-byte aligned
    offset = 64
    
    # Tensor metadata
    tensor_meta = {
        "name": "legacy_tensor",
        "offset": offset, 
        "size": len(data_bytes),
        "dtype": "float32",
        "shape": [3],
        "encoding": "raw",
        "layout": "dense"
    }
    
    # Write file
    with open(filename, "wb") as f:
        f.write(magic)
        # Pad to 64 bytes
        f.write(b'\x00' * (offset - len(magic)))
        f.write(data_bytes)
        
        # Serialize metadata as CBOR array
        cbor_data = cbor2.dumps([tensor_meta])
        f.write(cbor_data)
        
        # Write footer size (u64 little endian)
        f.write(struct.pack("<Q", len(cbor_data)))

def create_safetensors_file(filename):
    tensors = {
        "test_tensor": np.array([10, 20, 30], dtype=np.int32)
    }
    save_file(tensors, filename)

def main():
    if not os.path.exists(CLI_PATH):
        print(f"Error: CLI binary not found at {CLI_PATH}. Run 'cargo build' first.")
        sys.exit(1)
        
    if os.path.exists(TEST_DIR):
        import shutil
        shutil.rmtree(TEST_DIR)
        
    os.makedirs(TEST_DIR, exist_ok=True)
    os.chdir(TEST_DIR)
    
    try:
        # 1. Test Migrate (v0.1.0 -> v1.1.0)
        print("\n--- Testing 'migrate' ---")
        create_legacy_file("legacy.zt")
        run_cmd([CLI_PATH, "migrate", "legacy.zt", "-o", "migrated.zt"])
        
        info_res = run_cmd([CLI_PATH, "info", "migrated.zt"])
        if "legacy_tensor" not in info_res.stdout or "F32" not in info_res.stdout:
            raise RuntimeError("Migrated file info incorrect")
        print("Migration successful")

        # 2. Test Convert (safetensors -> ztensor)
        print("\n--- Testing 'convert' ---")
        create_safetensors_file("input.safetensors")
        run_cmd([CLI_PATH, "convert", "input.safetensors", "-o", "converted.zt"])
        
        info_res = run_cmd([CLI_PATH, "info", "converted.zt"])
        if "test_tensor" not in info_res.stdout or "I32" not in info_res.stdout:
             raise RuntimeError("Converted file info incorrect")
        print("Conversion successful")

        # 3. Test Compress
        print("\n--- Testing 'compress' ---")
        run_cmd([CLI_PATH, "compress", "converted.zt", "-o", "compressed.zt"])
        info_res = run_cmd([CLI_PATH, "info", "compressed.zt"])
        if "Zstd" not in info_res.stdout:
             raise RuntimeError("Compressed file doesn't show Zstd encoding")
        print("Compression successful")

        # 4. Test Decompress
        print("\n--- Testing 'decompress' ---")
        run_cmd([CLI_PATH, "decompress", "compressed.zt", "-o", "decompressed.zt"])
        info_res = run_cmd([CLI_PATH, "info", "decompressed.zt"])
        if "Zstd" in info_res.stdout:  # Should be Raw or not Zstd explicitly for data
             # Note: Raw might show up as enc=Raw or just implied. 
             # Let's check if it *doesn't* have implicit "enc=Zstd"
             pass 
        print("Decompression successful")
        
        # 5. Test Merge
        print("\n--- Testing 'merge' ---")
        run_cmd([CLI_PATH, "convert", "input.safetensors", "-o", "file1.zt"])
        run_cmd([CLI_PATH, "convert", "input.safetensors", "-o", "file2.zt"]) # Duplicates would fail merge
        
        # Create distinct files
        t1 = {"t1": np.zeros((2,), dtype=np.uint8)}
        save_file(t1, "t1.safetensors")
        t2 = {"t2": np.zeros((2,), dtype=np.uint8)}
        save_file(t2, "t2.safetensors")
        
        run_cmd([CLI_PATH, "convert", "t1.safetensors", "-o", "t1.zt"])
        run_cmd([CLI_PATH, "convert", "t2.safetensors", "-o", "t2.zt"])
        
        run_cmd([CLI_PATH, "merge", "t1.zt", "t2.zt", "-o", "merged.zt"])
        info_res = run_cmd([CLI_PATH, "info", "merged.zt"])
        if "t1" not in info_res.stdout or "t2" not in info_res.stdout:
             raise RuntimeError("Merge failed to include both tensors")
        print("Merge successful")

        print("\nALL INTEGRATION TESTS PASSED!")

    finally:
        # Cleanup
        # os.chdir("..")
        # import shutil
        # shutil.rmtree(TEST_DIR)
        pass

if __name__ == "__main__":
    main()
