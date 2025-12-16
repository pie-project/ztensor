#!/usr/bin/env python
"""Quick test for ztensor.numpy module."""
import tempfile
import os

import numpy as np
from ztensor.numpy import save_file, load_file, save, load

print("Testing ztensor.numpy module...")

# Test 1: save_file and load_file
print("1. Testing save_file/load_file...", end=" ")
tensors = {"a": np.zeros((2, 3)), "b": np.ones((4,))}
with tempfile.NamedTemporaryFile(suffix=".zt", delete=False) as f:
    path = f.name
try:
    save_file(tensors, path)
    loaded = load_file(path)
    assert set(loaded.keys()) == {"a", "b"}
    assert np.allclose(loaded["a"], tensors["a"])
    assert np.allclose(loaded["b"], tensors["b"])
    print("OK")
finally:
    os.unlink(path)

# Test 2: save and load (bytes)
print("2. Testing save/load (bytes)...", end=" ")
data = save(tensors)
assert isinstance(data, bytes)
loaded = load(data)
assert np.allclose(loaded["a"], tensors["a"])
print("OK")

# Test 3: Various dtypes
print("3. Testing various dtypes...", end=" ")
tensors = {
    "float32": np.array([1.0, 2.0], dtype=np.float32),
    "float64": np.array([1.0, 2.0], dtype=np.float64),
    "int32": np.array([1, 2], dtype=np.int32),
    "int64": np.array([1, 2], dtype=np.int64),
}
data = save(tensors)
loaded = load(data)
for name in tensors:
    assert np.allclose(loaded[name], tensors[name]), f"Mismatch for {name}"
print("OK")

print("\nAll tests passed!")
