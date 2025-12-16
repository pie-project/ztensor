#!/usr/bin/env python
"""Quick test for ztensor.torch module."""
import tempfile
import os

import torch
from ztensor.torch import save_file, load_file, save, load, save_model, load_model

print("Testing ztensor.torch module...")

# Test 1: save_file and load_file
print("1. Testing save_file/load_file...", end=" ")
tensors = {"a": torch.zeros((2, 3)), "b": torch.ones((4,))}
with tempfile.NamedTemporaryFile(suffix=".zt", delete=False) as f:
    path = f.name
try:
    save_file(tensors, path)
    loaded = load_file(path)
    assert set(loaded.keys()) == {"a", "b"}
    assert torch.allclose(loaded["a"], tensors["a"])
    assert torch.allclose(loaded["b"], tensors["b"])
    print("OK")
finally:
    os.unlink(path)

# Test 2: save and load (bytes)
print("2. Testing save/load (bytes)...", end=" ")
data = save(tensors)
assert isinstance(data, bytes)
loaded = load(data)
assert torch.allclose(loaded["a"], tensors["a"])
print("OK")

# Test 3: save_model and load_model
print("3. Testing save_model/load_model...", end=" ")
model = torch.nn.Linear(5, 3)
with tempfile.NamedTemporaryFile(suffix=".zt", delete=False) as f:
    path = f.name
try:
    save_model(model, path)
    model2 = torch.nn.Linear(5, 3)
    missing, unexpected = load_model(model2, path)
    assert len(missing) == 0 and len(unexpected) == 0
    print("OK")
finally:
    os.unlink(path)

print("\nAll tests passed!")
