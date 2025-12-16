import numpy as np
from ztensor import Writer
import os

file_path = "cli_test.zt"
if os.path.exists(file_path):
    os.remove(file_path)

print(f"--- Writing to {file_path} ---")
with Writer(file_path) as writer:
    tensor1 = np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float32)
    tensor2 = np.arange(24, dtype=np.uint8).reshape((2, 3, 4))
    
    writer.add_tensor("float_tensor", tensor1)
    writer.add_tensor("int_tensor", tensor2)
    print("Write complete.")
