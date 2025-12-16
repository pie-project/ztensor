import os

import numpy as np
from ztensor import Writer, Reader, ZTensorError

file_path = "test_tensors.zt"

# --- Write Tensors ---
print(f"--- Writing to {file_path} ---")
try:
    with Writer(file_path) as writer:
        # Create some numpy arrays
        tensor1 = np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float32)
        tensor2 = np.arange(24, dtype=np.uint8).reshape((2, 3, 4))

        print(f"Adding tensor 'float_tensor' with shape {tensor1.shape}")
        writer.add_tensor("float_tensor", tensor1)

        print(f"Adding tensor 'int_tensor' with shape {tensor2.shape}")
        writer.add_tensor("int_tensor", tensor2)

        print("Finalizing writer...")
    print("Write complete.")

except ZTensorError as e:
    print(f"An error occurred during writing: {e}")

# --- Read and Verify Tensors ---
if os.path.exists(file_path):
    print(f"\n--- Reading from {file_path} ---")
    try:
        with Reader(file_path) as reader:
            # Read tensor1
            print("Reading 'float_tensor'...")
            read_tensor1 = reader.read_tensor("float_tensor")
            print(f"Read tensor with shape {read_tensor1.shape} and dtype {read_tensor1.dtype}")
            print(read_tensor1)
            print(tensor1)
            assert np.array_equal(tensor1, read_tensor1)

            # Read tensor2
            print("\nReading 'int_tensor'...")
            read_tensor2 = reader.read_tensor("int_tensor")
            print(f"Read tensor with shape {read_tensor2.shape} and dtype {read_tensor2.dtype}")
            print(read_tensor2)
            assert np.array_equal(tensor2, read_tensor2)

            print("\nVerification successful!")

    except ZTensorError as e:
        print(f"An error occurred during reading: {e}")
    finally:
        # Clean up the test file
        os.remove(file_path)