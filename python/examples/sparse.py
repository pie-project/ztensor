
import os
import numpy as np
import torch
from ztensor import Writer, Reader, ZTensorError

FILE_PATH = "sparse_test.zt"

def test_sparse_csr():
    print("\n--- Testing Sparse CSR ---")
    # CSR: value, indices (col), indptr (row start)
    # Matrix:
    # [[0, 1.0, 0],
    #  [0, 0, 2.0]]
    # Shape: (2, 3)
    values = np.array([1.0, 2.0], dtype=np.float32)
    indices = np.array([1, 2], dtype=np.uint64)
    indptr = np.array([0, 1, 2], dtype=np.uint64)
    shape = (2, 3)

    with Writer(FILE_PATH) as writer:
        writer.add_sparse_csr("csr_tensor", values, indices, indptr, shape)
        print("Written CSR tensor.")

    with Reader(FILE_PATH) as reader:
        # Read as Torch
        t_csr = reader.read_tensor("csr_tensor", to="torch")
        print("Read Torch CSR:", t_csr)
        assert t_csr.is_sparse_csr
        assert t_csr.shape == shape
        # Verify values
        assert torch.allclose(t_csr.values(), torch.from_numpy(values))

        # Try to read as Numpy (requires Scipy)
        try:
            import scipy
            s_csr = reader.read_tensor("csr_tensor", to="numpy")
            print("Read Scipy CSR:\n", s_csr.toarray())
            assert np.allclose(s_csr.data, values)
        except ImportError:
            print("Skipping Scipy test (scipy not installed).")

def test_sparse_coo():
    print("\n--- Testing Sparse COO ---")
    # COO:
    # [[10, 0],
    #  [0, 20]]
    # Shape: (2, 2)
    # Values: [10, 20] at (0,0) and (1,1)
    values = np.array([10, 20], dtype=np.int32)
    # Indices: (ndim, nnz) -> [[0, 1], [0, 1]]
    indices = np.array([[0, 1], [0, 1]], dtype=np.uint64)
    shape = (2, 2)

    with Writer(FILE_PATH) as writer:
        writer.add_sparse_coo("coo_tensor", values, indices, shape)
        print("Written COO tensor.")

    with Reader(FILE_PATH) as reader:
        # Read as Torch
        t_coo = reader.read_tensor("coo_tensor", to="torch")
        print("Read Torch COO:", t_coo)
        assert t_coo.is_sparse
        assert t_coo.shape == shape
        # Verify using to_dense()
        dense = t_coo.to_dense()
        expected = torch.tensor([[10, 0], [0, 20]], dtype=torch.int32)
        assert torch.equal(dense, expected)

if __name__ == "__main__":
    try:
        test_sparse_csr()
        os.remove(FILE_PATH)
        test_sparse_coo()
    finally:
        if os.path.exists(FILE_PATH):
            os.remove(FILE_PATH)
