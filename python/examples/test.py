import unittest
import os
import shutil
import tempfile
import numpy as np

# --- Conditional PyTorch Import ---
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# --- Import the ztensor wrapper ---
# Assuming the bindings file is named 'bindings.py' in a 'ztensor' package/directory
from ztensor import Reader, Writer, ZTensorError, TensorMetadata


class TestZTensorBindings(unittest.TestCase):

    def setUp(self):
        """Set up a temporary directory for test files."""
        self.test_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.test_dir, "test.zt")

    def tearDown(self):
        """Remove the temporary directory and its contents."""
        shutil.rmtree(self.test_dir)

    def test_01_writer_and_reader_numpy(self):
        """Test writing and reading a single NumPy tensor."""
        tensor_a = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        tensor_b = np.array([[True, False], [False, True]], dtype=bool)

        with Writer(self.test_file) as writer:
            writer.add_tensor("tensor_a", tensor_a)
            writer.add_tensor("tensor_b", tensor_b)

        self.assertTrue(os.path.exists(self.test_file))

        with Reader(self.test_file) as reader:
            read_a = reader.read_tensor("tensor_a")
            read_b = reader.read_tensor("tensor_b")

            self.assertIsInstance(read_a, np.ndarray)
            self.assertTrue(np.array_equal(tensor_a, read_a))
            self.assertEqual(tensor_a.dtype, read_a.dtype)
            self.assertEqual(tensor_a.shape, read_a.shape)

            self.assertIsInstance(read_b, np.ndarray)
            self.assertTrue(np.array_equal(tensor_b, read_b))

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not installed")
    def test_02_writer_and_reader_torch(self):
        """Test writing and reading a PyTorch tensor."""
        tensor_a = torch.randn(10, 20, dtype=torch.float16)
        tensor_b = torch.randint(0, 255, (100,), dtype=torch.uint8)

        with Writer(self.test_file) as writer:
            writer.add_tensor("torch_a", tensor_a)
            writer.add_tensor("torch_b", tensor_b)

        with Reader(self.test_file) as reader:
            # Read back as torch tensor
            read_a_torch = reader.read_tensor("torch_a", to='torch')
            self.assertIsInstance(read_a_torch, torch.Tensor)
            self.assertTrue(torch.equal(tensor_a, read_a_torch))
            self.assertEqual(tensor_a.dtype, read_a_torch.dtype)

            # Read back as numpy array
            read_b_np = reader.read_tensor("torch_b", to='numpy')
            self.assertIsInstance(read_b_np, np.ndarray)
            self.assertTrue(np.array_equal(tensor_b.numpy(), read_b_np))

    def test_03_metadata_access_and_iteration(self):
        """Test the reader's container and metadata features."""
        tensors = {
            "tensor_int": np.ones((5, 5), dtype=np.int32),
            "tensor_float": np.zeros((10,), dtype=np.float64),
            "scalar": np.array(3.14, dtype=np.float32)
        }
        tensor_names = sorted(tensors.keys())

        with Writer(self.test_file) as writer:
            for name, tensor in tensors.items():
                writer.add_tensor(name, tensor)

        with Reader(self.test_file) as reader:
            # Test __len__
            self.assertEqual(len(reader), 3)

            # Test get_tensor_names
            read_names = sorted(reader.get_tensor_names())
            self.assertEqual(tensor_names, read_names)

            # Test __iter__ and __getitem__
            all_meta = []
            for i in range(len(reader)):
                meta = reader[i]
                self.assertIsInstance(meta, TensorMetadata)
                all_meta.append(meta.name)
            self.assertEqual(tensor_names, sorted(all_meta))

            # Test list_tensors
            self.assertEqual(len(reader.list_tensors()), 3)

            # Test specific metadata properties
            meta_scalar = reader.get_metadata("scalar")
            self.assertEqual(meta_scalar.name, "scalar")
            self.assertEqual(meta_scalar.shape, (1, ))
            self.assertEqual(meta_scalar.dtype, np.dtype('float32'))
            self.assertEqual(meta_scalar.dtype_str, 'float32')
            self.assertGreater(meta_scalar.offset, 0)
            self.assertGreater(meta_scalar.size, 0)
            self.assertEqual(meta_scalar.layout, "dense")
            self.assertEqual(meta_scalar.encoding, "raw")
            self.assertIn(meta_scalar.endianness, ["little", "big"])
            self.assertIsNone(meta_scalar.checksum)

    def test_04_error_handling(self):
        """Test expected failure modes."""
        tensor = np.arange(10)
        with Writer(self.test_file) as writer:
            writer.add_tensor("my_tensor", tensor)

        # Test reading non-existent tensor
        with Reader(self.test_file) as reader:
            with self.assertRaisesRegex(ZTensorError, "Tensor not found"):
                reader.read_tensor("non_existent_tensor")

        # Test accessing closed reader
        reader = Reader(self.test_file)
        reader.__exit__(None, None, None)  # Manually close
        with self.assertRaisesRegex(ZTensorError, "Reader is closed"):
            len(reader)
        with self.assertRaisesRegex(ZTensorError, "Reader is closed"):
            reader.read_tensor("my_tensor")

        # Test index out of range
        with Reader(self.test_file) as reader:
            with self.assertRaises(IndexError):
                reader[99]

        # Test writing to a finalized writer
        writer = Writer(self.test_file)
        writer.add_tensor("t1", tensor)
        writer.finalize()
        with self.assertRaisesRegex(ZTensorError, "Writer is closed or finalized."):
            writer.add_tensor("t2", tensor)

    def test_05_writer_context_manager_exception(self):
        """Ensure writer does not finalize on error inside `with` block."""
        bad_file_path = os.path.join(self.test_dir, "bad_file.zt")
        try:
            with Writer(bad_file_path) as writer:
                writer.add_tensor("good_tensor", np.arange(5))
                raise ValueError("Simulating an error during writing")
        except ValueError:
            pass  # Expected

        # The file should exist but be empty/invalid because finalize was not called.
        self.assertTrue(os.path.exists(bad_file_path))
        # Attempting to read should fail
        with self.assertRaises(ZTensorError):
            with Reader(bad_file_path) as reader:
                len(reader)


if __name__ == '__main__':
    unittest.main(verbosity=2)
