import unittest
import os
import shutil
import tempfile
import numpy as np
from ztensor import Reader, Writer, TensorMetadata

class TestNewAPI(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.test_dir, "api_test.zt")
        
        # Create a dummy file
        self.data = np.arange(10, dtype=np.float32)
        with Writer(self.test_file) as writer:
            writer.add_tensor("tensor_1", self.data)
            writer.add_tensor("tensor_2", self.data)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_properties(self):
        with Reader(self.test_file) as reader:
            # Test .tensor_names property
            self.assertEqual(sorted(reader.tensor_names), ["tensor_1", "tensor_2"])
            
            # Test .tensors property
            tensors = reader.tensors
            self.assertEqual(len(tensors), 2)
            self.assertIsInstance(tensors[0], TensorMetadata)

    def test_dict_access(self):
        with Reader(self.test_file) as reader:
            # Test __getitem__ with string
            t1 = reader["tensor_1"]
            self.assertTrue(np.array_equal(t1, self.data))
            
            # Test __contains__
            self.assertIn("tensor_1", reader)
            self.assertNotIn("tensor_3", reader)

    def test_metadata_renamed(self):
        with Reader(self.test_file) as reader:
            # Test metadata() method (renamed from get_metadata)
            meta = reader.metadata("tensor_1")
            self.assertEqual(meta.name, "tensor_1")

if __name__ == '__main__':
    unittest.main()
