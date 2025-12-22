import torch
from torch.testing._internal.common_utils import TestCase
import unittest
import sys
import os

# Skip tests if mpi4py is not available
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False

try:
    import torch_mpi_ext
    EXTENSION_AVAILABLE = True
except ImportError:
    EXTENSION_AVAILABLE = False

@unittest.skipIf(not MPI_AVAILABLE, "mpi4py not available")
@unittest.skipIf(not EXTENSION_AVAILABLE, "torch_mpi_ext not available")
class TestMPIOperations(TestCase):
    def setUp(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.comm_ptr = self.comm.py2f()
    def test_all_reduce_different_dtypes(self):
        """Test all_reduce with different data types"""
        dtypes = [torch.float32, torch.float64, torch.int8, torch.int16, 
                  torch.int32, torch.int64, torch.float16, torch.bfloat16]
        
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                if dtype in [torch.float32, torch.float64]:
                    tensor = torch.ones(3, dtype=dtype) * (self.rank + 1)
                else:
                    tensor = torch.ones(3, dtype=dtype) * (self.rank + 1)
                    
                result = torch_mpi_ext.ops.all_reduce(tensor, self.comm_ptr)
                
                expected_sum = self.size * (self.size + 1) / 2
                if dtype in [torch.float32, torch.float64]:
                    expected = torch.ones(3, dtype=dtype) * expected_sum
                else:
                    expected = torch.ones(3, dtype=dtype) * int(expected_sum)
                    
                torch.testing.assert_close(result, expected)

    def test_all_reduce_inplace_different_dtypes(self):
        """Test all_reduce with different data types"""
        dtypes = [torch.float32, torch.float64, torch.int8, torch.int16, 
                  torch.int32, torch.int64, torch.float16, torch.bfloat16]
        
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                tensor = torch.ones(3, dtype=dtype) * (self.rank + 1)
                    
                result = torch_mpi_ext.ops.all_reduce_(tensor, self.comm_ptr)
                self.assertTrue(result == None)

                expected_sum = self.size * (self.size + 1) / 2
                if dtype in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
                    expected = torch.ones(3, dtype=dtype) * expected_sum
                else:
                    expected = torch.ones(3, dtype=dtype) * int(expected_sum)
                    
                torch.testing.assert_close(tensor, expected)

    def test_all_gather_negative_dim(self):
        """Test all_gather with negative dim"""

        dims = [-1, -2, -3, 0, 1, 2]

        for dim in dims:
            with self.subTest(dim=dim):

                # Create rank-specific tensor
                tensor = torch.arange(0, 2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4) * (self.rank + 1)

                # Perform all-gather on dim=-2 (should be same as dim=1)
                result = torch_mpi_ext.ops.all_gather(tensor, comm_ptr=self.comm_ptr, dim=dim)

                expected_shape = [2, 3, 4]
                expected_shape[dim] *= self.size
                self.assertEqual(result.shape, torch.Size(expected_shape))

                expected_parts = []
                for r in range(self.size):
                    part = torch.arange(0, 2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4) * (r + 1)
                    expected_parts.append(part)

                expected = torch.cat(expected_parts, dim=dim)

                # Check values
                torch.testing.assert_close(result, expected)

if __name__ == "__main__":
    # Only run tests if we're in an MPI environment
    try:
        from mpi4py import MPI
        unittest.main()
    except ImportError:
        print("Skipping MPI tests as mpi4py is not available")
        sys.exit(0)