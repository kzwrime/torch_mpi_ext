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

    def test_all_gather_into_tensor_different_dtypes_and_dims(self):
        """Test all_gather with negative dim"""

        dims = [-1, -2, -3, 0, 1, 2]
        dtypes = [torch.float32, torch.float64, torch.int8, torch.int16, 
                  torch.int32, torch.int64, torch.float16, torch.bfloat16]

        for dim in dims:
            for dtype in dtypes:
                with self.subTest(dim=dim, dtype=dtype):

                    # Create rank-specific tensor
                    tensor = torch.arange(0, 2 * 3 * 4, dtype=dtype).reshape(2, 3, 4) * (self.rank + 1)

                    # Perform all-gather on dim=-2 (should be same as dim=1)
                    result = torch_mpi_ext.ops.all_gather_into_tensor(tensor, comm_ptr=self.comm_ptr, dim=dim)

                    expected_shape = [2, 3, 4]
                    expected_shape[dim] *= self.size
                    self.assertEqual(result.shape, torch.Size(expected_shape))

                    expected_parts = []
                    for r in range(self.size):
                        part = torch.arange(0, 2 * 3 * 4, dtype=dtype).reshape(2, 3, 4) * (r + 1)
                        expected_parts.append(part)

                    expected = torch.cat(expected_parts, dim=dim)

                    # Check values
                    torch.testing.assert_close(result, expected)

    def test_all_reduce_non_contiguous(self):
        """Test all_reduce with non-contiguous tensors"""
        dtypes = [torch.float32, torch.float64, torch.int8, torch.int16, 
                  torch.int32, torch.int64, torch.float16, torch.bfloat16]
        
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                # Test with strided tensor (every other element)
                tensor = torch.arange(0, 10, dtype=dtype).expand(3, 10)[::2]  # Non-contiguous
                tensor = tensor * (self.rank + 1)  # Rank-specific values
                
                result = torch_mpi_ext.ops.all_reduce(tensor, self.comm_ptr)
                
                expected_sum = self.size * (self.size + 1) // 2
                expected = torch.arange(0, 10, dtype=dtype).expand(3, 10)[::2] * expected_sum
                
                torch.testing.assert_close(result, expected)
                
                # Test with transposed tensor (non-contiguous)
                tensor2 = torch.arange(0, 24, dtype=dtype).reshape(2, 3, 4).transpose(0, 2) * (self.rank + 1)
                result2 = torch_mpi_ext.ops.all_reduce(tensor2, self.comm_ptr)
                
                expected_sum2 = self.size * (self.size + 1) // 2
                expected2 = torch.arange(0, 24, dtype=dtype).reshape(2, 3, 4).transpose(0, 2) * expected_sum2
                
                torch.testing.assert_close(result2, expected2)

    def test_all_gather_into_tensor_non_contiguous(self):
        """Test all_gather_into_tensor with non-contiguous tensors"""
        # Test with strided tensor (every other element)
        tensor = torch.arange(0, 12, dtype=torch.float32).reshape(3, 4)[::2]  # Non-contiguous, shape (1, 4)
        tensor = tensor * (self.rank + 1)  # Rank-specific values
        
        result = torch_mpi_ext.ops.all_gather_into_tensor(tensor, self.comm_ptr, dim=0)
        
        # Expected: concatenated results from all ranks along dim=0
        expected_parts = []
        for r in range(self.size):
            part = torch.arange(0, 12, dtype=torch.float32).reshape(3, 4)[::2] * (r + 1)
            expected_parts.append(part)
        expected = torch.cat(expected_parts, dim=0)
        
        torch.testing.assert_close(result, expected)
        
        # Test with transposed tensor (non-contiguous)
        tensor2 = torch.arange(0, 24, dtype=torch.float32).reshape(2, 3, 4).transpose(1, 2) * (self.rank + 1)
        result2 = torch_mpi_ext.ops.all_gather_into_tensor(tensor2, self.comm_ptr, dim=1)
        
        expected_parts2 = []
        for r in range(self.size):
            part2 = torch.arange(0, 24, dtype=torch.float32).reshape(2, 3, 4).transpose(1, 2) * (r + 1)
            expected_parts2.append(part2)
        expected2 = torch.cat(expected_parts2, dim=1)
        
        torch.testing.assert_close(result2, expected2)

    def test_all_gather_into_tensor_out_different_dtypes_and_dims(self):
        """Test all_gather_into_tensor_out with different data types and dimensions"""
        dims = [-1, -2, -3, 0, 1, 2]
        dtypes = [torch.float32, torch.float64, torch.int8, torch.int16, 
                  torch.int32, torch.int64, torch.float16, torch.bfloat16]

        for dim in dims:
            for dtype in dtypes:
                with self.subTest(dim=dim, dtype=dtype):
                    # Create rank-specific tensor
                    tensor = torch.arange(0, 2 * 3 * 4, dtype=dtype).reshape(2, 3, 4) * (self.rank + 1)

                    # Calculate expected output shape
                    expected_shape = list(tensor.shape)
                    actual_dim = dim
                    if actual_dim < 0:
                        actual_dim += tensor.dim()
                    expected_shape[actual_dim] *= self.size

                    # Create output tensor with correct shape
                    output_tensor = torch.empty(expected_shape, dtype=dtype)

                    # Perform all-gather-into-tensor on specified dim
                    torch_mpi_ext.ops.all_gather_into_tensor_out(output_tensor, tensor, self.comm_ptr, dim=dim)

                    # Verify shape
                    self.assertEqual(output_tensor.shape, torch.Size(expected_shape))

                    # Prepare expected result
                    expected_parts = []
                    for r in range(self.size):
                        part = torch.arange(0, 2 * 3 * 4, dtype=dtype).reshape(2, 3, 4) * (r + 1)
                        expected_parts.append(part)

                    expected = torch.cat(expected_parts, dim=dim)

                    # Check values
                    torch.testing.assert_close(output_tensor, expected)

    def test_all_gather_into_tensor_out_non_contiguous(self):
        """Test all_gather_into_tensor_out with non-contiguous tensors"""
        dtypes = [torch.float32, torch.float64, torch.int8, torch.int16, 
                  torch.int32, torch.int64, torch.float16, torch.bfloat16]
        
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                # Test with strided tensor (every other element) - non-contiguous
                input_tensor = torch.arange(0, 12, dtype=dtype).reshape(3, 4)[::2]  # Shape (1, 4)
                input_tensor = input_tensor * (self.rank + 1)  # Rank-specific values
                
                # Calculate expected output shape
                expected_shape = list(input_tensor.shape)
                expected_shape[0] *= self.size  # Concatenate along dim 0 by default
                output_tensor = torch.empty(expected_shape, dtype=dtype)
                
                # Perform all-gather-into-tensor-out
                torch_mpi_ext.ops.all_gather_into_tensor_out(output_tensor, input_tensor, self.comm_ptr, dim=0)
                
                # Prepare expected result
                expected_parts = []
                for r in range(self.size):
                    part = torch.arange(0, 12, dtype=dtype).reshape(3, 4)[::2] * (r + 1)
                    expected_parts.append(part)
                expected = torch.cat(expected_parts, dim=0)
                
                # Check values
                torch.testing.assert_close(output_tensor, expected)
                
                # Test with transposed tensor (non-contiguous)
                input_tensor2 = torch.arange(0, 24, dtype=dtype).reshape(2, 3, 4).transpose(1, 2) * (self.rank + 1)  # Shape (2, 4, 3)
                
                # Calculate expected output shape for dim=1
                expected_shape2 = list(input_tensor2.shape)
                expected_shape2[1] *= self.size  # Concatenate along dim 1
                output_tensor2 = torch.empty(expected_shape2, dtype=dtype)
                
                # Perform all-gather-into-tensor-out along dim=1
                torch_mpi_ext.ops.all_gather_into_tensor_out(output_tensor2, input_tensor2, self.comm_ptr, dim=1)

                # Prepare expected result
                expected_parts2 = []
                for r in range(self.size):
                    part = torch.arange(0, 24, dtype=dtype).reshape(2, 3, 4).transpose(1, 2) * (r + 1)
                    expected_parts2.append(part)
                expected2 = torch.cat(expected_parts2, dim=1)
                
                # Check values
                torch.testing.assert_close(output_tensor2, expected2)

@unittest.skipIf(not MPI_AVAILABLE, "mpi4py not available")
@unittest.skipIf(not EXTENSION_AVAILABLE, "torch_mpi_ext not available")
class TestMPICompile(TestCase):
    """Tests for torch.compile compatibility with MPI operations"""
    
    def setUp(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.comm_ptr = self.comm.py2f()
    
    def test_compile_all_reduce(self):
        """Test all_reduce with torch.compile"""
        
        def model(x, comm_ptr):
            # Simple operations before the MPI operation
            x = x + 1.0
            x = x * 2.0
            
            # MPI operation
            result = torch_mpi_ext.ops.all_reduce(x, comm_ptr)
            
            # Simple operations after the MPI operation
            result = result - 1.0
            result = result / self.size  # Expected average after allreduce
            
            return result
        
        for size in [1000, 5000]:
            with self.subTest(size=size):
                torch.manual_seed(42)
                x = torch.randn((size,), dtype=torch.float32) * (self.rank + 1)
                
                with torch.no_grad():
                    expected = model(x, self.comm_ptr)
                    compiled_model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
                    actual = compiled_model(x, self.comm_ptr)
                
                self.assertEqual(actual, expected, rtol=1e-3, atol=1e-3)

    def test_compile_all_reduce_inplace(self):
        """Test all_reduce_ with torch.compile"""
        
        def model(x, comm_ptr):
            # Store original values to verify inplace operation
            original_shape = x.shape
            original_device = x.device
            
            # Simple operations before the MPI operation
            x = x + 1.0
            x = x * 2.0
            
            # MPI inplace operation
            torch_mpi_ext.ops.all_reduce_(x, comm_ptr)

            # Simple operations after the MPI operation
            x = x - 1.0
            x = x / self.size  # Expected average after allreduce
            
            return x
        
        for size in [1000, 5000]:
            with self.subTest(size=size):
                torch.manual_seed(42)
                x = torch.randn((size,), dtype=torch.float32) * (self.rank + 1)
                
                with torch.no_grad():
                    # Clone x for expected and actual to avoid in-place modifications affecting both
                    x_expected = x.clone()
                    x_actual = x.clone()
                    
                    expected = model(x_expected, self.comm_ptr)
                    # compiled_model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
                    compiled_model = torch.compile(model, fullgraph=True)
                    actual = compiled_model(x_actual, self.comm_ptr)
                
                self.assertEqual(actual, expected, rtol=1e-3, atol=1e-3)

    def test_compile_all_gather_into_tensor(self):
        """Test all_gather_into_tensor with torch.compile"""
        
        def model(input_tensor, output_tensor, comm_ptr, dim):
            # Simple operations before the MPI operation
            input_tensor = input_tensor + 1.0
            input_tensor = input_tensor * 2.0
            
            # MPI operation - all_gather_into_tensor modifies output_tensor in-place
            torch_mpi_ext.ops.all_gather_into_tensor_out(output_tensor, input_tensor, comm_ptr, dim=dim)
            
            # Simple operations after the MPI operation
            output_tensor = output_tensor - 1.0
            output_tensor = output_tensor / 2.0
            
            return output_tensor
        
        for size in [1000, 2000]:  # Using smaller sizes due to memory requirements for all_gather
            with self.subTest(size=size):
                torch.manual_seed(42)
                
                # Create input tensor
                input_tensor = torch.randn((size,), dtype=torch.float32) * (self.rank + 1)
                
                # Create output tensor with appropriate size (will be size * world_size)
                output_shape = list(input_tensor.shape)
                output_shape[0] *= self.size
                output_tensor = torch.empty(output_shape, dtype=torch.float32)
                
                # Clone tensors for expected and actual
                input_expected = input_tensor.clone()
                input_actual = input_tensor.clone()
                output_expected = torch.empty(output_shape, dtype=torch.float32)
                output_actual = torch.empty(output_shape, dtype=torch.float32)
                
                with torch.no_grad():
                    expected = model(input_expected, output_expected, self.comm_ptr, dim=0)
                    compiled_model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
                    actual = compiled_model(input_actual, output_actual, self.comm_ptr, dim=0)
                
                self.assertEqual(actual, expected, rtol=1e-3, atol=1e-3)

if __name__ == "__main__":
    # Only run tests if we're in an MPI environment
    try:
        from mpi4py import MPI
        unittest.main()
    except ImportError:
        print("Skipping MPI tests as mpi4py is not available")
        sys.exit(0)