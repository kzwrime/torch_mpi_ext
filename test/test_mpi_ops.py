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


def _synchronize_accelerator_if_needed():
    if hasattr(torch, "accelerator") and hasattr(torch.accelerator, "synchronize"):
        torch.accelerator.synchronize()


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
                _synchronize_accelerator_if_needed()
                
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
                _synchronize_accelerator_if_needed()

                expected_sum = self.size * (self.size + 1) / 2
                if dtype in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
                    expected = torch.ones(3, dtype=dtype) * expected_sum
                else:
                    expected = torch.ones(3, dtype=dtype) * int(expected_sum)
                    
                torch.testing.assert_close(tensor, expected)

    def test_all_reduce_wrappers_require_cpu_comm_ptr(self):
        """Test all_reduce wrappers reject device comm_ptr_wrapper tensors"""
        comm_ptr_wrapper = torch.tensor([self.comm_ptr], dtype=torch.int64)
        tensor = torch.ones(3, dtype=torch.float32) * (self.rank + 1)

        result = torch_mpi_ext.ops.all_reduce_wrapper(tensor, comm_ptr_wrapper)
        _synchronize_accelerator_if_needed()

        expected_sum = self.size * (self.size + 1) / 2
        expected = torch.ones(3, dtype=torch.float32) * expected_sum
        torch.testing.assert_close(result, expected)

        tensor_inplace = torch.ones(3, dtype=torch.float32) * (self.rank + 1)
        torch_mpi_ext.ops.all_reduce__wrapper(tensor_inplace, comm_ptr_wrapper)
        _synchronize_accelerator_if_needed()
        torch.testing.assert_close(tensor_inplace, expected)

        try:
            device_comm_ptr_wrapper = comm_ptr_wrapper.to("mcpu")
        except Exception as exc:
            self.skipTest(f"mcpu device is not available: {exc}")

        with self.assertRaisesRegex(RuntimeError, "comm_ptr_wrapper must be a CPU tensor"):
            torch_mpi_ext.ops.all_reduce_wrapper(tensor, device_comm_ptr_wrapper)

        with self.assertRaisesRegex(RuntimeError, "comm_ptr_wrapper must be a CPU tensor"):
            torch_mpi_ext.ops.all_reduce__wrapper(tensor_inplace, device_comm_ptr_wrapper)

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
                    _synchronize_accelerator_if_needed()

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
                _synchronize_accelerator_if_needed()
                
                expected_sum = self.size * (self.size + 1) // 2
                expected = torch.arange(0, 10, dtype=dtype).expand(3, 10)[::2] * expected_sum
                
                torch.testing.assert_close(result, expected)
                
                # Test with transposed tensor (non-contiguous)
                tensor2 = torch.arange(0, 24, dtype=dtype).reshape(2, 3, 4).transpose(0, 2) * (self.rank + 1)
                result2 = torch_mpi_ext.ops.all_reduce(tensor2, self.comm_ptr)
                _synchronize_accelerator_if_needed()
                
                expected_sum2 = self.size * (self.size + 1) // 2
                expected2 = torch.arange(0, 24, dtype=dtype).reshape(2, 3, 4).transpose(0, 2) * expected_sum2
                
                torch.testing.assert_close(result2, expected2)

    def test_all_gather_into_tensor_non_contiguous(self):
        """Test all_gather_into_tensor with non-contiguous tensors"""
        # Test with strided tensor (every other element)
        tensor = torch.arange(0, 12, dtype=torch.float32).reshape(3, 4)[::2]  # Non-contiguous, shape (1, 4)
        tensor = tensor * (self.rank + 1)  # Rank-specific values
        
        result = torch_mpi_ext.ops.all_gather_into_tensor(tensor, self.comm_ptr, dim=0)
        _synchronize_accelerator_if_needed()
        
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
        _synchronize_accelerator_if_needed()
        
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
                    _synchronize_accelerator_if_needed()

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
                _synchronize_accelerator_if_needed()
                
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
                _synchronize_accelerator_if_needed()

                # Prepare expected result
                expected_parts2 = []
                for r in range(self.size):
                    part = torch.arange(0, 24, dtype=dtype).reshape(2, 3, 4).transpose(1, 2) * (r + 1)
                    expected_parts2.append(part)
                expected2 = torch.cat(expected_parts2, dim=1)
                
                # Check values
                torch.testing.assert_close(output_tensor2, expected2)

    def test_all_gather_wrappers_require_cpu_comm_ptr(self):
        """Test all_gather wrappers reject device comm_ptr_wrapper tensors"""
        comm_ptr_wrapper = torch.tensor([self.comm_ptr], dtype=torch.int64)
        tensor = torch.arange(0, 6, dtype=torch.float32).reshape(2, 3) * (self.rank + 1)

        result = torch_mpi_ext.ops.all_gather_into_tensor_wrapper(
            tensor, comm_ptr_wrapper, dim=0)
        _synchronize_accelerator_if_needed()

        expected_parts = []
        for r in range(self.size):
            expected_parts.append(
                torch.arange(0, 6, dtype=torch.float32).reshape(2, 3) * (r + 1))
        expected = torch.cat(expected_parts, dim=0)
        torch.testing.assert_close(result, expected)

        output = torch.empty_like(result)
        torch_mpi_ext.ops.all_gather_into_tensor_out_wrapper(
            output, tensor, comm_ptr_wrapper, dim=0)
        _synchronize_accelerator_if_needed()
        torch.testing.assert_close(output, expected)

        try:
            device_comm_ptr_wrapper = comm_ptr_wrapper.to("mcpu")
        except Exception as exc:
            self.skipTest(f"mcpu device is not available: {exc}")

        with self.assertRaisesRegex(RuntimeError, "comm_ptr_wrapper must be a CPU tensor"):
            torch_mpi_ext.ops.all_gather_into_tensor_wrapper(
                tensor, device_comm_ptr_wrapper, dim=0)

        with self.assertRaisesRegex(RuntimeError, "comm_ptr_wrapper must be a CPU tensor"):
            torch_mpi_ext.ops.all_gather_into_tensor_out_wrapper(
                output, tensor, device_comm_ptr_wrapper, dim=0)

    def test_reduce_scatter_different_dims_and_wrappers(self):
        comm_ptr_wrapper = torch.tensor([self.comm_ptr], dtype=torch.int64)

        for dim in range(3):
            with self.subTest(dim=dim):
                shape = [2, 3, 4]
                shape[dim] = 2 * self.size
                tensor = (
                    torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32)
                    .reshape(shape)
                    .add_(self.rank * 1000)
                )
                reduced = sum(
                    torch.arange(
                        torch.tensor(shape).prod().item(), dtype=torch.float32
                    ).reshape(shape).add_(rank * 1000)
                    for rank in range(self.size)
                )
                expected = reduced.narrow(dim, self.rank * 2, 2)

                actual = torch_mpi_ext.ops.reduce_scatter(
                    tensor, self.comm_ptr, dim
                )
                torch.testing.assert_close(actual, expected)

                output = torch.empty_like(expected)
                torch_mpi_ext.ops.reduce_scatter_out_wrapper(
                    output, tensor, comm_ptr_wrapper, dim
                )
                torch.testing.assert_close(output, expected)

    def test_reduce_scatterv_different_dims_dtypes_and_wrappers(self):
        sizes = [rank + 1 for rank in range(self.size)]
        sizes_tensor = torch.tensor(sizes, dtype=torch.int64)
        comm_ptr_wrapper = torch.tensor([self.comm_ptr], dtype=torch.int64)

        for dim in range(3):
            for dtype in (torch.float32, torch.bfloat16, torch.int64):
                with self.subTest(dim=dim, dtype=dtype):
                    shape = [2, 3, 4]
                    shape[dim] = sum(sizes)
                    tensor = (
                        torch.arange(torch.tensor(shape).prod().item(), dtype=dtype)
                        .reshape(shape)
                        .add_(self.rank * 100)
                    )
                    reduced = sum(
                        torch.arange(torch.tensor(shape).prod().item(), dtype=dtype)
                        .reshape(shape)
                        .add_(rank * 100)
                        for rank in range(self.size)
                    )
                    expected = reduced.narrow(
                        dim, sum(sizes[: self.rank]), sizes[self.rank]
                    )

                    actual = torch_mpi_ext.ops.reduce_scatterv(
                        tensor, sizes_tensor, self.comm_ptr, dim
                    )
                    torch.testing.assert_close(actual, expected)

                    output = torch.empty_like(expected)
                    torch_mpi_ext.ops.reduce_scatterv_out_wrapper(
                        output,
                        tensor,
                        sizes_tensor,
                        comm_ptr_wrapper,
                        dim,
                    )
                    torch.testing.assert_close(output, expected)

    def test_zero_sized_reduce_scatterv(self):
        sizes = list(range(self.size))
        sizes_tensor = torch.tensor(sizes, dtype=torch.int64)

        reduce_input = torch.arange(
            2 * sum(sizes) * 3, dtype=torch.float32
        ).reshape(2, sum(sizes), 3)
        scattered = torch_mpi_ext.ops.reduce_scatterv(
            reduce_input, sizes_tensor, self.comm_ptr, dim=1
        )
        expected = (reduce_input * self.size).narrow(
            1, sum(sizes[: self.rank]), sizes[self.rank]
        )
        torch.testing.assert_close(scattered, expected)

    def test_mcpu_variable_collectives(self):
        try:
            import torch_mcpu  # noqa: F401

            torch.empty(1).to("mcpu")
        except (ImportError, RuntimeError) as exc:
            self.skipTest(f"mcpu device is not available: {exc}")

        sizes = [rank + 1 for rank in range(self.size)]
        sizes_tensor = torch.tensor(sizes, dtype=torch.int64)
        comm_ptr_wrapper = torch.tensor([self.comm_ptr], dtype=torch.int64)

        total_size = sum(sizes)
        reduce_input = (
            torch.arange(2 * total_size * 3, dtype=torch.bfloat16)
            .reshape(2, total_size, 3)
            .add_(self.rank * 100)
        )
        reduced = sum(
            torch.arange(2 * total_size * 3, dtype=torch.bfloat16)
            .reshape(2, total_size, 3)
            .add_(rank * 100)
            for rank in range(self.size)
        )
        expected_scattered = reduced.narrow(
            1, sum(sizes[: self.rank]), sizes[self.rank]
        )
        scattered = torch.empty_like(expected_scattered, device="mcpu")
        torch_mpi_ext.ops.reduce_scatterv_out_wrapper(
            scattered,
            reduce_input.to("mcpu"),
            sizes_tensor,
            comm_ptr_wrapper,
            dim=1,
        )
        _synchronize_accelerator_if_needed()
        torch.testing.assert_close(scattered.cpu(), expected_scattered)

    def test_alltoallv_different_dtypes(self):
        """Test alltoallv with different data types"""
        dtypes = [torch.float32, torch.float64, torch.int8, torch.int16,
                  torch.int32, torch.int64, torch.float16, torch.bfloat16]

        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                # Each rank sends different amount of data to each other rank
                # For simplicity, each rank i sends (i+1) elements to each rank j
                sendcounts = torch.tensor([(self.rank + 1) for _ in range(self.size)], dtype=torch.int32)
                recvcounts = torch.tensor([(r + 1) for r in range(self.size)], dtype=torch.int32)

                # Calculate displacements
                sdispls = torch.tensor([sum(sendcounts[:j].tolist()) for j in range(self.size)], dtype=torch.int32)
                rdispls = torch.tensor([sum(recvcounts[:j].tolist()) for j in range(self.size)], dtype=torch.int32)

                # Total elements to send
                total_send = int(sendcounts.sum().item())
                total_recv = int(recvcounts.sum().item())

                # Create send buffer with rank-specific values
                sendbuf = torch.arange(total_send, dtype=dtype) * (self.rank + 1)

                # Perform alltoallv
                result = torch_mpi_ext.ops.alltoallv(sendbuf, sendcounts, sdispls, recvcounts, rdispls, self.comm_ptr)

                # Verify shape
                self.assertEqual(result.shape, torch.Size([total_recv]))

                # Verify values - each rank should receive data from all ranks
                # Data from rank r at position corresponding to rank r
                expected = torch.empty(total_recv, dtype=dtype)
                for rank in range(self.size):
                    # Re-create sendbuf like rank-i sends
                    sendcounts = torch.tensor([(rank + 1) for _ in range(self.size)], dtype=torch.int32)
                    sdispls = torch.tensor([sum(sendcounts[:j].tolist()) for j in range(self.size)], dtype=torch.int32)
                    total_send = int(sendcounts.sum().item())
                    sendbuf = torch.arange(total_send, dtype=dtype) * (rank + 1)

                    expected[rdispls[rank]:rdispls[rank]+recvcounts[rank]] = sendbuf[sdispls[self.rank]:sdispls[self.rank]+sendcounts[self.rank]]

                torch.testing.assert_close(result, expected)


    def test_alltoallv_out_different_dtypes(self):
        """Test alltoallv_out with different data types"""
        dtypes = [torch.float32, torch.float64, torch.int8, torch.int16,
                  torch.int32, torch.int64, torch.float16, torch.bfloat16]

        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                # Each rank i sends different amount of data to each other rank
                sendcounts = torch.tensor([(self.rank + 1) for _ in range(self.size)], dtype=torch.int32)
                recvcounts = torch.tensor([(r + 1) for r in range(self.size)], dtype=torch.int32)

                # Calculate displacements
                sdispls = torch.tensor([sum(sendcounts[:j].tolist()) for j in range(self.size)], dtype=torch.int32)
                rdispls = torch.tensor([sum(recvcounts[:j].tolist()) for j in range(self.size)], dtype=torch.int32)

                # Total elements
                total_send = int(sendcounts.sum().item())
                total_recv = int(recvcounts.sum().item())

                # Create send buffer
                sendbuf = torch.arange(total_send, dtype=dtype) * (self.rank + 1)

                # Create receive buffer
                recvbuf = torch.empty(total_recv, dtype=dtype)

                # Perform alltoallv_out
                result = torch_mpi_ext.ops.alltoallv_out(recvbuf, sendbuf, sendcounts, sdispls, recvcounts, rdispls, self.comm_ptr)

                # Verify return value (in-place returns None)
                self.assertTrue(result == None)

                # print(f"[{self.rank}] sendbuf: {sendbuf}, sendcounts {sendcounts}, sdispls {sdispls}")
                # print(f"[{self.rank}] recvbuf: {recvbuf}, recvcounts {recvcounts}, rdispls {rdispls}")

                # Verify values
                expected = torch.empty(total_recv, dtype=dtype)
                for rank in range(self.size):

                    # Re-create sendbuf like rank-i sends
                    sendcounts = torch.tensor([(rank + 1) for _ in range(self.size)], dtype=torch.int32)
                    sdispls = torch.tensor([sum(sendcounts[:j].tolist()) for j in range(self.size)], dtype=torch.int32)
                    total_send = int(sendcounts.sum().item())
                    sendbuf = torch.arange(total_send, dtype=dtype) * (rank + 1)
    
                    # print(f"[{self.rank}][{rank}] sendbuf: {sendbuf}, sendcounts {sendcounts}, sdispls {sdispls}, ")
                    expected[rdispls[rank]:rdispls[rank]+recvcounts[rank]] = sendbuf[sdispls[self.rank]:sdispls[self.rank]+sendcounts[self.rank]]

                torch.testing.assert_close(recvbuf, expected)

    def test_alltoallv_non_contiguous(self):
        """Test alltoallv with non-contiguous tensors"""
        dtypes = [torch.float32, torch.float64, torch.int8, torch.int16,
                  torch.int32, torch.int64, torch.float16, torch.bfloat16]

        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                # Setup counts and displacements
                sendcounts = torch.tensor([(self.rank + 1) for _ in range(self.size)], dtype=torch.int32)
                recvcounts = torch.tensor([(r + 1) for r in range(self.size)], dtype=torch.int32)

                sdispls = torch.tensor([sum(sendcounts[:j].tolist()) for j in range(self.size)], dtype=torch.int32)
                rdispls = torch.tensor([sum(recvcounts[:j].tolist()) for j in range(self.size)], dtype=torch.int32)

                total_send = int(sendcounts.sum().item())

                # Create non-contiguous send buffer (strided)
                base_tensor = torch.arange(0, total_send * 2, dtype=dtype)
                sendbuf = base_tensor[::2] * (self.rank + 1)  # Non-contiguous

                # Perform alltoallv (should handle non-contiguous input)
                result = torch_mpi_ext.ops.alltoallv(sendbuf, sendcounts, sdispls, recvcounts, rdispls, self.comm_ptr)

                # Verify
                total_recv = int(recvcounts.sum().item())
                self.assertEqual(result.shape, torch.Size([total_recv]))

                # Verify values - reconstruct expected from each rank's perspective
                expected = torch.empty(total_recv, dtype=dtype)
                for rank in range(self.size):
                    # Re-create sendbuf like rank-i sends (with non-contiguous pattern)
                    sendcounts = torch.tensor([(rank + 1) for _ in range(self.size)], dtype=torch.int32)
                    sdispls = torch.tensor([sum(sendcounts[:j].tolist()) for j in range(self.size)], dtype=torch.int32)
                    total_send = int(sendcounts.sum().item())
                    base = torch.arange(0, total_send * 2, dtype=dtype)[::2]
                    sendbuf = base[:total_send] * (rank + 1)

                    expected[rdispls[rank]:rdispls[rank]+recvcounts[rank]] = sendbuf[sdispls[self.rank]:sdispls[self.rank]+sendcounts[self.rank]]

                torch.testing.assert_close(result, expected)

    def test_alltoallv_2d_matrix_rows(self):
        """Test alltoallv with 2D matrix - each rank sends different number of rows to each other rank"""
        dtypes = [torch.float32, torch.float64, torch.int32, torch.int64]

        # Test with different matrix sizes
        for num_cols in [4, 8]:
            for dtype in dtypes:
                with self.subTest(dtype=dtype, num_cols=num_cols):
                    # Each rank i sends (i+1) rows to each rank j
                    rows_per_send = self.rank + 1  # Number of rows this rank sends to each other rank
                    total_send_rows = rows_per_send * self.size

                    # Calculate send counts (in elements, not rows)
                    sendcounts = torch.tensor([rows_per_send * num_cols for _ in range(self.size)], dtype=torch.int32)

                    # Calculate receive counts (rank j sends (j+1) rows)
                    recvcounts = torch.tensor([(r + 1) * num_cols for r in range(self.size)], dtype=torch.int32)

                    # Calculate displacements (in elements)
                    sdispls = torch.tensor([sum(sendcounts[:j].tolist()) for j in range(self.size)], dtype=torch.int32)
                    rdispls = torch.tensor([sum(recvcounts[:j].tolist()) for j in range(self.size)], dtype=torch.int32)

                    # Total elements
                    total_send = int(sendcounts.sum().item())
                    total_recv = int(recvcounts.sum().item())
                    total_recv_rows = total_recv // num_cols

                    # Create send buffer (flattened 2D matrix)
                    # Each row contains unique values based on rank
                    sendbuf = torch.zeros(total_send, dtype=dtype)
                    for i in range(total_send_rows):
                        row_start = i * num_cols
                        sendbuf[row_start:row_start + num_cols] = torch.arange(num_cols, dtype=dtype) + i * 100 + (self.rank + 1) * 1000

                    # Perform alltoallv
                    result = torch_mpi_ext.ops.alltoallv(sendbuf, sendcounts, sdispls, recvcounts, rdispls, self.comm_ptr)

                    # Verify shape
                    self.assertEqual(result.shape, torch.Size([total_recv]))

                    # Verify values - reconstruct expected from each rank's perspective
                    expected = torch.zeros(total_recv, dtype=dtype)
                    for rank in range(self.size):
                        # Re-create sendbuf like rank-i sends
                        rows_per_send_from_rank = rank + 1
                        total_send_rows_from_rank = rows_per_send_from_rank * self.size
                        sendcounts_from_rank = torch.tensor([rows_per_send_from_rank * num_cols for _ in range(self.size)], dtype=torch.int32)
                        sdispls_from_rank = torch.tensor([sum(sendcounts_from_rank[:j].tolist()) for j in range(self.size)], dtype=torch.int32)
                        total_send_from_rank = int(sendcounts_from_rank.sum().item())

                        sendbuf_from_rank = torch.zeros(total_send_from_rank, dtype=dtype)
                        for i in range(total_send_rows_from_rank):
                            row_start = i * num_cols
                            sendbuf_from_rank[row_start:row_start + num_cols] = torch.arange(num_cols, dtype=dtype) + i * 100 + (rank + 1) * 1000

                        # Extract data that rank sends to self.rank
                        send_start = sdispls_from_rank[self.rank]
                        send_end = send_start + sendcounts_from_rank[self.rank]
                        recv_start = rdispls[rank]
                        recv_end = recv_start + recvcounts[rank]

                        expected[recv_start:recv_end] = sendbuf_from_rank[send_start:send_end]

                    torch.testing.assert_close(result, expected)

    def test_alltoall_different_dtypes(self):
        """Test alltoall with different data types"""
        dtypes = [torch.float32, torch.float64, torch.int8, torch.int16,
                  torch.int32, torch.int64, torch.float16, torch.bfloat16]

        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                # Each rank sends count elements to each rank
                count = 10  # Elements per rank
                total_elements = count * self.size

                # Create send buffer with rank-specific values
                sendbuf = torch.arange(total_elements, dtype=dtype) * (self.rank + 1)

                # Perform alltoall
                result = torch_mpi_ext.ops.alltoall(sendbuf, self.comm_ptr)

                # Verify shape
                self.assertEqual(result.shape, torch.Size([total_elements]))

                # Verify values - each rank should receive chunk i from rank i
                expected = torch.empty(total_elements, dtype=dtype)
                for rank in range(self.size):
                    # Rank rank sends: [0, 1, ..., total_elements-1] * (rank + 1)
                    # We receive the chunk that rank rank sent to us (rank self.rank)
                    # Chunk for rank self.rank is [self.rank * count : (self.rank + 1) * count]
                    chunk_start = self.rank * count
                    chunk_end = (self.rank + 1) * count
                    recv_pos = rank * count

                    sendbuf_from_rank = torch.arange(total_elements, dtype=dtype) * (rank + 1)
                    expected[recv_pos:recv_pos + count] = sendbuf_from_rank[chunk_start:chunk_end]

                torch.testing.assert_close(result, expected)

    def test_alltoall_out_different_dtypes(self):
        """Test alltoall_out (pre-allocated output) with different data types"""
        dtypes = [torch.float32, torch.float64, torch.int8, torch.int16,
                  torch.int32, torch.int64, torch.float16, torch.bfloat16]

        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                # Each rank sends count elements to each rank
                count = 10  # Elements per rank
                total_elements = count * self.size

                # Create send buffer with rank-specific values
                sendbuf = torch.arange(total_elements, dtype=dtype) * (self.rank + 1)

                # Create receive buffer
                recvbuf = torch.empty(total_elements, dtype=dtype)

                # Perform alltoall_out
                result = torch_mpi_ext.ops.alltoall_out(recvbuf, sendbuf, self.comm_ptr)

                # Verify return value (in-place returns None)
                self.assertTrue(result == None)

                # Verify values
                expected = torch.empty(total_elements, dtype=dtype)
                for rank in range(self.size):
                    chunk_start = self.rank * count
                    chunk_end = (self.rank + 1) * count
                    recv_pos = rank * count

                    sendbuf_from_rank = torch.arange(total_elements, dtype=dtype) * (rank + 1)
                    expected[recv_pos:recv_pos + count] = sendbuf_from_rank[chunk_start:chunk_end]

                torch.testing.assert_close(recvbuf, expected)

    def test_alltoall_2d_matrix_rows(self):
        """Test alltoall with 2D matrix - each rank sends equal rows to all other ranks"""
        dtypes = [torch.float32, torch.float64, torch.int32, torch.int64]

        # Test with different matrix sizes
        for num_cols in [4, 8]:
            for dtype in dtypes:
                with self.subTest(dtype=dtype, num_cols=num_cols):
                    # Each rank sends same number of rows to each rank
                    rows_per_rank = 3
                    total_send_rows = rows_per_rank * self.size
                    total_elements = total_send_rows * num_cols

                    # Create send buffer (flattened 2D matrix)
                    sendbuf = torch.empty(total_elements, dtype=dtype)
                    for i in range(total_send_rows):
                        row_start = i * num_cols
                        sendbuf[row_start:row_start + num_cols] = torch.arange(num_cols, dtype=dtype) + i * 100 + (self.rank + 1) * 1000

                    # Perform alltoall
                    result = torch_mpi_ext.ops.alltoall(sendbuf, self.comm_ptr)

                    # Verify shape
                    self.assertEqual(result.shape, torch.Size([total_elements]))

                    # Verify values
                    expected = torch.empty(total_elements, dtype=dtype)
                    for rank in range(self.size):
                        # For each rank, reconstruct the chunk they sent to self.rank
                        sendbuf_from_rank = torch.empty(total_elements, dtype=dtype)
                        for i in range(total_send_rows):
                            row_start = i * num_cols
                            sendbuf_from_rank[row_start:row_start + num_cols] = torch.arange(num_cols, dtype=dtype) + i * 100 + (rank + 1) * 1000

                        # Extract chunk that rank sent to self.rank
                        chunk_start = self.rank * rows_per_rank * num_cols
                        chunk_end = (self.rank + 1) * rows_per_rank * num_cols
                        recv_pos = rank * rows_per_rank * num_cols

                        expected[recv_pos:recv_pos + rows_per_rank * num_cols] = sendbuf_from_rank[chunk_start:chunk_end]

                    torch.testing.assert_close(result, expected)

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
                    _synchronize_accelerator_if_needed()
                    compiled_model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
                    actual = compiled_model(x, self.comm_ptr)
                    _synchronize_accelerator_if_needed()
                
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
                    _synchronize_accelerator_if_needed()
                    # compiled_model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
                    compiled_model = torch.compile(model, fullgraph=True)
                    actual = compiled_model(x_actual, self.comm_ptr)
                    _synchronize_accelerator_if_needed()
                
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
                    _synchronize_accelerator_if_needed()
                    compiled_model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
                    actual = compiled_model(input_actual, output_actual, self.comm_ptr, dim=0)
                    _synchronize_accelerator_if_needed()
                
                self.assertEqual(actual, expected, rtol=1e-3, atol=1e-3)

    def test_compile_reduce_scatterv_out_wrapper(self):
        sizes = torch.arange(1, self.size + 1, dtype=torch.int64)
        comm_ptr_wrapper = torch.tensor([self.comm_ptr], dtype=torch.int64)
        local_size = int(sizes[self.rank])

        def reduce_scatterv_model(output, input_):
            torch_mpi_ext.ops.reduce_scatterv_out_wrapper(
                output, input_, sizes, comm_ptr_wrapper, dim=1
            )
            return output + 1

        scatter_input = torch.arange(
            2 * int(sizes.sum()) * 3, dtype=torch.float32
        ).reshape(2, int(sizes.sum()), 3).add_(self.rank * 100)
        scatter_shape = (2, local_size, 3)
        expected_scatter = reduce_scatterv_model(
            torch.empty(scatter_shape), scatter_input
        )
        compiled_reduce_scatterv = torch.compile(
            reduce_scatterv_model, fullgraph=True
        )
        actual_scatter = compiled_reduce_scatterv(
            torch.empty(scatter_shape), scatter_input
        )
        torch.testing.assert_close(actual_scatter, expected_scatter)

    # def test_compile_alltoallv(self):
    #     """Test alltoallv with torch.compile"""

    #     def model(sendbuf, sendcounts, sdispls, recvcounts, rdispls, comm_ptr):
    #         # Simple operations before the MPI operation
    #         sendbuf = sendbuf + 1.0
    #         sendbuf = sendbuf * 2.0

    #         # MPI operation
    #         result = torch_mpi_ext.ops.alltoallv(sendbuf, sendcounts, sdispls, recvcounts, rdispls, comm_ptr)

    #         # Simple operations after the MPI operation
    #         result = result - 1.0
    #         result = result / 2.0

    #         return result

    #     # Test with different data sizes
    #     for data_size in [100, 500]:
    #         with self.subTest(data_size=data_size):
    #             torch.manual_seed(42)

    #             # Each rank sends different amount of data to each other rank
    #             sendcounts = torch.tensor([(self.rank + 1) * data_size // self.size for _ in range(self.size)], dtype=torch.int32)
    #             recvcounts = torch.tensor([(r + 1) * data_size // self.size for r in range(self.size)], dtype=torch.int32)

    #             sdispls = torch.tensor([sum(sendcounts[:j].tolist()) for j in range(self.size)], dtype=torch.int32)
    #             rdispls = torch.tensor([sum(recvcounts[:j].tolist()) for j in range(self.size)], dtype=torch.int32)

    #             total_send = int(sendcounts.sum().item())
    #             total_recv = int(recvcounts.sum().item())

    #             sendbuf = torch.randn(total_send, dtype=torch.float32) * (self.rank + 1)

    #             with torch.no_grad():
    #                 expected = model(sendbuf, sendcounts, sdispls, recvcounts, rdispls, self.comm_ptr)
    #                 compiled_model = torch.compile(model, fullgraph=True)
    #                 actual = compiled_model(sendbuf, sendcounts, sdispls, recvcounts, rdispls, self.comm_ptr)

    #             self.assertEqual(actual, expected, rtol=1e-3, atol=1e-3)

    # def test_compile_alltoallv_out(self):
    #     """Test alltoallv_out (in-place) with torch.compile"""

    #     def model(recvbuf, sendbuf, sendcounts, sdispls, recvcounts, rdispls, comm_ptr):
    #         # Simple operations before the MPI operation
    #         sendbuf = sendbuf + 1.0
    #         sendbuf = sendbuf * 2.0

    #         # MPI inplace operation
    #         torch_mpi_ext.ops.alltoallv_out(recvbuf, sendbuf, sendcounts, sdispls, recvcounts, rdispls, comm_ptr)

    #         # Simple operations after the MPI operation
    #         recvbuf = recvbuf - 1.0
    #         recvbuf = recvbuf / 2.0

    #         return recvbuf

    #     # Test with different data sizes
    #     for data_size in [100, 500]:
    #         with self.subTest(data_size=data_size):
    #             torch.manual_seed(42)

    #             # Each rank sends different amount of data to each other rank
    #             sendcounts = torch.tensor([(self.rank + 1) * data_size // self.size for _ in range(self.size)], dtype=torch.int32)
    #             recvcounts = torch.tensor([(r + 1) * data_size // self.size for r in range(self.size)], dtype=torch.int32)

    #             sdispls = torch.tensor([sum(sendcounts[:j].tolist()) for j in range(self.size)], dtype=torch.int32)
    #             rdispls = torch.tensor([sum(recvcounts[:j].tolist()) for j in range(self.size)], dtype=torch.int32)

    #             total_send = int(sendcounts.sum().item())
    #             total_recv = int(recvcounts.sum().item())

    #             sendbuf = torch.randn(total_send, dtype=torch.float32) * (self.rank + 1)
    #             recvbuf = torch.empty(total_recv, dtype=torch.float32)

    #             with torch.no_grad():
    #                 # Clone tensors for expected and actual
    #                 sendbuf_expected = sendbuf.clone()
    #                 sendbuf_actual = sendbuf.clone()
    #                 recvbuf_expected = torch.empty(total_recv, dtype=torch.float32)
    #                 recvbuf_actual = torch.empty(total_recv, dtype=torch.float32)

    #                 expected = model(recvbuf_expected, sendbuf_expected, sendcounts, sdispls, recvcounts, rdispls, self.comm_ptr)
    #                 compiled_model = torch.compile(model, fullgraph=True)
    #                 actual = compiled_model(recvbuf_actual, sendbuf_actual, sendcounts, sdispls, recvcounts, rdispls, self.comm_ptr)

    #             self.assertEqual(actual, expected, rtol=1e-3, atol=1e-3)

    def test_compile_alltoall(self):
        """Test alltoall with torch.compile"""

        def model(sendbuf, comm_ptr):
            # Simple operations before the MPI operation
            sendbuf = sendbuf + 1.0
            sendbuf = sendbuf * 2.0

            # MPI operation
            result = torch_mpi_ext.ops.alltoall(sendbuf, comm_ptr)

            # Simple operations after the MPI operation
            result = result - 1.0
            result = result / 2.0

            return result

        # Test with different data sizes
        for data_size in [100, 500]:
            with self.subTest(data_size=data_size):
                torch.manual_seed(42)

                # Each rank sends equal amount of data to each rank
                total_elements = data_size * self.size
                sendbuf = torch.randn(total_elements, dtype=torch.float32) * (self.rank + 1)

                with torch.no_grad():
                    expected = model(sendbuf, self.comm_ptr)
                    compiled_model = torch.compile(model, fullgraph=True)
                    actual = compiled_model(sendbuf, self.comm_ptr)

                self.assertEqual(actual, expected, rtol=1e-3, atol=1e-3)

    def test_compile_alltoall_out(self):
        """Test alltoall_out (pre-allocated output) with torch.compile"""

        def model(recvbuf, sendbuf, comm_ptr):
            # Simple operations before the MPI operation
            sendbuf = sendbuf + 1.0
            sendbuf = sendbuf * 2.0

            # MPI inplace operation
            torch_mpi_ext.ops.alltoall_out(recvbuf, sendbuf, comm_ptr)

            # Simple operations after the MPI operation
            recvbuf = recvbuf - 1.0
            recvbuf = recvbuf / 2.0

            return recvbuf

        # Test with different data sizes
        for data_size in [100, 500]:
            with self.subTest(data_size=data_size):
                torch.manual_seed(42)

                # Each rank sends equal amount of data to each rank
                total_elements = data_size * self.size
                sendbuf = torch.randn(total_elements, dtype=torch.float32) * (self.rank + 1)
                recvbuf = torch.empty(total_elements, dtype=torch.float32)

                with torch.no_grad():
                    # Clone tensors for expected and actual
                    sendbuf_expected = sendbuf.clone()
                    sendbuf_actual = sendbuf.clone()
                    recvbuf_expected = torch.empty(total_elements, dtype=torch.float32)
                    recvbuf_actual = torch.empty(total_elements, dtype=torch.float32)

                    expected = model(recvbuf_expected, sendbuf_expected, self.comm_ptr)
                    compiled_model = torch.compile(model, fullgraph=True)
                    actual = compiled_model(recvbuf_actual, sendbuf_actual, self.comm_ptr)

                self.assertEqual(actual, expected, rtol=1e-3, atol=1e-3)

if __name__ == "__main__":
    # Only run tests if we're in an MPI environment
    try:
        from mpi4py import MPI
        unittest.main()
    except ImportError:
        print("Skipping MPI tests as mpi4py is not available")
        sys.exit(status=0)
