import torch
from torch import Tensor

__all__ = [
    "all_gather_into_tensor",
    "all_gather_into_tensor_out",
    "all_gather_into_tensor_out_wrapper",
    "all_gather_into_tensor_wrapper",
    "all_reduce",
    "all_reduce_",
    "all_reduce__wrapper",
    "all_reduce_wrapper",
    "alltoall",
    "alltoall_out",
    "alltoallv",
    "alltoallv_out",
    "myadd_out",
    "mymuladd",
    "reduce_scatter",
    "reduce_scatter_out",
    "reduce_scatter_out_wrapper",
    "reduce_scatter_wrapper",
    "reduce_scatterv",
    "reduce_scatterv_out",
    "reduce_scatterv_out_wrapper",
    "reduce_scatterv_wrapper",
]


def mymuladd(a: Tensor, b: Tensor, c: float) -> Tensor:
    """Performs a * b + c in an efficient fused kernel"""
    return torch.ops.torch_mpi_ext.mymuladd.default(a, b, c)


# Registers a FakeTensor kernel (aka "meta kernel", "abstract impl")
# that describes what the properties of the output Tensor are given
# the properties of the input Tensor. The FakeTensor kernel is necessary
# for the op to work performantly with torch.compile.
@torch.library.register_fake("torch_mpi_ext::mymuladd")
def _(a, b, c):
    torch._check(a.shape == b.shape)
    torch._check(a.dtype == torch.float)
    torch._check(b.dtype == torch.float)
    torch._check(a.device == b.device)
    return torch.empty_like(a)


def _backward(ctx, grad):
    a, b = ctx.saved_tensors
    grad_a, grad_b = None, None
    if ctx.needs_input_grad[0]:
        grad_a = torch.ops.torch_mpi_ext.mymul.default(grad, b)
    if ctx.needs_input_grad[1]:
        grad_b = torch.ops.torch_mpi_ext.mymul.default(grad, a)
    return grad_a, grad_b, None


def _setup_context(ctx, inputs, output):
    a, b, c = inputs
    saved_a, saved_b = None, None
    if ctx.needs_input_grad[0]:
        saved_b = b
    if ctx.needs_input_grad[1]:
        saved_a = a
    ctx.save_for_backward(saved_a, saved_b)


# This adds training support for the operator. You must provide us
# the backward formula for the operator and a `setup_context` function
# to save values to be used in the backward.
torch.library.register_autograd(
    "torch_mpi_ext::mymuladd", _backward, setup_context=_setup_context)


@torch.library.register_fake("torch_mpi_ext::mymul")
def _(a, b):
    torch._check(a.shape == b.shape)
    torch._check(a.dtype == torch.float)
    torch._check(b.dtype == torch.float)
    torch._check(a.device == b.device)
    return torch.empty_like(a)


def myadd_out(a: Tensor, b: Tensor, out: Tensor) -> None:
    """Writes a + b into out"""
    torch.ops.torch_mpi_ext.myadd_out.default(a, b, out)


def all_reduce_(input: Tensor, comm_ptr: int):
    """
    Performs MPI allreduce operation on the input tensor in-place.
    
    Args:
        input: Input tensor to reduce (will be modified in-place)
        comm_ptr: MPI communicator handle obtained from comm.py2f()
        
    Returns:
        Reduced tensor (same tensor as input)
    """
    return torch.ops.torch_mpi_ext.all_reduce_.default(input, comm_ptr)


def all_reduce(input: Tensor, comm_ptr: int) -> Tensor:
    """
    Performs MPI allreduce operation on the input tensor.
    
    Args:
        input: Input tensor to reduce
        comm_ptr: MPI communicator handle obtained from comm.py2f()
        
    Returns:
        Reduced tensor with same shape as input
    """
    return torch.ops.torch_mpi_ext.all_reduce.default(input, comm_ptr)


def all_reduce__wrapper(input: Tensor, comm_ptr_wrapper: Tensor):
    """
    Performs MPI allreduce operation on the input tensor in-place.

    Wrapper function that accepts comm_ptr_wrapper (Tensor) and calls
    the C++ wrapper function which extracts comm_ptr and calls the original function.

    Args:
        input: Input tensor to reduce (will be modified in-place)
        comm_ptr_wrapper: MPI communicator handle wrapped in a tensor [1]

    Returns:
        Reduced tensor (same tensor as input)
    """
    return torch.ops.torch_mpi_ext.all_reduce__wrapper.default(input, comm_ptr_wrapper)


def all_reduce_wrapper(input: Tensor, comm_ptr_wrapper: Tensor) -> Tensor:
    """
    Performs MPI allreduce operation on the input tensor.

    Wrapper function that accepts comm_ptr_wrapper (Tensor) and calls
    the C++ wrapper function which extracts comm_ptr and calls the original function.

    Args:
        input: Input tensor to reduce
        comm_ptr_wrapper: MPI communicator handle wrapped in a tensor [1]

    Returns:
        Reduced tensor with same shape as input
    """
    return torch.ops.torch_mpi_ext.all_reduce_wrapper.default(input, comm_ptr_wrapper)


def all_gather_into_tensor(input: Tensor, comm_ptr: int, dim: int = -1) -> Tensor:
    """
    Performs MPI allgather operation on the input tensor.
    
    Args:
        input: Input tensor to gather
        comm_ptr: MPI communicator handle obtained from comm.py2f()
        dim: Dimension along which to concatenate the gathered tensors
        
    Returns:
        Gathered tensor with specified dimension scaled by world size
    """
    return torch.ops.torch_mpi_ext.all_gather_into_tensor.default(input, comm_ptr, dim)


def all_gather_into_tensor_out(output: Tensor, input: Tensor, comm_ptr: int, dim: int = -1) -> Tensor:
    """
    Performs MPI allgather operation on the input tensor and writes the result directly to the output tensor.
    
    Args:
        output: Output tensor to write the gathered result to
        input: Input tensor to gather
        comm_ptr: MPI communicator handle obtained from comm.py2f()
        dim: Dimension along which to concatenate the gathered tensors
        
    Returns:
        Same output tensor with gathered values written to it
    """
    return torch.ops.torch_mpi_ext.all_gather_into_tensor_out.default(output, input, comm_ptr, dim)


def all_gather_into_tensor_wrapper(input: Tensor, comm_ptr_wrapper: Tensor, dim: int = -1) -> Tensor:
    """
    Performs MPI allgather operation on the input tensor.

    Wrapper function that accepts comm_ptr_wrapper (Tensor) and calls
    the C++ wrapper function which extracts comm_ptr and calls the original function.

    Args:
        input: Input tensor to gather
        comm_ptr_wrapper: MPI communicator handle wrapped in a tensor [1]
        dim: Dimension along which to concatenate the gathered tensors

    Returns:
        Gathered tensor with specified dimension scaled by world size
    """
    return torch.ops.torch_mpi_ext.all_gather_into_tensor_wrapper.default(input, comm_ptr_wrapper, dim)


def all_gather_into_tensor_out_wrapper(output: Tensor, input: Tensor, comm_ptr_wrapper: Tensor, dim: int = -1) -> Tensor:
    """
    Performs MPI allgather operation on the input tensor and writes the result directly to the output tensor.

    Wrapper function that accepts comm_ptr_wrapper (Tensor) and calls
    the C++ wrapper function which extracts comm_ptr and calls the original function.

    Args:
        output: Output tensor to write the gathered result to
        input: Input tensor to gather
        comm_ptr_wrapper: MPI communicator handle wrapped in a tensor [1]
        dim: Dimension along which to concatenate the gathered tensors

    Returns:
        Same output tensor with gathered values written to it
    """
    return torch.ops.torch_mpi_ext.all_gather_into_tensor_out_wrapper.default(output, input, comm_ptr_wrapper, dim)


def reduce_scatter(input: Tensor, comm_ptr: int, dim: int = -1) -> Tensor:
    """Sum-reduce equal chunks and return this rank's chunk along dim."""
    return torch.ops.torch_mpi_ext.reduce_scatter.default(input, comm_ptr, dim)


def reduce_scatter_out(
    output: Tensor, input: Tensor, comm_ptr: int, dim: int = -1
) -> None:
    """Out variant of :func:`reduce_scatter`."""
    torch.ops.torch_mpi_ext.reduce_scatter_out.default(
        output, input, comm_ptr, dim
    )


def reduce_scatter_wrapper(
    input: Tensor, comm_ptr_wrapper: Tensor, dim: int = -1
) -> Tensor:
    """Tensor-communicator wrapper for :func:`reduce_scatter`."""
    return torch.ops.torch_mpi_ext.reduce_scatter_wrapper.default(
        input, comm_ptr_wrapper, dim
    )


def reduce_scatter_out_wrapper(
    output: Tensor, input: Tensor, comm_ptr_wrapper: Tensor, dim: int = -1
) -> None:
    """Tensor-communicator wrapper for :func:`reduce_scatter_out`."""
    torch.ops.torch_mpi_ext.reduce_scatter_out_wrapper.default(
        output, input, comm_ptr_wrapper, dim
    )


def reduce_scatterv(
    input: Tensor, sizes: Tensor, comm_ptr: int, dim: int = -1
) -> Tensor:
    """Sum-reduce variable chunks and return this rank's chunk along dim."""
    return torch.ops.torch_mpi_ext.reduce_scatterv.default(
        input, sizes, comm_ptr, dim
    )


def reduce_scatterv_out(
    output: Tensor,
    input: Tensor,
    sizes: Tensor,
    comm_ptr: int,
    dim: int = -1,
) -> None:
    """Out variant of :func:`reduce_scatterv`."""
    torch.ops.torch_mpi_ext.reduce_scatterv_out.default(
        output, input, sizes, comm_ptr, dim
    )


def reduce_scatterv_wrapper(
    input: Tensor, sizes: Tensor, comm_ptr_wrapper: Tensor, dim: int = -1
) -> Tensor:
    """Tensor-communicator wrapper for :func:`reduce_scatterv`."""
    return torch.ops.torch_mpi_ext.reduce_scatterv_wrapper.default(
        input, sizes, comm_ptr_wrapper, dim
    )


def reduce_scatterv_out_wrapper(
    output: Tensor,
    input: Tensor,
    sizes: Tensor,
    comm_ptr_wrapper: Tensor,
    dim: int = -1,
) -> None:
    """Tensor-communicator wrapper for :func:`reduce_scatterv_out`."""
    torch.ops.torch_mpi_ext.reduce_scatterv_out_wrapper.default(
        output, input, sizes, comm_ptr_wrapper, dim
    )


def alltoall(sendbuf: Tensor, comm_ptr: int) -> Tensor:
    """
    Performs MPI all-to-all operation.

    All-to-all is a collective communication operation where each process
    sends the same amount of data to all other processes and receives data
    from all other processes.

    The send buffer is divided into equal-sized chunks, one for each process.
    Each process sends chunk j to process j, and receives chunk i from process i.

    Args:
        sendbuf: Tensor containing data to send. Size must be divisible by world_size.
        comm_ptr: MPI communicator handle obtained from comm.py2f()

    Returns:
        Tensor containing the received data from all processes
    """
    _alltoall_check(sendbuf, comm_ptr)
    recvbuf = torch.empty_like(sendbuf)
    torch.ops.torch_mpi_ext.alltoall_out.default(recvbuf, sendbuf, comm_ptr)
    return recvbuf


def alltoall_out(recvbuf: Tensor, sendbuf: Tensor, comm_ptr: int) -> None:
    """
    Performs MPI all-to-all operation with pre-allocated output buffer.

    All-to-all is a collective communication operation where each process
    sends the same amount of data to all other processes and receives data
    from all other processes.

    The send buffer is divided into equal-sized chunks, one for each process.
    Each process sends chunk j to process j, and receives chunk i from process i.

    Args:
        recvbuf: Pre-allocated tensor to write the received data to. Must have same size as sendbuf.
        sendbuf: Tensor containing data to send. Size must be divisible by world_size.
        comm_ptr: MPI communicator handle obtained from comm.py2f()

    Returns:
        None (result is written to recvbuf in-place)
    """
    _alltoall_out_check(recvbuf, sendbuf, comm_ptr)
    torch.ops.torch_mpi_ext.alltoall_out.default(recvbuf, sendbuf, comm_ptr)


def alltoallv(
    sendbuf: Tensor,
    sendcounts: Tensor,
    sdispls: Tensor,
    recvcounts: Tensor,
    rdispls: Tensor,
    comm_ptr: int,
) -> Tensor:
    """
    Performs MPI alltoallv operation.

    Alltoallv is a generalized all-to-all communication operation where each process
    sends different amounts of data to each other process, and receives different
    amounts of data from each other process.

    Args:
        sendbuf: Tensor containing data to send to all processes
        sendcounts: 1D Tensor of length world_size, specifying the number of elements
                    to send to each process (sendcounts[j] = number of elements to send to process j)
        sdispls: 1D Tensor of length world_size, specifying the displacement (in elements)
                 from the start of sendbuf for data to send to each process
        recvcounts: 1D Tensor of length world_size, specifying the number of elements
                    to receive from each process (recvcounts[j] = number of elements to receive from process j)
        rdispls: 1D Tensor of length world_size, specifying the displacement (in elements)
                 from the start of recvbuf for data to receive from each process
        comm_ptr: MPI communicator handle obtained from comm.py2f()

    Returns:
        Tensor containing the received data from all processes
    """
    _alltoallv_check(sendbuf, sendcounts, sdispls, recvcounts, rdispls, comm_ptr)
    return torch.ops.torch_mpi_ext.alltoallv.default(sendbuf, sendcounts, sdispls, recvcounts, rdispls, comm_ptr)


def alltoallv_out(
    recvbuf: Tensor,
    sendbuf: Tensor,
    sendcounts: Tensor,
    sdispls: Tensor,
    recvcounts: Tensor,
    rdispls: Tensor,
    comm_ptr: int,
) -> None:
    """
    Performs MPI alltoallv operation in-place.

    Alltoallv is a generalized all-to-all communication operation where each process
    sends different amounts of data to each other process, and receives different
    amounts of data from each other process.

    This in-place version writes the result directly to a pre-allocated recvbuf tensor.

    Args:
        recvbuf: Pre-allocated tensor to write the received data to
        sendbuf: Tensor containing data to send to all processes
        sendcounts: 1D Tensor of length world_size, specifying the number of elements
                    to send to each process
        sdispls: 1D Tensor of length world_size, specifying the displacement (in elements)
                 from the start of sendbuf for data to send to each process
        recvcounts: 1D Tensor of length world_size, specifying the number of elements
                    to receive from each process
        rdispls: 1D Tensor of length world_size, specifying the displacement (in elements)
                 from the start of recvbuf for data to receive from each process
        comm_ptr: MPI communicator handle obtained from comm.py2f()

    Returns:
        None (result is written to recvbuf in-place)
    """
    _alltoallv_out_check(recvbuf, sendbuf, sendcounts, sdispls, recvcounts, rdispls, comm_ptr)
    return torch.ops.torch_mpi_ext.alltoallv_out.default(recvbuf, sendbuf, sendcounts, sdispls, recvcounts, rdispls, comm_ptr)


def _alltoall_check(sendbuf: Tensor, comm_ptr: int) -> None:
    """Check function for alltoall - validates inputs"""
    torch._check(isinstance(comm_ptr, int), "comm_ptr must be an integer")


def _alltoall_out_check(recvbuf: Tensor, sendbuf: Tensor, comm_ptr: int) -> None:
    """Check function for alltoall_out - validates inputs"""
    torch._check(isinstance(comm_ptr, int), "comm_ptr must be an integer")
    torch._check(recvbuf.device.type == sendbuf.device.type, "recvbuf must be on the same device as sendbuf")
    torch._check(recvbuf.numel() == sendbuf.numel(), "recvbuf must have same size as sendbuf")


def _alltoallv_check(
    sendbuf: Tensor,
    sendcounts: Tensor,
    sdispls: Tensor,
    recvcounts: Tensor,
    rdispls: Tensor,
    comm_ptr: int,
) -> None:
    """Check function for alltoallv - validates inputs"""
    torch._check(isinstance(comm_ptr, int), "comm_ptr must be an integer")
    device_type = sendbuf.device.type
    torch._check(sendbuf.device.type == device_type, "sendbuf must be on the same device as other tensors")
    torch._check(sendcounts.device.type == device_type, "sendcounts must be on the same device as other tensors")
    torch._check(sdispls.device.type == device_type, "sdispls must be on the same device as other tensors")
    torch._check(recvcounts.device.type == device_type, "recvcounts must be on the same device as other tensors")
    torch._check(rdispls.device.type == device_type, "rdispls must be on the same device as other tensors")
    torch._check(sendcounts.ndim == 1, "sendcounts must be 1-dimensional")
    torch._check(sdispls.ndim == 1, "sdispls must be 1-dimensional")
    torch._check(recvcounts.ndim == 1, "recvcounts must be 1-dimensional")
    torch._check(rdispls.ndim == 1, "rdispls must be 1-dimensional")
    torch._check(sendcounts.numel() == sdispls.numel(), "sendcounts and sdispls must have same length")
    torch._check(recvcounts.numel() == rdispls.numel(), "recvcounts and rdispls must have same length")


def _alltoallv_out_check(
    recvbuf: Tensor,
    sendbuf: Tensor,
    sendcounts: Tensor,
    sdispls: Tensor,
    recvcounts: Tensor,
    rdispls: Tensor,
    comm_ptr: int,
) -> None:
    """Check function for alltoallv_out - validates inputs"""
    torch._check(isinstance(comm_ptr, int), "comm_ptr must be an integer")
    device_type = sendbuf.device.type
    torch._check(recvbuf.device.type == device_type, "recvbuf must be on the same device as sendbuf")
    torch._check(sendbuf.device.type == device_type, "sendbuf must be on the same device as recvbuf")
    torch._check(sendcounts.device.type == device_type, "sendcounts must be on the same device as sendbuf")
    torch._check(sdispls.device.type == device_type, "sdispls must be on the same device as sendbuf")
    torch._check(recvcounts.device.type == device_type, "recvcounts must be on the same device as sendbuf")
    torch._check(rdispls.device.type == device_type, "rdispls must be on the same device as other tensors")
    torch._check(sendcounts.ndim == 1, "sendcounts must be 1-dimensional")
    torch._check(sdispls.ndim == 1, "sdispls must be 1-dimensional")
    torch._check(recvcounts.ndim == 1, "recvcounts must be 1-dimensional")
    torch._check(rdispls.ndim == 1, "rdispls must be 1-dimensional")
    torch._check(sendcounts.numel() == sdispls.numel(), "sendcounts and sdispls must have same length")
    torch._check(recvcounts.numel() == rdispls.numel(), "recvcounts and rdispls must have same length")


@torch.library.register_fake("torch_mpi_ext::all_reduce_")
def _(input: Tensor, comm_ptr):
    torch._check(isinstance(comm_ptr, int))
    # In-place operation returns the same tensor
    return


@torch.library.register_fake("torch_mpi_ext::all_reduce")
def _(input: Tensor, comm_ptr):
    torch._check(isinstance(comm_ptr, int))
    return torch.empty_like(input)


@torch.library.register_fake("torch_mpi_ext::all_gather_into_tensor_out")
def _(output: Tensor, input: Tensor, comm_ptr: int, dim: int = -1):
    torch._check(isinstance(comm_ptr, int))
    torch._check(input.device.type == output.device.type)

    # Normalize negative dim values
    ndim = input.ndim
    torch._check(dim >= -ndim and dim < ndim)
    if dim < 0:
        dim = dim + ndim

    # Check that shapes match in all dimensions except dim
    for i in range(ndim):
        if i != dim:
            torch._check(input.size(i) == output.size(i))

    # Check that output size at dim is a multiple of input size at dim
    torch._check(output.size(dim) % input.size(dim) == 0)

    # This is an out-of-place operation that writes to output tensor


@torch.library.register_fake("torch_mpi_ext::all_reduce__wrapper")
def _(input: Tensor, comm_ptr_wrapper: Tensor):
    torch._check(comm_ptr_wrapper.ndim == 1)
    torch._check(comm_ptr_wrapper.size(0) == 1)
    torch._check(comm_ptr_wrapper.dtype == torch.int64)
    # In-place operation returns the same tensor
    return


@torch.library.register_fake("torch_mpi_ext::all_reduce_wrapper")
def _(input: Tensor, comm_ptr_wrapper: Tensor):
    torch._check(comm_ptr_wrapper.ndim == 1)
    torch._check(comm_ptr_wrapper.size(0) == 1)
    torch._check(comm_ptr_wrapper.dtype == torch.int64)
    return torch.empty_like(input)


@torch.library.register_fake("torch_mpi_ext::all_gather_into_tensor_wrapper")
def _(input: Tensor, comm_ptr_wrapper: Tensor, dim: int = -1):
    torch._check(comm_ptr_wrapper.ndim == 1)
    torch._check(comm_ptr_wrapper.size(0) == 1)
    torch._check(comm_ptr_wrapper.dtype == torch.int64)

    # Normalize negative dim values
    ndim = input.ndim
    torch._check(dim >= -ndim and dim < ndim)
    if dim < 0:
        dim = dim + ndim

    # Calculate output shape
    output_size = list(input.size())
    output_size[dim] = output_size[dim] * 2  # Placeholder for world_size
    return torch.empty(output_size, dtype=input.dtype, device=input.device)


@torch.library.register_fake("torch_mpi_ext::all_gather_into_tensor_out_wrapper")
def _(output: Tensor, input: Tensor, comm_ptr_wrapper: Tensor, dim: int = -1):
    torch._check(comm_ptr_wrapper.ndim == 1)
    torch._check(comm_ptr_wrapper.size(0) == 1)
    torch._check(comm_ptr_wrapper.dtype == torch.int64)
    torch._check(input.device.type == output.device.type)

    # Normalize negative dim values
    ndim = input.ndim
    torch._check(dim >= -ndim and dim < ndim)
    if dim < 0:
        dim = dim + ndim

    # Check that shapes match in all dimensions except dim
    for i in range(ndim):
        if i != dim:
            torch._check(input.size(i) == output.size(i))

    # Check that output size at dim is a multiple of input size at dim
    torch._check(output.size(dim) % input.size(dim) == 0)

    # This is an out-of-place operation that writes to output tensor


def _variable_collective_out_check(
    output: Tensor, input: Tensor, sizes: Tensor | None, dim: int
) -> None:
    ndim = input.ndim
    torch._check(ndim > 0)
    torch._check(dim >= -ndim and dim < ndim)
    if dim < 0:
        dim += ndim
    torch._check(output.ndim == ndim)
    torch._check(output.device.type == input.device.type)
    torch._check(output.dtype == input.dtype)
    for index in range(ndim):
        if index != dim:
            torch._check(output.size(index) == input.size(index))
    if sizes is not None:
        torch._check(sizes.ndim == 1)
        torch._check(sizes.dtype == torch.int64)


def _comm_ptr_wrapper_check(comm_ptr_wrapper: Tensor) -> None:
    torch._check(comm_ptr_wrapper.ndim == 1)
    torch._check(comm_ptr_wrapper.size(0) == 1)
    torch._check(comm_ptr_wrapper.dtype == torch.int64)


@torch.library.register_fake("torch_mpi_ext::reduce_scatter_out")
def _(output: Tensor, input: Tensor, comm_ptr: int, dim: int = -1):
    _variable_collective_out_check(output, input, None, dim)


@torch.library.register_fake("torch_mpi_ext::reduce_scatter_out_wrapper")
def _(
    output: Tensor,
    input: Tensor,
    comm_ptr_wrapper: Tensor,
    dim: int = -1,
):
    _comm_ptr_wrapper_check(comm_ptr_wrapper)
    _variable_collective_out_check(output, input, None, dim)


@torch.library.register_fake("torch_mpi_ext::reduce_scatterv_out")
def _(output: Tensor, input: Tensor, sizes: Tensor, comm_ptr: int, dim: int = -1):
    _variable_collective_out_check(output, input, sizes, dim)


@torch.library.register_fake("torch_mpi_ext::reduce_scatterv_out_wrapper")
def _(
    output: Tensor,
    input: Tensor,
    sizes: Tensor,
    comm_ptr_wrapper: Tensor,
    dim: int = -1,
):
    _comm_ptr_wrapper_check(comm_ptr_wrapper)
    _variable_collective_out_check(output, input, sizes, dim)


@torch.library.register_fake("torch_mpi_ext::alltoallv")
def _fake_alltoallv(sendbuf: Tensor, sendcounts: Tensor, sdispls: Tensor,
                    recvcounts: Tensor, rdispls: Tensor, comm_ptr: int) -> Tensor:
    """FakeTensor kernel for alltoallv"""
    _alltoallv_check(sendbuf, sendcounts, sdispls, recvcounts, rdispls, comm_ptr)
    # Calculate total receive count
    total_recv = int(recvcounts.sum().item())
    return torch.empty((total_recv,), dtype=sendbuf.dtype, device=sendbuf.device)


@torch.library.register_fake("torch_mpi_ext::alltoallv_out")
def _fake_alltoallv_out(recvbuf: Tensor, sendbuf: Tensor, sendcounts: Tensor,
                        sdispls: Tensor, recvcounts: Tensor, rdispls: Tensor,
                        comm_ptr: int) -> None:
    """FakeTensor kernel for alltoallv_out"""
    _alltoallv_out_check(recvbuf, sendbuf, sendcounts, sdispls, recvcounts, rdispls, comm_ptr)
    # In-place operation returns None


@torch.library.register_fake("torch_mpi_ext::alltoall_out")
def _fake_alltoall_out(recvbuf: Tensor, sendbuf: Tensor, comm_ptr: int) -> None:
    """FakeTensor kernel for alltoall_out"""
    _alltoall_out_check(recvbuf, sendbuf, comm_ptr)
    # In-place operation returns None
