import torch
from torch import Tensor

__all__ = ["mymuladd", "myadd_out", "all_reduce", "all_gather_into_tensor", 
           "alltoall", "alltoall_out", "alltoallv", "alltoallv_out"]


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
    torch._check(sendbuf.device.type == "cpu", "sendbuf must be on CPU")


def _alltoall_out_check(recvbuf: Tensor, sendbuf: Tensor, comm_ptr: int) -> None:
    """Check function for alltoall_out - validates inputs"""
    torch._check(isinstance(comm_ptr, int), "comm_ptr must be an integer")
    torch._check(recvbuf.device.type == "cpu", "recvbuf must be on CPU")
    torch._check(sendbuf.device.type == "cpu", "sendbuf must be on CPU")
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
    torch._check(sendbuf.device.type == "cpu", "sendbuf must be on CPU")
    torch._check(sendcounts.device.type == "cpu", "sendcounts must be on CPU")
    torch._check(sdispls.device.type == "cpu", "sdispls must be on CPU")
    torch._check(recvcounts.device.type == "cpu", "recvcounts must be on CPU")
    torch._check(rdispls.device.type == "cpu", "rdispls must be on CPU")
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
    torch._check(recvbuf.device.type == "cpu", "recvbuf must be on CPU")
    torch._check(sendbuf.device.type == "cpu", "sendbuf must be on CPU")
    torch._check(sendcounts.device.type == "cpu", "sendcounts must be on CPU")
    torch._check(sdispls.device.type == "cpu", "sdispls must be on CPU")
    torch._check(recvcounts.device.type == "cpu", "recvcounts must be on CPU")
    torch._check(rdispls.device.type == "cpu", "rdispls must be on CPU")
    torch._check(sendcounts.ndim == 1, "sendcounts must be 1-dimensional")
    torch._check(sdispls.ndim == 1, "sdispls must be 1-dimensional")
    torch._check(recvcounts.ndim == 1, "recvcounts must be 1-dimensional")
    torch._check(rdispls.ndim == 1, "rdispls must be 1-dimensional")
    torch._check(sendcounts.numel() == sdispls.numel(), "sendcounts and sdispls must have same length")
    torch._check(recvcounts.numel() == rdispls.numel(), "recvcounts and rdispls must have same length")


@torch.library.register_fake("torch_mpi_ext::all_reduce_")
def _(input: Tensor, comm_ptr):
    torch._check(isinstance(comm_ptr, int))
    torch._check(input.device.type == "cpu")
    # In-place operation returns the same tensor
    return


@torch.library.register_fake("torch_mpi_ext::all_reduce")
def _(input: Tensor, comm_ptr):
    torch._check(isinstance(comm_ptr, int))
    torch._check(input.device.type == "cpu")
    return torch.empty_like(input)


@torch.library.register_fake("torch_mpi_ext::all_gather_into_tensor_out")
def _(output: Tensor, input: Tensor, comm_ptr: int, dim: int = -1):
    torch._check(isinstance(comm_ptr, int))
    torch._check(input.device.type == "cpu")
    torch._check(output.device.type == "cpu")
    
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


@torch.library.register_fake("torch_mpi_ext::alltoallv")
def _fake_alltoallv(sendbuf: Tensor, sendcounts: Tensor, sdispls: Tensor,
                    recvcounts: Tensor, rdispls: Tensor, comm_ptr: int) -> Tensor:
    """FakeTensor kernel for alltoallv"""
    _alltoallv_check(sendbuf, sendcounts, sdispls, recvcounts, rdispls, comm_ptr)
    # Calculate total receive count
    total_recv = int(recvcounts.sum().item())
    return torch.empty((total_recv,), dtype=sendbuf.dtype)


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
