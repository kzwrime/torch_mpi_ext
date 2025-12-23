import torch
from torch import Tensor

__all__ = ["mymuladd", "myadd_out", "all_reduce", "all_reduce_", "all_gather"]


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


def all_reduce_(input: Tensor, comm_ptr: int) -> Tensor:
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


def all_gather(input: Tensor, comm_ptr: int, dim: int = -1) -> Tensor:
    """
    Performs MPI allgather operation on the input tensor.
    
    Args:
        input: Input tensor to gather
        comm_ptr: MPI communicator handle obtained from comm.py2f()
        dim: Dimension along which to concatenate the gathered tensors
        
    Returns:
        Gathered tensor with specified dimension scaled by world size
    """
    return torch.ops.torch_mpi_ext.all_gather.default(input, comm_ptr, dim)


def all_gather_into_tensor(output: Tensor, input: Tensor, comm_ptr: int, dim: int = -1) -> None:
    """
    Performs MPI allgather operation on the input tensor and writes the result directly to the output tensor.
    
    Args:
        output: Output tensor to write the gathered result to
        input: Input tensor to gather
        comm_ptr: MPI communicator handle obtained from comm.py2f()
        dim: Dimension along which to concatenate the gathered tensors
        
    Returns:
        None (result is written to the output tensor)
    """
    torch.ops.torch_mpi_ext.all_gather_into_tensor.default(output, input, comm_ptr, dim)


@torch.library.register_fake("torch_mpi_ext::all_reduce_")
def _(input: Tensor, comm_ptr):
    torch._check(isinstance(comm_ptr, int))
    torch._check(input.device.type == "cpu")
    # In-place operation returns the same tensor
    return input


@torch.library.register_fake("torch_mpi_ext::all_reduce")
def _(input: Tensor, comm_ptr):
    torch._check(isinstance(comm_ptr, int))
    torch._check(input.device.type == "cpu")
    return torch.empty_like(input)


@torch.library.register_fake("torch_mpi_ext::all_gather_into_tensor")
def _(output: Tensor, input: Tensor, comm_ptr: int, dim: int = -1):
    torch._check(isinstance(comm_ptr, int))
    torch._check(input.device.type == "cpu")
    torch._check(output.device.type == "cpu")
    # This is an out-of-place operation that writes to output tensor
