#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <vector>

#include "mpi.h"

// Convert PyTorch dtype to MPI datatype for calculations
MPI_Datatype get_mpi_cal_datatype(const at::Tensor& tensor) {
  MPI_Datatype datatype;
  if (tensor.dtype() == torch::kFloat32) {
    datatype = MPI_FLOAT;
  } else if (tensor.dtype() == torch::kFloat16) {
    datatype = MPI_FLOAT;  // TODO
  } else if (tensor.dtype() == torch::kBFloat16) {
    datatype = MPI_FLOAT;  // TODO
  } else if (tensor.dtype() == torch::kInt32) {
    datatype = MPI_INT;
  } else if (tensor.dtype() == torch::kInt8) {
    datatype = MPI_CHAR;
  } else if (tensor.dtype() == torch::kInt16) {
    datatype = MPI_SHORT;
  } else if (tensor.dtype() == torch::kInt64) {
    datatype = MPI_LONG_LONG;
  } else if (tensor.dtype() == torch::kFloat64) {
    datatype = MPI_DOUBLE;
  } else {
    TORCH_CHECK(false, "Unsupported tensor dtype for MPI operations");
  }
  return datatype;
}

// Get MPI datatype for width-based operations
MPI_Datatype get_mpi_width_datatype(const at::Tensor& tensor) {
  MPI_Datatype datatype;
  if (tensor.dtype() == torch::kFloat32 || tensor.dtype() == torch::kInt32) {
    datatype = MPI_INT;
  } else if (tensor.dtype() == torch::kFloat16 ||
             tensor.dtype() == torch::kBFloat16 ||
             tensor.dtype() == torch::kInt16) {
    datatype = MPI_SHORT;
  } else if (tensor.dtype() == torch::kFloat64 ||
             tensor.dtype() == torch::kInt64) {
    datatype = MPI_LONG_LONG;
  } else {
    TORCH_CHECK(false, "Unsupported tensor dtype for MPI operations");
  }
  return datatype;
}

// In-place all-reduce operation
void all_reduce_(at::Tensor& input, long comm_ptr) {
  MPI_Fint f_handle = (MPI_Fint)comm_ptr;
  MPI_Comm c_comm = MPI_Comm_f2c(f_handle);

  TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");

  // TODO
  if (input.dtype() == torch::kFloat16 || input.dtype() == torch::kBFloat16) {
    at::Tensor input_fp32 = input.to(torch::kFloat32);
    auto datatype = get_mpi_cal_datatype(input_fp32);
    int result = MPI_Allreduce(MPI_IN_PLACE,           // send buffer
                               input_fp32.data_ptr(),  // receive buffer
                               input_fp32.numel(),     // count
                               datatype,               // datatype
                               MPI_SUM,                // operation
                               c_comm                  // communicator
    );
    TORCH_CHECK(result == MPI_SUCCESS, "MPI_Allreduce failed");
    at::native::copy_(input, input_fp32);

  } else {
    auto datatype = get_mpi_cal_datatype(input);

    int result = MPI_Allreduce(MPI_IN_PLACE,      // send buffer
                               input.data_ptr(),  // receive buffer
                               input.numel(),     // count
                               datatype,          // datatype
                               MPI_SUM,           // operation
                               c_comm             // communicator
    );

    TORCH_CHECK(result == MPI_SUCCESS, "MPI_Allreduce failed");
  }
}

// All-reduce operation (out-of-place)
at::Tensor all_reduce(const at::Tensor& input, long comm_ptr) {
  if (input.is_contiguous()) {
    at::Tensor input_new = input.clone();
    all_reduce_(input_new, comm_ptr);
    return input_new;
  } else {
    at::Tensor input_new = input.contiguous();
    all_reduce_(input_new, comm_ptr);
    return input_new;
  }
}

// All-gather operation
at::Tensor all_gather(const at::Tensor& input_, long comm_ptr, int64_t dim) {
  MPI_Fint f_handle = (MPI_Fint)comm_ptr;
  MPI_Comm c_comm = MPI_Comm_f2c(f_handle);

  const at::Tensor& input = input_.contiguous();
  auto datatype = get_mpi_width_datatype(input);

  int comm_size;
  int result = MPI_Comm_size(c_comm, &comm_size);
  TORCH_CHECK(result == MPI_SUCCESS, "MPI_Comm_size failed");

  int actual_dim = dim;
  if (actual_dim < 0) {
    actual_dim += input.dim();
  }

  auto input_sizes = input.sizes();

  // First reshape to (world_size, ...) format
  std::vector<int64_t> intermediate_shape;
  intermediate_shape.push_back(comm_size);
  for (size_t i = 0; i < input_sizes.size(); ++i) {
    intermediate_shape.push_back(input_sizes[i]);
  }
  at::Tensor output = at::empty(intermediate_shape, input.options());

  result = MPI_Allgather(input.data_ptr(),   // send buffer
                         input.numel(),      // send count
                         datatype,           // send datatype
                         output.data_ptr(),  // receive buffer
                         input.numel(),      // receive count
                         datatype,           // receive datatype
                         c_comm              // communicator
  );

  TORCH_CHECK(result == MPI_SUCCESS, "MPI_Allgather failed");

  // Move the world_size dimension to the target dim
  output = output.movedim(0, actual_dim);

  // Reshape to final form
  std::vector<int64_t> final_shape(input_sizes.begin(), input_sizes.end());
  final_shape[actual_dim] = final_shape[actual_dim] * comm_size;
  output = output.reshape(final_shape);

  return output;
}