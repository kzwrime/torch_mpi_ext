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
  } else if (tensor.dtype() == torch::kInt8) {
    datatype = MPI_CHAR;
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
at::Tensor all_gather_into_tensor(const at::Tensor& input_, long comm_ptr, int64_t dim) {
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

  TORCH_CHECK(output.is_contiguous(), "tensor must be contiguous");

  return output;
}

// All-gather operation that writes directly into a pre-allocated output tensor
at::Tensor& all_gather_into_tensor_out(at::Tensor& output, const at::Tensor& input_, long comm_ptr, int64_t dim) {
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

  // Validate that the output tensor has the correct shape
  std::vector<int64_t> expected_shape(input_sizes.begin(), input_sizes.end());
  expected_shape[actual_dim] = expected_shape[actual_dim] * comm_size;
  
  TORCH_CHECK(output.sizes().vec() == expected_shape, 
              "Output tensor has incorrect shape. Expected: ", expected_shape, 
              " Got: ", output.sizes().vec());

  // Create intermediate tensor with shape (world_size, ...) to receive the allgather result
  std::vector<int64_t> intermediate_shape;
  intermediate_shape.push_back(comm_size);
  for (size_t i = 0; i < input_sizes.size(); ++i) {
    intermediate_shape.push_back(input_sizes[i]);
  }
  
  // We need a temporary tensor to hold the allgather result in the intermediate shape
  // First, we create a tensor with the intermediate shape but flattened in the first 2 dimensions
  // Then we use MPI_Allgather to fill it
  at::Tensor temp_buffer = at::empty(intermediate_shape, input.options());

  result = MPI_Allgather(input.data_ptr(),   // send buffer
                         input.numel(),      // send count
                         datatype,           // send datatype
                         temp_buffer.data_ptr(),  // receive buffer
                         input.numel(),      // receive count
                         datatype,           // receive datatype
                         c_comm              // communicator
  );

  TORCH_CHECK(result == MPI_SUCCESS, "MPI_Allgather failed");

  // Move the world_size dimension to the target dim
  temp_buffer = temp_buffer.movedim(0, actual_dim);

  // Copy the result to the output tensor
  output.copy_(temp_buffer.reshape(expected_shape));

  TORCH_CHECK(output.is_contiguous(), "tensor must be contiguous");

  return output;
}
