#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <runtime/McpuKernelLaunch.h>

#include <limits>
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

template <typename Dst, typename Src>
void copy_with_cast(void* dst_ptr, const void* src_ptr, int64_t numel) {
  auto* dst = static_cast<Dst*>(dst_ptr);
  const auto* src = static_cast<const Src*>(src_ptr);
  for (int64_t i = 0; i < numel; ++i) {
    dst[i] = static_cast<Dst>(src[i]);
  }
}

struct DirectAllgatherShape {
  int send_count = 0;
  int outer = 0;
  int block_elements = 0;
  int stride_elements = 0;
};

DirectAllgatherShape validate_allgather_direct_output_dim(
    int64_t numel,
    int64_t comm_size,
    int64_t outer,
    int64_t dim_size,
    int64_t inner) {
  const int64_t block_elements = dim_size * inner;
  const int64_t stride_elements = comm_size * block_elements;

  TORCH_CHECK(numel <= std::numeric_limits<int>::max(),
              "MPI_Allgather count exceeds int range: ", numel);
  TORCH_CHECK(outer <= std::numeric_limits<int>::max(),
              "MPI receive type outer dimension exceeds int range: ", outer);
  TORCH_CHECK(block_elements <= std::numeric_limits<int>::max(),
              "MPI receive type block length exceeds int range: ",
              block_elements);
  TORCH_CHECK(stride_elements <= std::numeric_limits<int>::max(),
              "MPI receive type stride exceeds int range: ", stride_elements);

  DirectAllgatherShape shape;
  shape.send_count = static_cast<int>(numel);
  shape.outer = static_cast<int>(outer);
  shape.block_elements = static_cast<int>(block_elements);
  shape.stride_elements = static_cast<int>(stride_elements);
  return shape;
}

void allgather_direct_into_output_dim(
    const void* input_ptr,
    void* output_ptr,
    MPI_Datatype datatype,
    MPI_Comm c_comm,
    const DirectAllgatherShape& shape) {
  MPI_Datatype strided_type = MPI_DATATYPE_NULL;
  int result = MPI_Type_vector(
      shape.outer,
      shape.block_elements,
      shape.stride_elements,
      datatype,
      &strided_type);
  TORCH_CHECK(result == MPI_SUCCESS, "MPI_Type_vector failed");

  MPI_Aint lb = 0;
  MPI_Aint datatype_extent = 0;
  result = MPI_Type_get_extent(datatype, &lb, &datatype_extent);
  if (result != MPI_SUCCESS) {
    MPI_Type_free(&strided_type);
  }
  TORCH_CHECK(result == MPI_SUCCESS, "MPI_Type_get_extent failed");

  MPI_Datatype recv_type = MPI_DATATYPE_NULL;
  result = MPI_Type_create_resized(
      strided_type, 0, shape.block_elements * datatype_extent, &recv_type);
  if (result != MPI_SUCCESS) {
    MPI_Type_free(&strided_type);
  }
  TORCH_CHECK(result == MPI_SUCCESS, "MPI_Type_create_resized failed");

  result = MPI_Type_commit(&recv_type);
  if (result != MPI_SUCCESS) {
    MPI_Type_free(&recv_type);
    MPI_Type_free(&strided_type);
  }
  TORCH_CHECK(result == MPI_SUCCESS, "MPI_Type_commit failed");

  const int allgather_result = MPI_Allgather(
      input_ptr,
      shape.send_count,
      datatype,
      output_ptr,
      1,
      recv_type,
      c_comm);

  const int free_recv_result = MPI_Type_free(&recv_type);
  const int free_strided_result = MPI_Type_free(&strided_type);

  TORCH_CHECK(allgather_result == MPI_SUCCESS, "MPI_Allgather failed");
  TORCH_CHECK(free_recv_result == MPI_SUCCESS, "MPI_Type_free failed");
  TORCH_CHECK(free_strided_result == MPI_SUCCESS, "MPI_Type_free failed");
}

// In-place all-reduce operation
void all_reduce_(at::Tensor& input, long comm_ptr) {
  MPI_Fint f_handle = (MPI_Fint)comm_ptr;
  MPI_Comm c_comm = MPI_Comm_f2c(f_handle);

  TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");

  // TODO
  if (input.dtype() == torch::kFloat16 || input.dtype() == torch::kBFloat16) {
    if (input.device().type() == c10::DeviceType::PrivateUse1) {
      void* input_ptr = input.data_ptr();
      int64_t numel = input.numel();
      auto scalar_type = input.scalar_type();
      at::mcpu::launch_timed_kernel(
          "mcpu::torch_mpi_ext::all_reduce_",
          [input_ptr, numel, scalar_type,
           c_comm](at::mcpu::kernel_timing::Event* timing_event) mutable {
            MCPU_KERNEL_TIMING_SCOPE_EVENT("mcpu::torch_mpi_ext::all_reduce_",
                                           timing_event);
            at::mcpu::KernelPointerMemoryGuard guard({input_ptr});

            std::vector<float> buffer(numel);
            if (scalar_type == at::kHalf) {
              copy_with_cast<float, c10::Half>(buffer.data(), input_ptr, numel);
            } else {
              copy_with_cast<float, c10::BFloat16>(buffer.data(), input_ptr,
                                                   numel);
            }

            int result = MPI_Allreduce(MPI_IN_PLACE,   // send buffer
                                       buffer.data(),  // receive buffer
                                       numel,          // count
                                       MPI_FLOAT,      // datatype
                                       MPI_SUM,        // operation
                                       c_comm          // communicator
            );
            TORCH_CHECK(result == MPI_SUCCESS, "MPI_Allreduce failed");

            if (scalar_type == at::kHalf) {
              copy_with_cast<c10::Half, float>(input_ptr, buffer.data(), numel);
            } else {
              copy_with_cast<c10::BFloat16, float>(input_ptr, buffer.data(),
                                                   numel);
            }
          });
      return;
    } else {
      at::Tensor input_fp32 = input.to(torch::kFloat32);
      auto datatype = get_mpi_cal_datatype(input_fp32);
      void* input_ptr = input_fp32.data_ptr();
      int64_t numel = input_fp32.numel();
      int result = MPI_Allreduce(MPI_IN_PLACE,  // send buffer
                                 input_ptr,     // receive buffer
                                 numel,         // count
                                 datatype,      // datatype
                                 MPI_SUM,       // operation
                                 c_comm         // communicator
      );
      TORCH_CHECK(result == MPI_SUCCESS, "MPI_Allreduce failed");
      input.copy_(input_fp32);
    }
  } else {
    auto datatype = get_mpi_cal_datatype(input);
    void* input_ptr = input.data_ptr();
    int64_t numel = input.numel();

    if (input.device().type() == c10::DeviceType::PrivateUse1) {
      at::mcpu::launch_timed_kernel(
          "mcpu::torch_mpi_ext::all_reduce_",
          [input_ptr, numel, datatype, c_comm](
              at::mcpu::kernel_timing::Event* timing_event) mutable {
            MCPU_KERNEL_TIMING_SCOPE_EVENT(
                "mcpu::torch_mpi_ext::all_reduce_", timing_event);
            at::mcpu::KernelPointerMemoryGuard guard({input_ptr});
            int result = MPI_Allreduce(MPI_IN_PLACE,  // send buffer
                                       input_ptr,      // receive buffer
                                       numel,          // count
                                       datatype,       // datatype
                                       MPI_SUM,        // operation
                                       c_comm          // communicator
            );
            TORCH_CHECK(result == MPI_SUCCESS, "MPI_Allreduce failed");
          });
    } else {
      int result = MPI_Allreduce(MPI_IN_PLACE,  // send buffer
                                 input_ptr,     // receive buffer
                                 numel,         // count
                                 datatype,      // datatype
                                 MPI_SUM,       // operation
                                 c_comm         // communicator
      );
      TORCH_CHECK(result == MPI_SUCCESS, "MPI_Allreduce failed");
    }
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
at::Tensor all_gather_into_tensor(const at::Tensor& input_, long comm_ptr,
                                  int64_t dim) {
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
  std::vector<int64_t> final_shape(input_sizes.begin(), input_sizes.end());
  final_shape[actual_dim] = final_shape[actual_dim] * comm_size;
  at::Tensor output = at::empty(final_shape, input.options());

  if (input.device().type() == c10::DeviceType::PrivateUse1) {
    int64_t outer = 1;
    for (int64_t i = 0; i < actual_dim; ++i) {
      outer *= input_sizes[i];
    }
    int64_t dim_size = input_sizes[actual_dim];
    int64_t inner = 1;
    for (int64_t i = actual_dim + 1; i < input.dim(); ++i) {
      inner *= input_sizes[i];
    }

    const void* input_ptr = input.data_ptr();
    void* output_ptr = output.data_ptr();
    int64_t numel = input.numel();
    DirectAllgatherShape allgather_shape =
        validate_allgather_direct_output_dim(
            numel, comm_size, outer, dim_size, inner);

    at::mcpu::launch_timed_kernel(
        "mcpu::torch_mpi_ext::all_gather_into_tensor",
        [input_ptr,
         output_ptr,
         datatype,
         c_comm,
         allgather_shape](
            at::mcpu::kernel_timing::Event* timing_event) mutable {
          MCPU_KERNEL_TIMING_SCOPE_EVENT(
              "mcpu::torch_mpi_ext::all_gather_into_tensor", timing_event);
          at::mcpu::KernelPointerMemoryGuard guard({input_ptr, output_ptr});
          allgather_direct_into_output_dim(
              input_ptr,
              output_ptr,
              datatype,
              c_comm,
              allgather_shape);
        });
  } else {
    at::Tensor temp_buffer = at::empty(intermediate_shape, input.options());

    result = MPI_Allgather(input.data_ptr(),        // send buffer
                           input.numel(),           // send count
                           datatype,                // send datatype
                           temp_buffer.data_ptr(),  // receive buffer
                           input.numel(),           // receive count
                           datatype,                // receive datatype
                           c_comm                   // communicator
    );

    TORCH_CHECK(result == MPI_SUCCESS, "MPI_Allgather failed");

    // Move the world_size dimension to the target dim
    temp_buffer = temp_buffer.movedim(0, actual_dim);

    output.copy_(temp_buffer.reshape(final_shape));
  }

  TORCH_CHECK(output.is_contiguous(), "tensor must be contiguous");

  return output;
}

// All-gather operation that writes directly into a pre-allocated output tensor
at::Tensor& all_gather_into_tensor_out(at::Tensor& output,
                                       const at::Tensor& input_, long comm_ptr,
                                       int64_t dim) {
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

  // Create intermediate tensor with shape (world_size, ...) to receive the
  // allgather result
  std::vector<int64_t> intermediate_shape;
  intermediate_shape.push_back(comm_size);
  for (size_t i = 0; i < input_sizes.size(); ++i) {
    intermediate_shape.push_back(input_sizes[i]);
  }

  if (input.device().type() == c10::DeviceType::PrivateUse1) {
    int64_t outer = 1;
    for (int64_t i = 0; i < actual_dim; ++i) {
      outer *= input_sizes[i];
    }
    int64_t dim_size = input_sizes[actual_dim];
    int64_t inner = 1;
    for (int64_t i = actual_dim + 1; i < input.dim(); ++i) {
      inner *= input_sizes[i];
    }

    const void* input_ptr = input.data_ptr();
    void* output_ptr = output.data_ptr();
    int64_t numel = input.numel();
    DirectAllgatherShape allgather_shape =
        validate_allgather_direct_output_dim(
            numel, comm_size, outer, dim_size, inner);

    at::mcpu::launch_timed_kernel(
        "mcpu::torch_mpi_ext::all_gather_into_tensor_out",
        [input_ptr,
         output_ptr,
         datatype,
         c_comm,
         allgather_shape](
            at::mcpu::kernel_timing::Event* timing_event) mutable {
          MCPU_KERNEL_TIMING_SCOPE_EVENT(
              "mcpu::torch_mpi_ext::all_gather_into_tensor_out", timing_event);
          at::mcpu::KernelPointerMemoryGuard guard({input_ptr, output_ptr});
          allgather_direct_into_output_dim(
              input_ptr,
              output_ptr,
              datatype,
              c_comm,
              allgather_shape);
        });
  } else {
    // We need a temporary tensor to hold the allgather result in the
    // intermediate shape. Then we move the world_size dimension into place.
    at::Tensor temp_buffer = at::empty(intermediate_shape, input.options());

    result = MPI_Allgather(input.data_ptr(),        // send buffer
                           input.numel(),           // send count
                           datatype,                // send datatype
                           temp_buffer.data_ptr(),  // receive buffer
                           input.numel(),           // receive count
                           datatype,                // receive datatype
                           c_comm                   // communicator
    );

    TORCH_CHECK(result == MPI_SUCCESS, "MPI_Allgather failed");

    // Move the world_size dimension to the target dim
    temp_buffer = temp_buffer.movedim(0, actual_dim);

    // Copy the result to the output tensor
    output.copy_(temp_buffer.reshape(expected_shape));
  }

  TORCH_CHECK(output.is_contiguous(), "tensor must be contiguous");

  return output;
}

// All-to-allv operation (non-in-place version)
at::Tensor alltoallv(const at::Tensor& sendbuf_, const at::Tensor& sendcounts_,
                     const at::Tensor& sdispls_, const at::Tensor& recvcounts_,
                     const at::Tensor& rdispls_, long comm_ptr) {
  MPI_Fint f_handle = (MPI_Fint)comm_ptr;
  MPI_Comm c_comm = MPI_Comm_f2c(f_handle);

  const at::Tensor& sendbuf = sendbuf_.contiguous();
  auto datatype = get_mpi_width_datatype(sendbuf);

  int comm_size;
  int result = MPI_Comm_size(c_comm, &comm_size);
  TORCH_CHECK(result == MPI_SUCCESS, "MPI_Comm_size failed");

  // Convert tensors to contiguous int32 arrays
  auto sendcounts = sendcounts_.to(torch::kInt32).contiguous();
  auto sdispls = sdispls_.to(torch::kInt32).contiguous();
  auto recvcounts = recvcounts_.to(torch::kInt32).contiguous();
  auto rdispls = rdispls_.to(torch::kInt32).contiguous();

  TORCH_CHECK(sendcounts.numel() == comm_size,
              "sendcounts must have size equal to comm_size");
  TORCH_CHECK(sdispls.numel() == comm_size,
              "sdispls must have size equal to comm_size");
  TORCH_CHECK(recvcounts.numel() == comm_size,
              "recvcounts must have size equal to comm_size");
  TORCH_CHECK(rdispls.numel() == comm_size,
              "rdispls must have size equal to comm_size");

  // Calculate total receive count
  int* recvcounts_ptr = (int*)recvcounts.data_ptr();
  int total_recv_count = 0;
  for (int i = 0; i < comm_size; ++i) {
    total_recv_count += recvcounts_ptr[i];
  }

  // Allocate output tensor
  at::Tensor recvbuf = at::empty({total_recv_count}, sendbuf.options());

  result = MPI_Alltoallv(sendbuf.data_ptr(),           // send buffer
                         (int*)sendcounts.data_ptr(),  // send counts
                         (int*)sdispls.data_ptr(),     // send displacements
                         datatype,                     // send datatype
                         recvbuf.data_ptr(),           // receive buffer
                         (int*)recvcounts.data_ptr(),  // receive counts
                         (int*)rdispls.data_ptr(),     // receive displacements
                         datatype,                     // receive datatype
                         c_comm                        // communicator
  );

  TORCH_CHECK(result == MPI_SUCCESS, "MPI_Alltoallv failed");

  return recvbuf;
}

// All-to-allv operation (in-place version with pre-allocated output)
void alltoallv_out(at::Tensor& recvbuf_, const at::Tensor& sendbuf_,
                   const at::Tensor& sendcounts_, const at::Tensor& sdispls_,
                   const at::Tensor& recvcounts_, const at::Tensor& rdispls_,
                   long comm_ptr) {
  MPI_Fint f_handle = (MPI_Fint)comm_ptr;
  MPI_Comm c_comm = MPI_Comm_f2c(f_handle);

  at::Tensor& recvbuf = recvbuf_;
  const at::Tensor& sendbuf = sendbuf_.contiguous();
  auto datatype = get_mpi_width_datatype(sendbuf);

  int comm_size;
  int result = MPI_Comm_size(c_comm, &comm_size);
  TORCH_CHECK(result == MPI_SUCCESS, "MPI_Comm_size failed");

  // Convert tensors to contiguous int32 arrays
  auto sendcounts = sendcounts_.to(torch::kInt32).contiguous();
  auto sdispls = sdispls_.to(torch::kInt32).contiguous();
  auto recvcounts = recvcounts_.to(torch::kInt32).contiguous();
  auto rdispls = rdispls_.to(torch::kInt32).contiguous();

  TORCH_CHECK(sendcounts.numel() == comm_size,
              "sendcounts must have size equal to comm_size");
  TORCH_CHECK(sdispls.numel() == comm_size,
              "sdispls must have size equal to comm_size");
  TORCH_CHECK(recvcounts.numel() == comm_size,
              "recvcounts must have size equal to comm_size");
  TORCH_CHECK(rdispls.numel() == comm_size,
              "rdispls must have size equal to comm_size");

  result = MPI_Alltoallv(sendbuf.data_ptr(),           // send buffer
                         (int*)sendcounts.data_ptr(),  // send counts
                         (int*)sdispls.data_ptr(),     // send displacements
                         datatype,                     // send datatype
                         recvbuf.data_ptr(),           // receive buffer
                         (int*)recvcounts.data_ptr(),  // receive counts
                         (int*)rdispls.data_ptr(),     // receive displacements
                         datatype,                     // receive datatype
                         c_comm                        // communicator
  );

  TORCH_CHECK(result == MPI_SUCCESS, "MPI_Alltoallv failed");
}

// All-to-all operation (out-of-place version with pre-allocated output)
void alltoall_out(at::Tensor& recvbuf, const at::Tensor& sendbuf_,
                  long comm_ptr) {
  MPI_Fint f_handle = (MPI_Fint)comm_ptr;
  MPI_Comm c_comm = MPI_Comm_f2c(f_handle);

  const at::Tensor& sendbuf = sendbuf_.contiguous();
  auto datatype = get_mpi_width_datatype(sendbuf);

  int comm_size;
  int result = MPI_Comm_size(c_comm, &comm_size);
  TORCH_CHECK(result == MPI_SUCCESS, "MPI_Comm_size failed");

  // Calculate the count per rank (assuming all ranks send/receive same amount)
  int total_count = sendbuf.numel();
  TORCH_CHECK(total_count % comm_size == 0,
              "Total send count must be divisible by comm_size");
  int count = total_count / comm_size;

  // Verify recvbuf has correct size
  TORCH_CHECK(recvbuf.numel() == total_count,
              "recvbuf must have same size as sendbuf");

  result = MPI_Alltoall(sendbuf.data_ptr(),  // send buffer
                        count,               // send count per rank
                        datatype,            // send datatype
                        recvbuf.data_ptr(),  // receive buffer
                        count,               // receive count per rank
                        datatype,            // receive datatype
                        c_comm               // communicator
  );

  TORCH_CHECK(result == MPI_SUCCESS, "MPI_Alltoall failed");
}

// =============================================================================
// Wrapper functions that accept comm_ptr_wrapper (Tensor)
// These extract the comm_ptr from the tensor and call the original functions
// =============================================================================

long get_comm_ptr_from_cpu_wrapper(const at::Tensor& comm_ptr_wrapper) {
  TORCH_CHECK(
      comm_ptr_wrapper.device().type() == c10::DeviceType::CPU,
      "comm_ptr_wrapper must be a CPU tensor");
  TORCH_CHECK(comm_ptr_wrapper.ndimension() == 1 &&
                  comm_ptr_wrapper.size(0) == 1,
              "comm_ptr_wrapper must be a 1-element tensor");
  TORCH_CHECK(comm_ptr_wrapper.dtype() == torch::kInt64,
              "comm_ptr_wrapper must be int64 dtype");
  return (long)comm_ptr_wrapper.data_ptr<int64_t>()[0];
}

// Wrapper for all_reduce_ (in-place)
void all_reduce__wrapper(at::Tensor& input,
                         const at::Tensor& comm_ptr_wrapper) {
  long comm_ptr = get_comm_ptr_from_cpu_wrapper(comm_ptr_wrapper);

  all_reduce_(input, comm_ptr);
}

// Wrapper for all_reduce (out-of-place)
at::Tensor all_reduce_wrapper(const at::Tensor& input,
                              const at::Tensor& comm_ptr_wrapper) {
  long comm_ptr = get_comm_ptr_from_cpu_wrapper(comm_ptr_wrapper);

  return all_reduce(input, comm_ptr);
}

// Wrapper for all_gather_into_tensor
at::Tensor all_gather_into_tensor_wrapper(const at::Tensor& input,
                                          const at::Tensor& comm_ptr_wrapper,
                                          int64_t dim) {
  long comm_ptr = get_comm_ptr_from_cpu_wrapper(comm_ptr_wrapper);

  return all_gather_into_tensor(input, comm_ptr, dim);
}

// Wrapper for all_gather_into_tensor_out
void all_gather_into_tensor_out_wrapper(at::Tensor& output,
                                        const at::Tensor& input,
                                        const at::Tensor& comm_ptr_wrapper,
                                        int64_t dim) {
  long comm_ptr = get_comm_ptr_from_cpu_wrapper(comm_ptr_wrapper);

  all_gather_into_tensor_out(output, input, comm_ptr, dim);
}
