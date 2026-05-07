#pragma once

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include "muladd.h"

void all_reduce_(at::Tensor& input, long comm_ptr);
at::Tensor all_reduce(const at::Tensor& input, long comm_ptr);
at::Tensor all_gather_into_tensor(const at::Tensor& input, long comm_ptr,
                                  int64_t dim);
void all_gather_into_tensor_out(at::Tensor& output, const at::Tensor& input,
                                long comm_ptr, int64_t dim);
at::Tensor alltoallv(const at::Tensor& sendbuf, const at::Tensor& sendcounts,
                     const at::Tensor& sdispls, const at::Tensor& recvcounts,
                     const at::Tensor& rdispls, long comm_ptr);
void alltoallv_out(at::Tensor& recvbuf, const at::Tensor& sendbuf,
                   const at::Tensor& sendcounts, const at::Tensor& sdispls,
                   const at::Tensor& recvcounts, const at::Tensor& rdispls,
                   long comm_ptr);
void alltoall_out(at::Tensor& recvbuf, const at::Tensor& sendbuf,
                  long comm_ptr);

void all_reduce__wrapper(at::Tensor& input, const at::Tensor& comm_ptr_wrapper);
at::Tensor all_reduce_wrapper(const at::Tensor& input,
                              const at::Tensor& comm_ptr_wrapper);
at::Tensor all_gather_into_tensor_wrapper(const at::Tensor& input,
                                          const at::Tensor& comm_ptr_wrapper,
                                          int64_t dim);
void all_gather_into_tensor_out_wrapper(at::Tensor& output,
                                        const at::Tensor& input,
                                        const at::Tensor& comm_ptr_wrapper,
                                        int64_t dim);
