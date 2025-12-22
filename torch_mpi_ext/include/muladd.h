#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <vector>

namespace torch_mpi_ext {

at::Tensor mymuladd_cpu(const at::Tensor& a, const at::Tensor& b, double c);

at::Tensor mymul_cpu(const at::Tensor& a, const at::Tensor& b);

// An example of an operator that mutates one of its inputs.
void myadd_out_cpu(const at::Tensor& a, const at::Tensor& b, at::Tensor& out);

}
