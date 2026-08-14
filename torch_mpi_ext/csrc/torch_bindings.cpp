#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include "torch_bindings.h"

extern "C" {
/* Creates a dummy empty _C module that can be imported from Python.
   The import from Python will load the .so consisting of this file
   in this extension, so that the TORCH_LIBRARY static initializers
   below are run. */
PyObject* PyInit__C(void) {
  static struct PyModuleDef module_def = {
      PyModuleDef_HEAD_INIT,
      "_C", /* name of module */
      NULL, /* module documentation, may be NULL */
      -1,   /* size of per-interpreter state of the module,
               or -1 if the module keeps state in global variables. */
      NULL, /* methods */
  };
  return PyModule_Create(&module_def);
}
}

namespace torch_mpi_ext {

// Defines the operators
TORCH_LIBRARY(torch_mpi_ext, m) {
  m.def("mymuladd(Tensor a, Tensor b, float c) -> Tensor");
  m.def("mymul(Tensor a, Tensor b) -> Tensor");
  m.def("myadd_out(Tensor a, Tensor b, Tensor(a!) out) -> ()");
  m.def("all_reduce_(Tensor(a!) input, int comm_ptr) -> ()");
  m.def("all_reduce(Tensor input, int comm_ptr) -> Tensor");
  m.def(
      "all_gather_into_tensor(Tensor input, int comm_ptr, int dim = -1) -> "
      "Tensor");
  m.def(
      "all_gather_into_tensor_out(Tensor(a!) output, Tensor input, int "
      "comm_ptr, int dim = -1) -> ()");
  m.def("reduce_scatter(Tensor input, int comm_ptr, int dim = -1) -> Tensor");
  m.def(
      "reduce_scatter_out(Tensor(a!) output, Tensor input, int comm_ptr, int "
      "dim = -1) -> ()");
  m.def(
      "reduce_scatterv(Tensor input, Tensor sizes, int comm_ptr, int dim = "
      "-1) -> Tensor");
  m.def(
      "reduce_scatterv_out(Tensor(a!) output, Tensor input, Tensor sizes, int "
      "comm_ptr, int dim = -1) -> ()");
  m.def(
      "alltoallv(Tensor sendbuf, Tensor sendcounts, Tensor sdispls, Tensor "
      "recvcounts, Tensor rdispls, int comm_ptr) -> Tensor");
  m.def(
      "alltoallv_out(Tensor(a!) recvbuf, Tensor sendbuf, Tensor sendcounts, "
      "Tensor sdispls, Tensor recvcounts, Tensor rdispls, int comm_ptr) -> ()");
  m.def("alltoall_out(Tensor(a!) recvbuf, Tensor sendbuf, int comm_ptr) -> ()");
  m.def("all_reduce__wrapper(Tensor(a!) input, Tensor comm_ptr_wrapper) -> ()");
  m.def("all_reduce_wrapper(Tensor input, Tensor comm_ptr_wrapper) -> Tensor");
  m.def(
      "all_gather_into_tensor_wrapper(Tensor input, Tensor comm_ptr_wrapper, int dim = -1) -> "
      "Tensor");
  m.def(
      "all_gather_into_tensor_out_wrapper(Tensor(a!) output, Tensor input, Tensor "
      "comm_ptr_wrapper, int dim = -1) -> ()");
  m.def(
      "reduce_scatter_wrapper(Tensor input, Tensor comm_ptr_wrapper, int dim "
      "= -1) -> Tensor");
  m.def(
      "reduce_scatter_out_wrapper(Tensor(a!) output, Tensor input, Tensor "
      "comm_ptr_wrapper, int dim = -1) -> ()");
  m.def(
      "reduce_scatterv_wrapper(Tensor input, Tensor sizes, Tensor "
      "comm_ptr_wrapper, int dim = -1) -> Tensor");
  m.def(
      "reduce_scatterv_out_wrapper(Tensor(a!) output, Tensor input, Tensor "
      "sizes, Tensor comm_ptr_wrapper, int dim = -1) -> ()");
}

// Registers CPU implementations for mymuladd, mymul, myadd_out
TORCH_LIBRARY_IMPL(torch_mpi_ext, CPU, m) {
  m.impl("mymuladd", &mymuladd_cpu);
  m.impl("mymul", &mymul_cpu);
  m.impl("myadd_out", &myadd_out_cpu);
  m.impl("all_reduce_", &all_reduce_);
  m.impl("all_reduce", &all_reduce);
  m.impl("all_gather_into_tensor", &all_gather_into_tensor);
  m.impl("all_gather_into_tensor_out", &all_gather_into_tensor_out);
  m.impl("reduce_scatter", &reduce_scatter);
  m.impl("reduce_scatter_out", &reduce_scatter_out);
  m.impl("reduce_scatterv", &reduce_scatterv);
  m.impl("reduce_scatterv_out", &reduce_scatterv_out);
  m.impl("alltoallv", &alltoallv);
  m.impl("alltoallv_out", &alltoallv_out);
  m.impl("alltoall_out", &alltoall_out);
  m.impl("all_reduce__wrapper", &all_reduce__wrapper);
  m.impl("all_reduce_wrapper", &all_reduce_wrapper);
  m.impl("all_gather_into_tensor_wrapper", &all_gather_into_tensor_wrapper);
  m.impl("all_gather_into_tensor_out_wrapper", &all_gather_into_tensor_out_wrapper);
  m.impl("reduce_scatter_wrapper", &reduce_scatter_wrapper);
  m.impl("reduce_scatter_out_wrapper", &reduce_scatter_out_wrapper);
  m.impl("reduce_scatterv_wrapper", &reduce_scatterv_wrapper);
  m.impl("reduce_scatterv_out_wrapper", &reduce_scatterv_out_wrapper);
}

TORCH_LIBRARY_IMPL(torch_mpi_ext, PrivateUse1, m) {
  m.impl("mymuladd", &mymuladd_cpu);
  m.impl("mymul", &mymul_cpu);
  m.impl("myadd_out", &myadd_out_cpu);
  m.impl("all_reduce_", &all_reduce_);
  m.impl("all_reduce", &all_reduce);
  m.impl("all_gather_into_tensor", &all_gather_into_tensor);
  m.impl("all_gather_into_tensor_out", &all_gather_into_tensor_out);
  m.impl("reduce_scatter", &reduce_scatter);
  m.impl("reduce_scatter_out", &reduce_scatter_out);
  m.impl("reduce_scatterv", &reduce_scatterv);
  m.impl("reduce_scatterv_out", &reduce_scatterv_out);
  m.impl("alltoallv", &alltoallv);
  m.impl("alltoallv_out", &alltoallv_out);
  m.impl("alltoall_out", &alltoall_out);
  m.impl("all_reduce__wrapper", &all_reduce__wrapper);
  m.impl("all_reduce_wrapper", &all_reduce_wrapper);
  m.impl("all_gather_into_tensor_wrapper", &all_gather_into_tensor_wrapper);
  m.impl("all_gather_into_tensor_out_wrapper", &all_gather_into_tensor_out_wrapper);
  m.impl("reduce_scatter_wrapper", &reduce_scatter_wrapper);
  m.impl("reduce_scatter_out_wrapper", &reduce_scatter_out_wrapper);
  m.impl("reduce_scatterv_wrapper", &reduce_scatterv_wrapper);
  m.impl("reduce_scatterv_out_wrapper", &reduce_scatterv_out_wrapper);
}

}  // namespace torch_mpi_ext
