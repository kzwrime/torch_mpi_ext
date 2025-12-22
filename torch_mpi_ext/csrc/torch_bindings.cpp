#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include "muladd.h"

// 声明 all_reduce 和 all_gather 函数
void all_reduce_(at::Tensor& input, long comm_ptr);
at::Tensor all_reduce(const at::Tensor& input, long comm_ptr);
at::Tensor all_gather(const at::Tensor& input, long comm_ptr, int64_t dim);

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
  m.def("all_gather(Tensor input, int comm_ptr, int dim = -1) -> Tensor");
}

// Registers CPU implementations for mymuladd, mymul, myadd_out
TORCH_LIBRARY_IMPL(torch_mpi_ext, CPU, m) {
  m.impl("mymuladd", &mymuladd_cpu);
  m.impl("mymul", &mymul_cpu);
  m.impl("myadd_out", &myadd_out_cpu);
  m.impl("all_reduce_", &all_reduce_);
  m.impl("all_reduce", &all_reduce);
  m.impl("all_gather", &all_gather);
}

}  // namespace torch_mpi_ext