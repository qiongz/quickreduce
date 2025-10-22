#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/core/ivalue.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include "device.h"
static pybind11::object allreduce_async_py(quickreduce::fptr_t fa_addr,
                                           at::Tensor& tensor,
                                           int64_t quant_level,
                                           bool cast_bf2half) {
  auto fa = reinterpret_cast<quickreduce::fptr_t>(fa_addr);
  auto fut = allreduce_async(fa, tensor, quant_level, cast_bf2half);
  return torch::jit::toPyObject(c10::IValue(fut));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init", &init);
  m.def("destroy", &destroy);
  m.def("get_handle", &get_handle);
  m.def("open_handles", &open_handles);
  m.def("allreduce", &allreduce);
  m.def("allreduce_async",
        &allreduce_async_py,
        pybind11::arg("fa_addr"),
        pybind11::arg("tensor"),
        pybind11::arg("quant_level"),
        pybind11::arg("cast_bf2half"),
        "Asynchronous quickreduce allreduce, returning torch._C.Future");
}