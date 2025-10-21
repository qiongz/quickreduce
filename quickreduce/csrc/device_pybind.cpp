#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/core/ivalue.h>
#include <c10/cuda/CUDACachingAllocator.h>

// 关键：引入把 IValue/Future 转 Python 对象的工具
#include <torch/csrc/jit/python/pybind_utils.h>

#include "device.h"

// 你已有的接口：
// c10::intrusive_ptr<c10::ivalue::Future>
// allreduce_async(quickreduce::fptr_t fa_addr,
//                 at::Tensor& tensor, int64_t quant_level, bool cast_bf2half);

// ★ 新增：Python 友好的包装，入参用 uintptr_t（指针值），返回 torch._C.Future
static pybind11::object allreduce_async_py(quickreduce::fptr_t fa_addr,
                                           at::Tensor& tensor,
                                           int64_t quant_level,
                                           bool cast_bf2half) {
  // 转回 quickreduce::fptr_t
  auto fa = reinterpret_cast<quickreduce::fptr_t>(fa_addr);

  // 调用你已经实现的 C++ 异步 allreduce
  auto fut = allreduce_async(fa, tensor, quant_level, cast_bf2half);

  // 转为 Python 侧的 torch._C.Future
  // 注意：toPyObject 需要 IValue；用 IValue 包住 Future 指针即可
  return torch::jit::toPyObject(c10::IValue(fut));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init", &init);
  m.def("qr_destroy", &qr_destroy);
  m.def("get_handle", &get_handle);
  m.def("open_handles", &open_handles);
  m.def("allreduce_async",
        &allreduce_async_py,
        pybind11::arg("fa_addr"),
        pybind11::arg("tensor"),
        pybind11::arg("quant_level"),
        pybind11::arg("cast_bf2half"),
        "Asynchronous quickreduce allreduce, returning torch._C.Future");
}