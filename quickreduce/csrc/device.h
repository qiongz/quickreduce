#pragma once

#include <torch/extension.h>                // pybind + ATen 基本
#include <ATen/cuda/CUDAContext.h>          // getCurrentCUDAStream / OptionalCUDAGuard
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <ATen/core/ivalue.h>               // c10::ivalue::Future / c10::IValue
#include <hip/hip_runtime_api.h>            // hipIpcMemHandle_t

#include <optional>
#include <thread>
#include <chrono>
#include <vector>
#include <cstring>
#include <iostream>

#include "quickreduce.h"

// quickreduce 句柄管理
quickreduce::fptr_t init(int world_size, int rank, std::optional<int64_t> qr_max_size);
void qr_destroy(quickreduce::fptr_t _fa);

// HIP IPC handle 传递
torch::Tensor get_handle(quickreduce::fptr_t _fa);
void open_handles(quickreduce::fptr_t _fa, const std::vector<torch::Tensor>& handles);

// 同步 allreduce（在给定流上入队）
//void allreduce(quickreduce::fptr_t _fa,
//               torch::Tensor& inp,
 //              int64_t quant_level,
 //              bool cast_bf2half);

// 异步 allreduce：返回 c10::ivalue::Future（Python 看起来是 torch.futures.Future[Tensor]）
c10::intrusive_ptr<c10::ivalue::Future>
allreduce_async(quickreduce::fptr_t fa_addr,
      at::Tensor & tensor, int64_t quant_level, bool cast_bf2half);

int64_t qr_max_size();