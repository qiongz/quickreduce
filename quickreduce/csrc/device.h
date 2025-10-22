#pragma once

#include <torch/extension.h>               
#include <ATen/cuda/CUDAContext.h>         
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <ATen/core/ivalue.h>              
#include <hip/hip_runtime_api.h>            

#include <optional>
#include <thread>
#include <chrono>
#include <vector>
#include <cstring>
#include <iostream>

#include "quickreduce.h"


quickreduce::fptr_t init(int world_size, int rank, std::optional<int64_t> qr_max_size);
void destroy(quickreduce::fptr_t _fa);

torch::Tensor get_handle(quickreduce::fptr_t _fa);
void open_handles(quickreduce::fptr_t _fa, const std::vector<torch::Tensor>& handles);

void allreduce(quickreduce::fptr_t _fa,
               at::Tensor& inp,
              int64_t quant_level,
              bool cast_bf2half);

c10::intrusive_ptr<c10::ivalue::Future>
allreduce_async(quickreduce::fptr_t fa_addr,
      at::Tensor & tensor, int64_t quant_level, bool cast_bf2half);

int64_t qr_max_size();