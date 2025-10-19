#pragma once

#include <torch/torch.h>
#include <optional>
#include <torch/all.h>
#include "quickreduce.h"


quickreduce::fptr_t init(int world_size, int rank, std::optional<int64_t> qr_max_size);
//void destroy(quickreduce::fptr_t _fa);
torch::Tensor get_handle(quickreduce::fptr_t _fa);
void open_handles(quickreduce::fptr_t _fa,
                     const std::vector<torch::Tensor>& handles);
void allreduce(quickreduce::fptr_t _fa, torch::Tensor& inp,
                   int64_t quant_level,bool cast_bf2half);
int64_t qr_max_size();
