#pragma once

#include <torch/torch.h>
#include "quickreduce.h"

quickreduce::fptr_t init(int world_size, int rank);
//void destroy(quickreduce::fptr_t _fa);
torch::Tensor get_handle(quickreduce::fptr_t _fa);
void open_handles(quickreduce::fptr_t _fa,
                     const std::vector<torch::Tensor>& handles);
void allreduce(quickreduce::fptr_t _fa, int64_t profile, torch::Tensor& inp );
