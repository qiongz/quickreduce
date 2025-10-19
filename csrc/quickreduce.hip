#include <hip/hip_runtime.h>
#include "quickreduce.h"
#include "core/allreduce.h"
#include <optional>

namespace quickreduce {

// ============================================================
// CONTEXT
// ============================================================
void DeviceComms::init(int world_size, int rank, std::optional<int64_t> max_problem_size = std::nullopt) {
    destroy();
    this->world_size = world_size;
    this->rank = rank;
    if (max_problem_size.has_value() && max_problem_size.value() > 0) {
      this->kMaxProblemSize = max_problem_size.value();
    }
    // Allocate buffer size for worst case: F16 2-stage buffer.
    uint32_t flags_buffer_size =
        2 * world_size * kMaxNumBlocks * sizeof(uint32_t);
    static int64_t data_buffer_size = 2 * this->kMaxProblemSize;
    int64_t total_buffer_size = flags_buffer_size + data_buffer_size;
    data_offset = flags_buffer_size;
    HIP_CHECK(hipExtMallocWithFlags((void**)&dbuffer, total_buffer_size,
                                    hipDeviceMallocUncached));

    // Clear the flags buffer.
    HIP_CHECK(hipMemset(dbuffer, 0, flags_buffer_size));

    // Device-side list of IPC buffers.
    buffer_list.resize(world_size);
    HIP_CHECK(hipMalloc(&dbuffer_list, world_size * sizeof(uint8_t*)));

    // Create IPC handles for rank's communication buffer.
    all_buffer_ipc_handles.resize(world_size);
    HIP_CHECK(hipIpcGetMemHandle(&buffer_ipc_handle, dbuffer));

    initialized = true;
}

void DeviceComms::destroy()  {
    if (initialized) {
      for (int i = 0; i < world_size; i++) {
        if (i != rank) {
          HIP_CHECK(hipIpcCloseMemHandle(dbuffer_list[i]));
        }
      }

      HIP_CHECK(hipFree(dbuffer));
      HIP_CHECK(hipFree(dbuffer_list));

      initialized = false;
    }
  }

void DeviceComms::open_ipc_handles(std::vector<hipIpcMemHandle_t> const& ipc_handles) {
    assert(ipc_handles.size() == all_buffer_ipc_handles.size());
    for (int i = 0; i < world_size; i++) {
      all_buffer_ipc_handles[i] = ipc_handles[i];
    }

    // Open device memory access to the IPC communication buffers.
    // Note: For our own rank, we do not need to open a handle.
    for (int i = 0; i < world_size; i++) {
      if (i != rank) {
        HIP_CHECK(hipIpcOpenMemHandle((void**)&buffer_list[i],
                                      all_buffer_ipc_handles[i],
                                      hipIpcMemLazyEnablePeerAccess));
      } else {
        buffer_list[i] = dbuffer;
      }
    }

    HIP_CHECK(hipMemcpy(dbuffer_list, buffer_list.data(),
                        world_size * sizeof(uint8_t*), hipMemcpyHostToDevice));
}

// ============================================================
// KERNEL
// ============================================================

template <typename AllReduceKernel>
__global__ __quickreduce_launch_bounds_two_shot__ static void
allreduce_prototype_twoshot(half * A, uint32_t N, uint32_t num_blocks,
                            int rank, uint8_t** dbuffer_list,
                            uint32_t data_offset, uint32_t flag_color) {
  int block = blockIdx.x;
  int grid = gridDim.x;

  while (block < num_blocks) {
    AllReduceKernel::run(A, N, block, rank, dbuffer_list, data_offset,
                         flag_color);
    block += grid;
    flag_color++;
  }
}

#define TWOSHOT_DISPATCH(__codec)                                           \
  if (world_size == 2) {                                                    \
    using LineCodec = __codec<2>;                                  \
    using AllReduceKernel = AllReduceTwoshot<LineCodec, false>;   \
    hipLaunchKernelGGL((allreduce_prototype_twoshot<AllReduceKernel>),   \
                       dim3(grid), dim3(kBlockTwoShot), 0, stream, A, N, \
                       num_blocks, rank, dbuffer_list, data_offset,         \
                       flag_color);                                         \
  } else if (world_size == 4) {                                             \
    using LineCodec = __codec<4>;                                    \
    using AllReduceKernel = AllReduceTwoshot<LineCodec, false>;   \
    hipLaunchKernelGGL((allreduce_prototype_twoshot<AllReduceKernel>),   \
                       dim3(grid), dim3(kBlockTwoShot), 0, stream, A, N, \
                       num_blocks, rank, dbuffer_list, data_offset,         \
                       flag_color);                                         \
  } else if (world_size == 8) {                                             \
    using LineCodec = __codec<8>;                                   \
    using AllReduceKernel = AllReduceTwoshot<LineCodec, false>;   \
    hipLaunchKernelGGL((allreduce_prototype_twoshot<AllReduceKernel>),   \
                       dim3(grid), dim3(kBlockTwoShot), 0, stream, A, N, \
                       num_blocks, rank, dbuffer_list, data_offset,         \
                       flag_color);                                         \
  }

enum QuickReduceQuantLevel {
  F16 = 0,
  INT8 = 1,
  INT6 = 2,
  INT4 = 3,
};

void DeviceComms::allreduce(half * A, uint32_t N, int quant_level,
                 hipStream_t stream, bool cast_bf2half) {
     if (world_size != 2 && world_size != 4 && world_size != 8) {
      throw std::runtime_error("All Reduce not supported for world_size = " +
                               std::to_string(world_size));
    }

    // Configuration.
    
    uint32_t msg_size = N * sizeof(half);
    uint32_t num_blocks = divceil(msg_size, kTileSize);
    uint32_t grid = min(kMaxNumBlocks, num_blocks);
    auto quant_level_ = static_cast<QuickReduceQuantLevel>(quant_level);
    switch (quant_level_) {
      case QuickReduceQuantLevel::INT8:
        TWOSHOT_DISPATCH(CodecQ8)
        break;
      case QuickReduceQuantLevel::INT6:
        TWOSHOT_DISPATCH(CodecQ6)
        break;
      case QuickReduceQuantLevel::INT4:
        TWOSHOT_DISPATCH(CodecQ4)
        break;
      default:
        TWOSHOT_DISPATCH(CodecFP)
        break;
    }
    HIP_CHECK(cudaGetLastError());

    // -------------------------------------------------
    // Rotate the flag color.
    flag_color++;
}

}  // namespace quickreduce

/*
#define INSTANTIATE_FOR_WORLDSIZE(T, Codec, cast_bf2half)       \
    template struct quickreduce::AllReduceTwoshot<T, Codec<T, 2>, \
                                                  cast_bf2half>;  \
    template struct quickreduce::AllReduceTwoshot<T, Codec<T, 4>, \
                                                  cast_bf2half>;  \
    template struct quickreduce::AllReduceTwoshot<T, Codec<T, 8>, cast_bf2half>;

INSTANTIATE_FOR_WORLDSIZE(quickreduce::nv_bfloat16, quickreduce::CodecFP, false)
INSTANTIATE_FOR_WORLDSIZE(quickreduce::nv_bfloat16, quickreduce::CodecQ4, false)
INSTANTIATE_FOR_WORLDSIZE(quickreduce::nv_bfloat16, quickreduce::CodecQ6, false)
INSTANTIATE_FOR_WORLDSIZE(quickreduce::nv_bfloat16, quickreduce::CodecQ8, false)
INSTANTIATE_FOR_WORLDSIZE(quickreduce::nv_bfloat16, quickreduce::CodecFP, true)
INSTANTIATE_FOR_WORLDSIZE(quickreduce::nv_bfloat16, quickreduce::CodecQ4, true)
INSTANTIATE_FOR_WORLDSIZE(quickreduce::nv_bfloat16, quickreduce::CodecQ6, true)
INSTANTIATE_FOR_WORLDSIZE(quickreduce::nv_bfloat16, quickreduce::CodecQ8, true)

INSTANTIATE_FOR_WORLDSIZE(half, quickreduce::CodecFP, false)
INSTANTIATE_FOR_WORLDSIZE(half, quickreduce::CodecQ4, false)
INSTANTIATE_FOR_WORLDSIZE(half, quickreduce::CodecQ6, false)
INSTANTIATE_FOR_WORLDSIZE(half, quickreduce::CodecQ8, false)
*/