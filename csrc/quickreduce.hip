#include <hip/hip_runtime.h>
#include <core/allreduce.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include "quickreduce.h"

namespace quickreduce {

// ============================================================
// CONTEXT
// ============================================================
void DeviceComms::init(int world_size, int rank) {
    destroy();
    this->stream = at::cuda::getCurrentHIPStreamMasqueradingAsCUDA();
    this->world_size = world_size;
    this->rank = rank;

    // Allocate buffer size for worst case: Twoshot FP16 2-stage buffer.
    long flags_buffer_size = 2 * world_size * kMaxTiles * sizeof(int);
    long data_buffer_size = 2 * kMaxProblemSize;
    long total_buffer_size = flags_buffer_size + data_buffer_size;
    data_offset = flags_buffer_size;
    HIP_CHECK(hipExtMallocWithFlags((void**)&dbuffer, total_buffer_size, hipDeviceMallocUncached));

    // Clear the flags buffer.
    hipMemset(dbuffer, 0, flags_buffer_size);

    // Device-side list of IPC buffers.
    buffer_list.resize(world_size);
    hipMalloc(&dbuffer_list, world_size * sizeof(uint8_t*));

    // Create IPC handles for rank's communication buffer.
    all_buffer_ipc_handles.resize(world_size);
    hipIpcGetMemHandle(&buffer_ipc_handle, dbuffer);

    initialized = true;
}

void DeviceComms::destroy() {
    if (initialized) {
        for (int i = 0; i < world_size; i++) {
            if (i != rank) {
                hipIpcCloseMemHandle(dbuffer_list[i]);
            }
        }

        hipFree(dbuffer);
        hipFree(dbuffer_list);

        initialized = false;
    }
}

void DeviceComms::open_ipc_handles(std::vector<hipIpcMemHandle_t> const& ipc_handles) {
    for (int i = 0; i < world_size; i++) {
        all_buffer_ipc_handles[i] = ipc_handles[i];
    }

    // Open device memory access to the IPC communication buffers.
    // Note: For our own rank, we do not need to open a handle.
    for (int i = 0; i < world_size; i++) {
        if (i != rank) {
            hipIpcOpenMemHandle((void**)&buffer_list[i], all_buffer_ipc_handles[i], hipIpcMemLazyEnablePeerAccess);
        } else {
            buffer_list[i] = dbuffer;
        }
    }

    hipMemcpy(dbuffer_list, buffer_list.data(), world_size * sizeof(uint8_t*), hipMemcpyHostToDevice);
}

// ============================================================
// KERNEL
// ============================================================
template <typename AllReduceKenel>
__global__ __quickreduce_launch_bounds__
static void allreduce_prototype(half * A,  int N, int num_blocks,
        int world_size, int rank, uint8_t** dbuffer_list, long data_offset, int flag_color) {

    int block = blockIdx.x;
    int grid = gridDim.x;

    while (block < num_blocks) {
        AllReduceKenel::run(A, N, block, num_blocks, world_size, rank, dbuffer_list, data_offset, flag_color);
        block += grid;
    }
}

// ============================================================
// DISPATCH
// ============================================================
#define TWOSHOT_DISPATCH(__codec)                                               \
    if (world_size == 2) {                                                      \
        using LineCodec = __codec<2>;                                           \
        using AllReduceKernel = AllReduceTwoshot<LineCodec>;                    \
        hipLaunchKernelGGL((allreduce_prototype<AllReduceKernel>),              \
            dim3(grid), dim3(kBlock), 0, stream,                                \
            A, N, num_blocks, world_size, rank, dbuffer_list,                \
            data_offset, flag_color);                                           \
    }                                                                           \
    else if (world_size == 4) {                                                 \
        using LineCodec = __codec<4>;                                           \
        using AllReduceKernel = AllReduceTwoshot<LineCodec>;                    \
        hipLaunchKernelGGL((allreduce_prototype<AllReduceKernel>),              \
            dim3(grid), dim3(kBlock), 0, stream,                                \
            A, N, num_blocks, world_size, rank, dbuffer_list,                \
            data_offset, flag_color);                                           \
    }                                                                           \
    else if (world_size == 8) {                                                 \
        using LineCodec = __codec<8>;                                           \
        using AllReduceKernel = AllReduceTwoshot<LineCodec>;                    \
        hipLaunchKernelGGL((allreduce_prototype<AllReduceKernel>),              \
            dim3(grid), dim3(kBlock), 0, stream,                                \
            A, N, num_blocks, world_size, rank, dbuffer_list,                \
            data_offset, flag_color);                                           \
    }

void DeviceComms::allreduce(int profile, hipStream_t stream, half * A,  int N) {
    if (world_size != 2 && world_size != 4 && world_size != 8) {
        throw std::runtime_error("All Reduce not supported for world_size = " + std::to_string(world_size));
    }

    // Configuration.
    long msg_size = N * sizeof(half);
    int num_blocks = divceil(msg_size, kTileSize);
    int grid = min(304 * 4, num_blocks);

    // -------------------------------------------------
    // All reduce dispatch.
    QuickReduceProfile dprofile = static_cast<QuickReduceProfile>(profile);

    switch (dprofile) {
        case QuickReduceProfile::TWOSHOT_FP8:
            TWOSHOT_DISPATCH(TwoshotFP8LineCodec)
            break;
        case QuickReduceProfile::TWOSHOT_Q8:
            TWOSHOT_DISPATCH(TwoshotQ8LineCodec)
            break;
        case QuickReduceProfile::TWOSHOT_Q6:
            TWOSHOT_DISPATCH(TwoshotQ6LineCodec)
            break;
        case QuickReduceProfile::TWOSHOT_Q4:
            TWOSHOT_DISPATCH(TwoshotQ4LineCodec)
            break;
        default:
            TWOSHOT_DISPATCH(TwoshotFP16LineCodec)
            break;
    }

    // -------------------------------------------------
    // Rotate the flag color.
    flag_color++;
}

}  // namespace quickreduce
