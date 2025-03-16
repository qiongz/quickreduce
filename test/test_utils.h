#include <mpi.h>
#include <iostream>
#include <string>
#include <vector>
#include <rccl/rccl.h>


#define NCCL_CHECK(call) \
    do { \
        ncclResult_t err = call; \
        if (err != ncclSuccess) { \
            std::cerr << "NCCL error: " << ncclGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


#define HIP_CHECK(err)                                                              \
    do {                                                                            \
        hipError_t err_ = (err);                                                    \
        if (err_ != hipSuccess) {                                                   \
            std::printf("HIP error %d at %s:%d\n", err_, __FILE__, __LINE__);       \
            throw std::runtime_error("HIP error");                                  \
        }                                                                           \
    } while (0)


static float randf() {
    return ((rand() % 1024) - 512) / 1024.0f;
}


static half randh() {
    return __float2half(randf());
}


// ============================================================
// KERNEL
// ============================================================
template <typename AllReduceKenel>
__global__ __quickreduce_launch_bounds__
static void all_reduce_kernel(half const* A, half* B, int N, int num_blocks,
        int world_size, int rank, uint8_t** dbuffer_list, long data_offset, int flag_color) {

    int block = blockIdx.x;
    int grid = gridDim.x;

    while (block < num_blocks) {
        AllReduceKenel::run(A, B, N, block, num_blocks, world_size, rank, dbuffer_list, data_offset, flag_color);
        block += grid;
    }
}


// ============================================================
// TEST
// ============================================================
template<
    class Dispatch,
    int kInit = 0,  // 0 = Fixed/Strict, 1 = Random
    int kBufferScale = 1
>
struct TestBench {
    // This is setup is fixed for our all reduce implementation.
    // Hence, there will be magic numbers here.

    // 256 threads can move 4096 bytes per atom.
    static int constexpr kAtomSize = 4096;
    static int constexpr kAtoms = 8;
    static int constexpr kTileSize = kAtomSize * kAtoms;

    // Set max problem size as 512MB (in bytes)
    static long constexpr kMaxProblemSize = 536870912;
    static long constexpr kMaxTiles = kMaxProblemSize / kTileSize;

    int rank;
    int world_size;
    int N;
    int flag_color;

    std::vector<half> A;
    std::vector<half> B;
    std::vector<half> C;

    hipStream_t stream;
    half *dA;
    half *dB;
    half *dC;
    uint8_t **dbuffer_list;
    uint8_t *dbuffer;

    hipIpcMemHandle_t buffer_ipc_handle;
    std::vector<hipIpcMemHandle_t> all_buffer_ipc_handles;
    std::vector<uint8_t*> buffer_list;

    long data_offset;

    TestBench(int N, int world_size, int rank)
    : N(N), world_size(world_size), rank(rank), flag_color(1),
      A(N), B(N), C(N),
      all_buffer_ipc_handles(world_size),
      buffer_list(world_size) {
        hipStreamCreate(&stream);

        if (N * sizeof(half) > kMaxProblemSize) {
            std::cerr << "Problem size too large" << std::endl;
            exit(1);
        }

        // ----------------------------------------------------
        // Setup test data.
        srand(42);
        if (kInit == 0) {
            // Basic test pattern to check that numbers add up correctly.
            // Note, the data is tame (integers) because we dont need to deal with the minor differences that arise
            // from rccl adding things in a different order from us, i.e. floats are not commutative.
            // This weird distribution of data is good enough for our purposes.
            for (int i = 0; i < N; ++i) A[i] = __float2half(1.0f * ((rank + i) % 23));
        } else {
            for (int i = 0; i < N; ++i) A[i] = randh();
        }

        // Device memory allocation, and device-side setup of test data.
        hipMalloc(&dA, N * sizeof(half));
        hipMalloc(&dB, N * sizeof(half));
        hipMalloc(&dC, N * sizeof(half));
        hipMalloc(&dbuffer_list, world_size * sizeof(uint8_t*));

        hipMemcpy(dA, A.data(), N * sizeof(half), hipMemcpyHostToDevice);
        hipMemset(dB, 42, N * sizeof(half));
        hipMemset(dC, 42, N * sizeof(half));

        // Setup reference implementation.
        reference();

        // ----------------------------------------------------
        // Setup communication buffers.
        // The first segment of the buffer is the communication flags, followed by the data.
        // OneShot only requires BufferScale = 1, while Twoshot requires BufferScale = 2 (for 2-stage sync interleaving)
        long flags_buffer_size = kBufferScale * world_size * kMaxTiles * sizeof(int);
        long data_buffer_size = kBufferScale * world_size * kMaxProblemSize;
        long total_buffer_size = flags_buffer_size + data_buffer_size;
        HIP_CHECK(hipExtMallocWithFlags((void**)&dbuffer, total_buffer_size, hipDeviceMallocUncached));

        // Clear the flags buffer.
        hipMemset(dbuffer, 0, flags_buffer_size);

        // Set the offset to the start of the data buffer within the communication buffer.
        data_offset = flags_buffer_size;

        // --------------------------------------------------------
        // Create IPC handles for rank's communication buffer.
        hipIpcGetMemHandle(&buffer_ipc_handle, dbuffer);

        // Gather all IPC handles from all ranks.
        MPI_Allgather(&buffer_ipc_handle, sizeof(hipIpcMemHandle_t), MPI_BYTE, all_buffer_ipc_handles.data(), sizeof(hipIpcMemHandle_t), MPI_BYTE, MPI_COMM_WORLD);

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

    ~TestBench() {
        hipFree(dA);
        hipFree(dB);
        hipFree(dC);
        hipFree(dbuffer_list);
        hipFree(dbuffer);
        hipStreamDestroy(stream);
    }

    void finalize() {
        // Sync the ranks to avoid a hazard.
        MPI_Barrier(MPI_COMM_WORLD);

        for (int i = 0; i < world_size; i++) {
            if (i != rank) {
                hipIpcCloseMemHandle(buffer_list[i]);
            }
        }
    }

    void reference() {
        // Reference implementation of the all reduce operation using rccl
        ncclComm_t comm;
        ncclUniqueId nccl_group_id;

        if (rank == 0) {
            NCCL_CHECK(ncclGetUniqueId(&nccl_group_id));
        }

        MPI_Bcast(&nccl_group_id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
        NCCL_CHECK(ncclCommInitRank(&comm, world_size, nccl_group_id, rank));

        NCCL_CHECK(ncclAllReduce(dA, dC, N, ncclFloat16, ncclSum, comm, stream));

        hipStreamSynchronize(stream);
        hipMemcpy(C.data(), dC, N * sizeof(half), hipMemcpyDeviceToHost);

        NCCL_CHECK(ncclCommDestroy(comm));
    }

    void test(float tolerance=0.0f, int trials=8) {
        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0) {
            printf("[%d] Problem Size = %lu / %lu\n", rank, N * sizeof(half), kMaxProblemSize);
        }

        bool test_ok = true;
        float max_error = 0.0f;
        for (int trial = 0; trial < trials; trial++) {
            hipMemsetAsync(dB, 42, N * sizeof(half), stream);

            Dispatch::run(stream, dA, dB, N, world_size, rank, dbuffer_list, data_offset, flag_color);
            flag_color++;

            hipStreamSynchronize(stream);
            hipMemcpy(B.data(), dB, N * sizeof(half), hipMemcpyDeviceToHost);

            for (int i = 0; i < N; ++i) {
                float expected = __half2float(C[i]);
                float actual = __half2float(B[i]);
                float error = abs(actual - expected);
                max_error = fmax(max_error, error);

                if (error > tolerance) {
                    printf("[%d] B[%d] = %f != %f, error = %f\n", rank, i, expected, actual, error);
                    test_ok = false;
                    break;
                }
            }

            if (not test_ok) break;
        }

        printf("[%d] Test: %s, max_error = %f\n", rank, test_ok ? "PASS" : "FAIL", max_error);
    }

    void bench(int trials=128) {
        MPI_Barrier(MPI_COMM_WORLD);

        // Warmup.
        for (int trial = 0; trial < 3; trial++) {
            Dispatch::run(stream, dA, dB, N, world_size, rank, dbuffer_list, data_offset, flag_color);
            flag_color++;
        }

        // bench
        hipEvent_t start, end;
        hipEventCreate(&start);
        hipEventCreate(&end);
        hipEventRecord(start, stream);

        for (int i = 0; i < trials; i++) {
            Dispatch::run(stream, dA, dB, N, world_size, rank, dbuffer_list, data_offset, flag_color);
            flag_color++;
        }

        hipEventRecord(end, stream);
        hipEventSynchronize(end);

        float elapsed_time;
        hipEventElapsedTime(&elapsed_time, start, end);

        float avg_time = elapsed_time / trials;
        if (rank == 0) printf("[%d] Size: %lu, Latency: %f ms, trials = %d\n", rank, N * sizeof(half), avg_time, trials);
    }
};