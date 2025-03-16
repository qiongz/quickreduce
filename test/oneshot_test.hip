#include <mpi.h>
#include <iostream>
#include <string>
#include <vector>
#include <rccl/rccl.h>

#include <core/allreduce.h>
#include "test_utils.h"


using namespace quickreduce;

struct Dispatch {
    // 256 threads can move 4096 bytes per atom.
    static int constexpr kAtomSize = 4096;
    static int constexpr kAtoms = 8;
    static int constexpr kTileSize = kAtomSize * kAtoms;

    // Magic maxgrid size. It's just MI300X cores with occupancy=4
    static int constexpr kMaxGrid = 304 * 4;

    static void run(hipStream_t stream, half const* A, half* B, int N, int world_size, int rank, uint8_t** dbuffer_list, long data_offset, int flag_color) {

        int num_blocks = divceil(N * sizeof(half), kTileSize);
        int grid = min(kMaxGrid, num_blocks);

        using AllReduceKernel = quickreduce::AllReduceOneshot;
        all_reduce_kernel<AllReduceKernel><<<grid, kBlock, 0, stream>>>(A, B, N, num_blocks, world_size, rank, dbuffer_list, data_offset, flag_color);
    }
};

int main(int argc, char** argv) {
    int world_size;
    int rank;
    bool bench = false;
    using TB = TestBench<Dispatch, 0, 1>;

    if (argc > 1) {
        bench = std::string(argv[1]) == "bench";
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    hipSetDevice(rank);

    printf("[%d] active\n", rank);
    MPI_Barrier(MPI_COMM_WORLD);

    if (bench) {
        // bench: sweep over problem sizes.
        int N = 2048 * 8;
        for (int k = 0; k < 12; k++) {
            TB bench(N * (1 << k), world_size, rank);
            bench.bench();
            bench.finalize();
        }

    } else {
        // test
        int N = 2048 * 8;
        for (int k = 0; k < 12; k++) {
            TB bench(N * (1 << k), world_size, rank);
            bench.test();
            bench.finalize();
        }

        // Wonky problem size, aligned to the nearest 16B
        for (int k = 1; k < 12; k++) {
            TB bench(k * 1816, world_size, rank);
            bench.test();
            bench.finalize();
        }
    }

    MPI_Finalize();
    return 0;
}
