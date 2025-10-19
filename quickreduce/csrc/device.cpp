#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include "device.h"


void Device::init(int world_size, int rank) {
    this->stream = at::cuda::getCurrentHIPStreamMasqueradingAsCUDA();
    this->comms.init(world_size, rank);
}


// ============================================================
// API
// ============================================================
// Unique instance of the Device with explicit ownership.
// Note: Alternatively, we can use a singleton pattern - but this has more cons than pros.
static std::unique_ptr<Device> device_ = std::make_unique<Device>();

std::unique_ptr<Device>& device() {
    return device_;
}

void init(int world_size, int rank) {
    device()->init(world_size, rank);
}

int get_world_size() {
    return device()->comms.get_world_size();
}

int get_rank() {
    return device()->comms.get_rank();
}

comm_handle get_comm_handle() {
    hipIpcMemHandle_t handle = device()->comms.get_handle();

    comm_handle msg;
    std::copy(handle.reserved, handle.reserved + sizeof(hipIpcMemHandle_t), msg.begin());
    return msg;
}

void set_comm_handles(std::vector<comm_handle> const& comm_handles) {
    int world_size = comm_handles.size();
    std::vector<hipIpcMemHandle_t> ipc_handles(world_size);

    for (int i = 0; i < world_size; i++) {
        std::copy(comm_handles[i].begin(), comm_handles[i].end(), ipc_handles[i].reserved);
    }

    device()->comms.open_ipc_handles(ipc_handles);
}

void allreduce(int profile, torch::Tensor & A) {
    device()->comms.allreduce(
        profile,
        device()->stream,
        reinterpret_cast<half*>(A.data_ptr()),
        A.numel());
}
