#pragma once

#include <core/base.h>

namespace quickreduce {

/*
===============================================================
Desc:
    Utility container to describe the Buffer Resource used in VMEM operations.

Operation:
    BufferResource can be initialized to tensor base address and range/size (in bytes).
    The range is used for OOB checks. For example the range for a MxK dtype=fp16 tensor
    would have a range of [M * K * sizeof(half)].

    The last dword of the buffer resource description is to a default config with DFMT=32b.

    Instructions that used the buffer resource (buffer_load/store_dwordx4) wait on the `vmcnt`.
*/

union BufferResource {
    __device_inline__ constexpr BufferResource()
        : config(0x00020000U) {}

    __device_inline__ constexpr BufferResource(void* buffer_address, uint32_t buffer_size)
        : address(buffer_address),
          range(buffer_size),
          config(0x00020000U) {}

    int32x4_t descriptor;
    struct{
        void* address;      // 8B, out of which first 48b is address, and 16b is stride (unused)
        uint32_t range;     // Byte range for the buffer resource
        uint32_t config;    // Constant, DFMT=32b
    };
};

__device_inline__
static int32x4_t buffer_load_dwordx4(int32x4_t srsrc,
                        int32_t voffset,
                        int32_t soffset,
                        int32_t aux) __asm("llvm.amdgcn.raw.buffer.load.v4i32");

__device_inline__
static void buffer_store_dwordx4(int32x4_t data,
                        int32x4_t srsrc,
                        int32_t voffset,
                        int32_t soffset,
                        int32_t aux) __asm("llvm.amdgcn.raw.buffer.store.v4i32");

}  // namespace quickreduce
