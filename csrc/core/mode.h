#pragma once

#include <core/base.h>

namespace quickreduce {

// MODE Register Fields

// Field        | Bit Position | Description
// -------------|--------------|-----------------------------------------------
// FP_ROUND     | 3:0          | Round mode for single, double/half precision.
// FP_DENORM    | 7:4          | Denormal mode for single, double/half precision.
// DX10_CLAMP   | 8            | Clamps NaN to zero in vector ALU.
// IEEE         | 9            | Enables IEEE 754-2008 NaN propagation and quieting.
// LOD_CLAMPED  | 10           | Indicates texture LOD was clamped.
// DEBUG        | 11           | Triggers exception handler after each instruction.
// EXCP_EN      | 18:12        | Enable mask for various exceptions.
// FP16_OVFL    | 23           | Clamps overflowed FP16 results, preserving INF.
// POPS_PACKER0 | 24           | Associates wave with packer 0.
// POPS_PACKER1 | 25           | Associates wave with packer 1.
// DISABLE_PERF | 26           | Disables performance counting for the wave.
// GPR_IDX_EN   | 27           | Enables GPR indexing.
// VSKIP        | 28           | Skips vector instructions rapidly.
// CSP          | 31:29        | Conditional branch stack pointer.

// Instruction to set mode register:
// S_SETREG_IMM32_B32 SIMM16, value;
// Where value is a 32-bit value to set the MODE register to.
// SIMM16 = {size[4:0], offset[4:0], hwRegId[5:0]}; offset is 0..31, size is 1..32.
// hwRegId[5:0] = 0x1 for MODE register.

// Example: setting the FP16_OVFL field to 1:
// S_SETREG_IMM32_B32 {5'b00001, 5'b10111, 6'b000001}, 1;

__device_inline__ static void set_fp16_ovfl(bool const value) {
    // short size = 0b00001;    // Specifies the bit size to modify
    // const short offset = 0b10111;  // Corrected offset to 23, which is the bit position of FP16_OVFL
    // const short hwRegId = 0b000001; // HW register ID for MODE
    // const short simm16 = (size << 11) | (offset << 6) | hwRegId;
    // simm16 = 0xdc1

#if defined(__gfx942__)
    if (value) {
        asm volatile("s_setreg_imm32_b32 0xdc1, 1;"::);
    } else {
        asm volatile("s_setreg_imm32_b32 0xdc1, 0;"::);
    }
#endif
}

}  // namespace quickreduce