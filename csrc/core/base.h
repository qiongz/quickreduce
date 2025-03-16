#pragma once

#include <cstdint>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>


#define __device_inline__               __device__ __forceinline__
#define __quickreduce_launch_bounds__   __launch_bounds__(256, 4)


// Setup acquire-release semantics for vector memory reads (mubuf instruction) as per architecture.
#if defined(__gfx942__)
    // CDNA3: Scope bits sc0, sc1
    #define MUBUF_ACQUIRE    16
    #define MUBUF_RELEASE    16
#elif (defined(__gfx908__) || defined(__gfx90a__))
    // CDNA1 and CDNA2 - glc bit
    #define MUBUF_ACQUIRE    1
    #define MUBUF_RELEASE    0
#endif


namespace quickreduce {

// Vector types
using int8x8_t = __attribute__((__vector_size__(8 * sizeof(int8_t)))) int8_t;

using int32x2_t = __attribute__((__vector_size__(2 * sizeof(int)))) int;
using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using int32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;
using int32x16_t = __attribute__((__vector_size__(16 * sizeof(int)))) int;

using fp8_t = uint8_t;
using fp8x8_t = __attribute__((__vector_size__(8 * sizeof(uint8_t)))) uint8_t;

using fp16x4_t  = __attribute__((__vector_size__(4 * sizeof(__fp16)))) __fp16;
using fp16x8_t  = __attribute__((__vector_size__(8 * sizeof(__fp16)))) __fp16;
using fp16x16_t = __attribute__((__vector_size__(16 * sizeof(__fp16)))) __fp16;

using fp32x2_t  = __attribute__((__vector_size__(2 * sizeof(float)))) float;
using fp32x4_t  = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using fp32x8_t  = __attribute__((__vector_size__(8 * sizeof(float)))) float;
using fp32x16_t = __attribute__((__vector_size__(16 * sizeof(float)))) float;


// Standard CDNA wavefront size.
static int constexpr kWavefront = 64;

// 256 thread, 4 wavefronts.
static dim3 constexpr kBlock = {64, 4, 1};


// Methods
__device_inline__ __host__
int divceil(int x, int y) { return ((x + y - 1 ) / y); }

__device_inline__ __host__
constexpr int divceil_constexpr(int const x, int const y) { return ((x + y - 1 ) / y); }

}  // namespace quickreduce

