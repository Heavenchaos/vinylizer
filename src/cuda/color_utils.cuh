#pragma once

#include <cuda_runtime.h>

__device__ __forceinline__ float srgb_to_linear(float c) {
    float cf = c / 255.0f;
    if (cf <= 0.04045f) return cf / 12.92f * 255.0f;
    return powf((cf + 0.055f) / 1.055f, 2.4f) * 255.0f;
}

__device__ __forceinline__ float linear_to_srgb(float c) {
    float cf = fmaxf(c / 255.0f, 0.0f);
    if (cf <= 0.0031308f) return cf * 12.92f * 255.0f;
    return (1.055f * powf(cf, 1.0f / 2.4f) - 0.055f) * 255.0f;
}
