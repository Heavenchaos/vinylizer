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

// 输入是**编码后的 sRGB 输出值**（0..255），返回 d(srgb_out)/d(linear) 的绝对值
// （即 linear_to_srgb 的导数，二者互为反函数，导数为倒数关系）。
// 前向在**线性空间**合成、输出前经 linear_to_srgb 编码回 sRGB；反向若在线性空间求梯度，
// 就必须除以这一项。否则反向微分的是"在 sRGB 空间直接合成"的另一个函数 ——
// 方向虽不至于反，但每个像素的缩放系数是错的，暗部被压缩、亮部被放大。
// 下限 1e-3：v→0 时导数趋于 12.92（线性段），不会发散；此夹紧只是数值保护。
__device__ __forceinline__ float srgb_encode_deriv(float srgb_out_0_255) {
    float cf = fmaxf(srgb_out_0_255, 0.0f) / 255.0f;
    if (cf <= 0.04045f) return 12.92f;
    return fmaxf((1.055f / 2.4f) * powf((cf + 0.055f) / 1.055f, 1.0f / 2.4f - 1.0f), 1e-3f);
}

