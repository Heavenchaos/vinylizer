// CUDA render kernels for Vinylizer C++ v7 - Template Sampling Only
// Pipeline: tile-based forward (fused_tile_forward_kernel) + tile_backward (local lTN)
// Old path (compute_hard_alpha + Gaussian blur + composite) has been removed.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <algorithm>

#include "common/alpha_228.h"
#include "cuda/color_utils.cuh"
#include "cuda/erf_coverage.cuh"
#include "cuda/template_sampling.cuh"

#define PI_F 3.14159265358979323846f

using vinylizer::ALPHA_LUT_SIZE;
using vinylizer::C1_228;
using vinylizer::C2_228;
using vinylizer::C3_228;

// 跨编译单元访问 template_sampling.cu 的 __constant__ 变量
extern __constant__ float g_alpha_upper_bound;

// GPU-side alpha_228 using texture LUT (hardware interpolation)
__device__ __forceinline__ float alpha_228_tex(float rn, cudaTextureObject_t tex) {
    if (rn >= 1.0f) return 0.0f;
    float coord = rn * (1.0f - 1.0f / ALPHA_LUT_SIZE) + 0.5f / ALPHA_LUT_SIZE;
    return tex1D<float>(tex, coord);
}

// ============================================================
// Sin/Cos computation kernel
// ============================================================

__global__ void compute_sin_cos_kernel(
    const float* __restrict__ angle,
    float* __restrict__ sina,
    float* __restrict__ cosa,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float rad = angle[i] * PI_F / 180.0f;
    sincosf(rad, &sina[i], &cosa[i]);
}

__global__ void compute_softmax_kernel(
    const float* __restrict__ logits,
    float* __restrict__ softmax_w,
    int N,
    float temperature,
    int num_types)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float invT = 1.0f / temperature;

    // 数值稳定: max-logit subtraction
    float max_l = logits[i * num_types];
    for (int t = 1; t < num_types; t++) {
        max_l = fmaxf(max_l, logits[i * num_types + t]);
    }

    float sum = 0.0f;
    for (int t = 0; t < num_types; t++) {
        float diff = logits[i * num_types + t] - max_l;
        float e = fminf(expf(fminf(diff * invT, 80.0f)), 1e10f);
        softmax_w[i * num_types + t] = e;
        sum += e;
    }
    float inv_sum = 1.0f / fmaxf(sum, 1e-10f);
    for (int t = 0; t < num_types; t++) {
        softmax_w[i * num_types + t] *= inv_sum;
    }
}

__global__ void update_locked_types_kernel(
    const float* __restrict__ softmax_w,
    int* __restrict__ shape_types,
    int N,
    int num_types,
    bool force_lock)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Preserve DARTS_MODE sentinel unless force-locking
    if (!force_lock && shape_types[i] < 0) return;

    int best = 0;
    float best_w = softmax_w[i * num_types + 0];
    for (int t = 1; t < num_types; t++) {
        float w = softmax_w[i * num_types + t];
        if (w > best_w) { best = t; best_w = w; }
    }
    shape_types[i] = best;  // 直接存储模板索引
}

// ============================================================

// ============================================================
// Forward kernel (full-pixel render, used for visualization only)
// ============================================================

__global__ void render_forward_kernel(
    const float* __restrict__ px, const float* __restrict__ py,
    const float* __restrict__ cx, const float* __restrict__ cy,
    const float* __restrict__ rx, const float* __restrict__ ry,
    const float* __restrict__ angle,
    const float* __restrict__ colors,
    const float* __restrict__ opacity,
    const float* __restrict__ bg_color,
    float* __restrict__ output,
    const float* __restrict__ sina,
    const float* __restrict__ cosa,
    int N, int C,
    cudaTextureObject_t alpha_tex)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C) return;
    float pxi = px[idx], pyi = py[idx];
    float rr = 0.0f, gg = 0.0f, bb = 0.0f, T = 1.0f;
    for (int i = N - 1; i >= 0; i--) {
        float coi = cosa[i], sii = sina[i];
        float dx = pxi - cx[i], dy = pyi - cy[i];
        float u = dx * coi + dy * sii, v = -dx * sii + dy * coi;
        float rxc = fmaxf(rx[i], 0.5f), ryc = fmaxf(ry[i], 0.5f);
        float dsq = (u/rxc)*(u/rxc) + (v/ryc)*(v/ryc);
        float rn = sqrtf(fmaxf(dsq, 0.0f));
        float ar = alpha_228_tex(rn, alpha_tex);
        // Logit parameterization: opacity/colors stored as logits; convert to linear at use
        float opi = sigmoid_logit(opacity[i]);
        float clr = 255.0f * sigmoid_logit(colors[i*3+0]);
        float clg = 255.0f * sigmoid_logit(colors[i*3+1]);
        float clb = 255.0f * sigmoid_logit(colors[i*3+2]);
        float ae = ar * opi * g_alpha_upper_bound, w = ae * T;
        // sRGB→linear：颜色以 sRGB 存储，混合须在线性空间进行
        rr += w * srgb_to_linear(clr);
        gg += w * srgb_to_linear(clg);
        bb += w * srgb_to_linear(clb);
        T *= (1.0f - ae);
        if (T < 1e-3f) break;
    }
    rr += T * srgb_to_linear(bg_color[0]);
    gg += T * srgb_to_linear(bg_color[1]);
    bb += T * srgb_to_linear(bg_color[2]);
    // linear→sRGB：输出回到 sRGB 空间
    output[idx*3+0] = linear_to_srgb(rr);
    output[idx*3+1] = linear_to_srgb(gg);
    output[idx*3+2] = linear_to_srgb(bb);
}

// Fill output buffer with background color
__global__ void fill_bg_kernel(
    float* __restrict__ output,
    const float* __restrict__ bg_color,
    int total_pixels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pixels) return;
    output[idx * 3 + 0] = bg_color[0];
    output[idx * 3 + 1] = bg_color[1];
    output[idx * 3 + 2] = bg_color[2];
}

// ============================================================
// C++ API: callable from diff_renderer.cpp
// ============================================================

extern "C" {

// Compute sin/cos on GPU from angle array
void cuda_compute_sin_cos(
    const float* d_angle,
    float* d_sina, float* d_cosa,
    int N, cudaStream_t stream)
{
    if (N == 0) return;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    compute_sin_cos_kernel<<<blocks, threads, 0, stream>>>(
        d_angle, d_sina, d_cosa, N);
}

void cuda_compute_softmax(
    const float* d_logits,
    float* d_softmax_w,
    int N, float temperature, cudaStream_t stream,
    int num_types)
{
    if (N == 0) return;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    compute_softmax_kernel<<<blocks, threads, 0, stream>>>(
        d_logits, d_softmax_w, N, temperature, num_types);
}

void cuda_update_locked_types(
    const float* d_softmax_w,
    int* d_shape_types,
    int N, cudaStream_t stream,
    int num_types,
    bool force_lock)
{
    if (N == 0) return;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    update_locked_types_kernel<<<blocks, threads, 0, stream>>>(
        d_softmax_w, d_shape_types, N, num_types, force_lock);
}

// Forward render (visualization only, no scratch saved)
void cuda_render_forward(
    const float* d_px, const float* d_py,
    const float* d_cx, const float* d_cy,
    const float* d_rx, const float* d_ry,
    const float* d_angle,
    const float* d_colors,
    const float* d_opacity,
    const float* d_bg_color,
    float* d_output,
    const float* d_sina, const float* d_cosa,
    int N, int C, cudaStream_t stream,
    cudaTextureObject_t alpha_tex)
{
    if (N == 0) return;
    int threads = 256;
    int blocks = (C + threads - 1) / threads;
    render_forward_kernel<<<blocks, threads, 0, stream>>>(
        d_px, d_py, d_cx, d_cy, d_rx, d_ry, d_angle,
        d_colors, d_opacity, d_bg_color, d_output,
        d_sina, d_cosa, N, C, alpha_tex);
}

} // extern "C"
