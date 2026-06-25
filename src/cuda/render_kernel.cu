// CUDA render kernels for Vinylizer C++

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <algorithm>

#include "common/alpha_228.h"
#include "cuda/color_utils.cuh"

#define PI_F 3.14159265358979323846f

using vinylizer::ALPHA_LUT_SIZE;
using vinylizer::C1_228;
using vinylizer::C2_228;
using vinylizer::C3_228;

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
        float ae = ar * opacity[i], w = ae * T;
        rr += w * srgb_to_linear(colors[i*3+0]);
        gg += w * srgb_to_linear(colors[i*3+1]);
        bb += w * srgb_to_linear(colors[i*3+2]);
        T *= (1.0f - ae);
        if (T < 1e-3f) break;
    }
    rr += T * srgb_to_linear(bg_color[0]);
    gg += T * srgb_to_linear(bg_color[1]);
    bb += T * srgb_to_linear(bg_color[2]);
    output[idx*3+0] = linear_to_srgb(rr);
    output[idx*3+1] = linear_to_srgb(gg);
    output[idx*3+2] = linear_to_srgb(bb);
}

// ============================================================
// Tile forward kernel (saves alpha/r_norm for backward)
// ============================================================

__global__ void render_tile_forward_kernel(
    const float* __restrict__ px,
    const float* __restrict__ py,
    const float* __restrict__ cx, const float* __restrict__ cy,
    const float* __restrict__ rx, const float* __restrict__ ry,
    const float* __restrict__ angle,
    const float* __restrict__ colors,
    const float* __restrict__ opacity,
    const float* __restrict__ sina, const float* __restrict__ cosa,
    float* __restrict__ alpha_scratch,
    float* __restrict__ r_norm_scratch,
    const int* __restrict__ active_indices,
    int num_active, int N, int tile_count,
    cudaTextureObject_t alpha_tex,
    float* __restrict__ output,
    const float* __restrict__ bg_color,
    float* __restrict__ rendered_alpha)
{
    int local_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_idx >= tile_count) return;
    float pxi = px[local_idx], pyi = py[local_idx];

    // Accumulate shape contributions for this pixel (back-to-front)
    // active_indices are in ascending order; iterate reverse for correct blending
    float rr = 0.0f, gg = 0.0f, bb = 0.0f, T = 1.0f;
    for (int j = num_active - 1; j >= 0; j--) {
        int i = active_indices[j];
        float coi = cosa[i], sii = sina[i];
        float dx = pxi - cx[i], dy = pyi - cy[i];
        float u = dx * coi + dy * sii, v = -dx * sii + dy * coi;
        float rxc = fmaxf(rx[i], 0.5f), ryc = fmaxf(ry[i], 0.5f);
        float dsq = (u/rxc)*(u/rxc) + (v/ryc)*(v/ryc);
        float rn = sqrtf(fmaxf(dsq, 0.0f));
        float ar = alpha_228_tex(rn, alpha_tex);
        alpha_scratch[(long long)i * tile_count + local_idx] = ar;
        r_norm_scratch[(long long)i * tile_count + local_idx] = rn;

        // Accumulate rendering (linear space, matching v5)
        float ae = ar * opacity[i], w = ae * T;
        rr += w * srgb_to_linear(colors[i*3+0]);
        gg += w * srgb_to_linear(colors[i*3+1]);
        bb += w * srgb_to_linear(colors[i*3+2]);
        T *= (1.0f - ae);
        if (T < 1e-3f) break;
    }
    rr += T * srgb_to_linear(bg_color[0]);
    gg += T * srgb_to_linear(bg_color[1]);
    bb += T * srgb_to_linear(bg_color[2]);
    if (output) {
        output[local_idx*3+0] = linear_to_srgb(rr);
        output[local_idx*3+1] = linear_to_srgb(gg);
        output[local_idx*3+2] = linear_to_srgb(bb);
    }
    if (rendered_alpha) {
        rendered_alpha[local_idx] = (1.0f - T) * 255.0f;
    }
}

// ============================================================
// Backward kernel (gradient computation)
// ============================================================

__global__ void render_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ px, const float* __restrict__ py,
    const float* __restrict__ cx, const float* __restrict__ cy,
    const float* __restrict__ rx, const float* __restrict__ ry,
    const float* __restrict__ angle,
    const float* __restrict__ colors,
    const float* __restrict__ opacity,
    const float* __restrict__ bg_color,
    const float* __restrict__ sina,
    const float* __restrict__ cosa,
    const float* __restrict__ alpha_scratch,
    const float* __restrict__ r_norm_scratch,
    float* __restrict__ grad_cx,
    float* __restrict__ grad_cy,
    float* __restrict__ grad_rx,
    float* __restrict__ grad_ry,
    float* __restrict__ grad_angle,
    float* __restrict__ grad_colors,
    float* __restrict__ grad_opacity,
    const int* __restrict__ active_indices,
    int num_active, int N, int tile_count,
    const float* __restrict__ grad_alpha)
{
    int local_idx = blockIdx.x * blockDim.x + threadIdx.x;
    bool valid = local_idx < tile_count;
    float pxi = valid ? px[local_idx] : 0.0f;
    float pyi = valid ? py[local_idx] : 0.0f;
    float gdr = valid ? grad_output[local_idx*3+0] : 0.0f;
    float gdg = valid ? grad_output[local_idx*3+1] : 0.0f;
    float gdb = valid ? grad_output[local_idx*3+2] : 0.0f;
    float gda = (valid && grad_alpha) ? grad_alpha[local_idx] : 0.0f;

    int wid = threadIdx.x >> 5;
    int lid = threadIdx.x & 31;
    int num_warps = blockDim.x >> 5;
    __shared__ float s_block[9 * 32];

    double lTN = 0.0;
    for (int j = 0; j < num_active; j++) {
        int i = active_indices[j];
        float ar = valid ? alpha_scratch[(long long)i * tile_count + local_idx] : 0.0f;
        if (ar <= 0.0f) continue;
        float ae = ar * opacity[i];
        if (ae >= 1.0f) { lTN = -1e10; break; }
        double oma = 1.0 - (double)ae;
        if (oma < 1e-15) oma = 1e-15;
        lTN += log(oma);
    }

    double lTB = 0.0;
    float car = bg_color[0], cag = bg_color[1], cab = bg_color[2];
    for (int j = 0; j < num_active; j++) {
        int i = active_indices[j];
        float ar = valid ? alpha_scratch[(long long)i * tile_count + local_idx] : 0.0f;
        float lcx=0,lcy=0,lrx=0,lry=0,lan=0,lcr=0,lcg=0,lcb=0,lop=0;
        if (ar > 0.0f) {
            float rn = valid ? r_norm_scratch[(long long)i * tile_count + local_idx] : 0.0f;
            float opi = opacity[i], clr = colors[i*3+0], clg = colors[i*3+1], clb = colors[i*3+2];
            float cxi = cx[i], cyi = cy[i], rxi = rx[i], ryi = ry[i];
            float ae = ar * opi;
            double oma = 1.0 - (double)ae;
            if (oma < 1e-15) oma = 1e-15;
            float omaf = (float)oma;
            double lT_i = lTN - lTB - log(oma);
            float Ti = exp((float)lT_i);
            if (Ti > 1.0f) Ti = 1.0f;
            float wi = ae * Ti; lcr=gdr*wi; lcg=gdg*wi; lcb=gdb*wi;
            if (Ti > 1e-6f) {
                float dcdar=Ti*(clr-car), dcdag=Ti*(clg-cag), dcdab=Ti*(clb-cab);
                float dlae = gdr*dcdar + gdg*dcdag + gdb*dcdab + gda*255.0f*Ti;
                float aval = ar;
                float denom = C1_228 + 2.0f*C2_228*aval + 3.0f*C3_228*aval*aval;
                float dadr = -1.0f / fmaxf(denom, 1e-10f);
                float rc = fmaxf(rn, 1e-6f), daddsq = dadr / (2.0f * rc);
                float coi=cosa[i], sii=sina[i];
                float dx = pxi - cxi, dy = pyi - cyi;
                float u=dx*coi+dy*sii, v=-dx*sii+dy*coi;
                float rxc=fmaxf(rxi,0.5f), ryc=fmaxf(ryi,0.5f);
                float rx2=rxc*rxc, ry2=ryc*ryc;
                float dadcx = daddsq * (-2.0f) * (u*coi/rx2 - v*sii/ry2);
                float dadcy = daddsq * (-2.0f) * (u*sii/rx2 + v*coi/ry2);
                float dadrx = daddsq * (-2.0f * u*u / (rxc*rx2));
                float dadry = daddsq * (-2.0f * v*v / (ryc*ry2));
                float dadan = daddsq * 2.0f*u*v * (PI_F/180.0f) * (1.0f/rx2 - 1.0f/ry2);
                lcx=dlae*opi*dadcx; lcy=dlae*opi*dadcy; lrx=dlae*opi*dadrx;
                lry=dlae*opi*dadry; lan=dlae*opi*dadan; lop=dlae*ar;
            }
            car=ae*clr+omaf*car; cag=ae*clg+omaf*cag; cab=ae*clb+omaf*cab;
            lTB += log(oma);
        }
        unsigned warp_active = __activemask();
        #pragma unroll
        for (int off=16; off>0; off>>=1) {
            lcx+=__shfl_down_sync(warp_active,lcx,off); lcy+=__shfl_down_sync(warp_active,lcy,off);
            lrx+=__shfl_down_sync(warp_active,lrx,off); lry+=__shfl_down_sync(warp_active,lry,off);
            lan+=__shfl_down_sync(warp_active,lan,off); lcr+=__shfl_down_sync(warp_active,lcr,off);
            lcg+=__shfl_down_sync(warp_active,lcg,off); lcb+=__shfl_down_sync(warp_active,lcb,off);
            lop+=__shfl_down_sync(warp_active,lop,off);
        }
        if (lid == 0) {
            s_block[wid * 9 + 0] = lcx; s_block[wid * 9 + 1] = lcy;
            s_block[wid * 9 + 2] = lrx; s_block[wid * 9 + 3] = lry;
            s_block[wid * 9 + 4] = lan;
            s_block[wid * 9 + 5] = lcr; s_block[wid * 9 + 6] = lcg;
            s_block[wid * 9 + 7] = lcb; s_block[wid * 9 + 8] = lop;
        }
        __syncthreads();

        if (wid == 0) {
            float blcx = (lid < num_warps) ? s_block[lid * 9 + 0] : 0.0f;
            float blcy = (lid < num_warps) ? s_block[lid * 9 + 1] : 0.0f;
            float blrx = (lid < num_warps) ? s_block[lid * 9 + 2] : 0.0f;
            float blry = (lid < num_warps) ? s_block[lid * 9 + 3] : 0.0f;
            float blan = (lid < num_warps) ? s_block[lid * 9 + 4] : 0.0f;
            float blcr = (lid < num_warps) ? s_block[lid * 9 + 5] : 0.0f;
            float blcg = (lid < num_warps) ? s_block[lid * 9 + 6] : 0.0f;
            float blcb = (lid < num_warps) ? s_block[lid * 9 + 7] : 0.0f;
            float blop = (lid < num_warps) ? s_block[lid * 9 + 8] : 0.0f;

            unsigned bmask = __activemask();
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                blcx += __shfl_down_sync(bmask, blcx, off);
                blcy += __shfl_down_sync(bmask, blcy, off);
                blrx += __shfl_down_sync(bmask, blrx, off);
                blry += __shfl_down_sync(bmask, blry, off);
                blan += __shfl_down_sync(bmask, blan, off);
                blcr += __shfl_down_sync(bmask, blcr, off);
                blcg += __shfl_down_sync(bmask, blcg, off);
                blcb += __shfl_down_sync(bmask, blcb, off);
                blop += __shfl_down_sync(bmask, blop, off);
            }

            if (lid == 0) {
                atomicAdd(&grad_cx[i], blcx); atomicAdd(&grad_cy[i], blcy);
                atomicAdd(&grad_rx[i], blrx); atomicAdd(&grad_ry[i], blry);
                atomicAdd(&grad_angle[i], blan);
                atomicAdd(&grad_colors[i*3+0], blcr);
                atomicAdd(&grad_colors[i*3+1], blcg);
                atomicAdd(&grad_colors[i*3+2], blcb);
                atomicAdd(&grad_opacity[i], blop);
            }
        }
        __syncthreads();
    }
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

// Tile forward only (produces d_output + scratch for backward)
void cuda_render_tile_forward(
    const float* d_px, const float* d_py,
    const float* d_cx, const float* d_cy,
    const float* d_rx, const float* d_ry,
    const float* d_angle,
    const float* d_colors,
    const float* d_opacity,
    const float* d_bg_color,
    const float* d_sina, const float* d_cosa,
    float* d_alpha_scratch,
    float* d_r_norm_scratch,
    float* d_output,
    const float* h_cx, const float* h_cy,
    const float* h_rx, const float* h_ry,
    const float* h_px, const float* h_py,
    int* d_active_indices,
    int N, int total_pixels, int tile_pixels,
    cudaStream_t stream,
    cudaTextureObject_t alpha_tex,
    const float* tile_tx1, const float* tile_tx2,
    const float* tile_ty1, const float* tile_ty2,
    int num_tiles,
    float* d_rendered_alpha)
{
    const int threads = 256;

    // Fill output with background color first (tiles only overwrite active pixels)
    int bg_blocks = (total_pixels + threads - 1) / threads;
    fill_bg_kernel<<<bg_blocks, threads, 0, stream>>>(d_output, d_bg_color, total_pixels);

    // Compute shape bounding radii on CPU
    std::vector<float> shape_R(N);
    for (int i = 0; i < N; i++) {
        shape_R[i] = std::max(h_rx[i], h_ry[i]);
    }

    // Pre-allocate active indices buffer (avoid per-tile heap allocation)
    std::vector<int> active_vec;
    active_vec.reserve(N);

    for (int t = 0; t < num_tiles; t++) {
        int p_start = t * tile_pixels;
        int p_count = (p_start + tile_pixels <= total_pixels) ? tile_pixels : (total_pixels - p_start);

        // Use precomputed tile AABB
        float tx1 = tile_tx1[t], tx2 = tile_tx2[t];
        float ty1 = tile_ty1[t], ty2 = tile_ty2[t];

        // Filter active shapes
        active_vec.clear();
        for (int i = 0; i < N; i++) {
            float R = shape_R[i];
            if (h_cx[i] + R > tx1 && h_cx[i] - R < tx2 &&
                h_cy[i] + R > ty1 && h_cy[i] - R < ty2) {
                active_vec.push_back(i);
            }
        }

        int na = static_cast<int>(active_vec.size());
        if (na == 0) continue;

        // Upload active indices to GPU
        cudaMemcpyAsync(d_active_indices, active_vec.data(),
                        na * sizeof(int), cudaMemcpyHostToDevice, stream);

        int blocks = (p_count + threads - 1) / threads;

        // Tile forward (also accumulates rendering output)
        render_tile_forward_kernel<<<blocks, threads, 0, stream>>>(
            d_px + p_start, d_py + p_start,
            d_cx, d_cy, d_rx, d_ry, d_angle,
            d_colors, d_opacity,
            d_sina, d_cosa,
            d_alpha_scratch, d_r_norm_scratch,
            d_active_indices, na, N, p_count,
            alpha_tex,
            d_output + p_start * 3,
            d_bg_color,
            d_rendered_alpha ? d_rendered_alpha + p_start : nullptr);
    }
}

// Tile backward only (uses scratch from forward pass)
void cuda_render_tile_backward(
    const float* d_px, const float* d_py,
    const float* d_cx, const float* d_cy,
    const float* d_rx, const float* d_ry,
    const float* d_angle,
    const float* d_colors,
    const float* d_opacity,
    const float* d_bg_color,
    const float* d_grad_output,
    const float* d_sina, const float* d_cosa,
    const float* d_alpha_scratch,
    const float* d_r_norm_scratch,
    float* d_grad_cx, float* d_grad_cy,
    float* d_grad_rx, float* d_grad_ry,
    float* d_grad_angle,
    float* d_grad_colors,
    float* d_grad_opacity,
    const float* h_cx, const float* h_cy,
    const float* h_rx, const float* h_ry,
    const float* h_px, const float* h_py,
    int* d_active_indices,
    int N, int total_pixels, int tile_pixels,
    cudaStream_t stream,
    const float* tile_tx1, const float* tile_tx2,
    const float* tile_ty1, const float* tile_ty2,
    int num_tiles,
    const float* d_grad_alpha)
{
    const int threads = 256;

    // Compute shape bounding radii on CPU
    std::vector<float> shape_R(N);
    for (int i = 0; i < N; i++) {
        shape_R[i] = std::max(h_rx[i], h_ry[i]);
    }

    // Pre-allocate active indices buffer
    std::vector<int> active_vec;
    active_vec.reserve(N);

    for (int t = 0; t < num_tiles; t++) {
        int p_start = t * tile_pixels;
        int p_count = (p_start + tile_pixels <= total_pixels) ? tile_pixels : (total_pixels - p_start);

        // Use precomputed tile AABB
        float tx1 = tile_tx1[t], tx2 = tile_tx2[t];
        float ty1 = tile_ty1[t], ty2 = tile_ty2[t];

        // Filter active shapes
        active_vec.clear();
        for (int i = 0; i < N; i++) {
            float R = shape_R[i];
            if (h_cx[i] + R > tx1 && h_cx[i] - R < tx2 &&
                h_cy[i] + R > ty1 && h_cy[i] - R < ty2) {
                active_vec.push_back(i);
            }
        }

        int na = static_cast<int>(active_vec.size());
        if (na == 0) continue;

        // Upload active indices to GPU
        cudaMemcpyAsync(d_active_indices, active_vec.data(),
                        na * sizeof(int), cudaMemcpyHostToDevice, stream);

        int blocks = (p_count + threads - 1) / threads;

        // Backward
        render_backward_kernel<<<blocks, threads, 0, stream>>>(
            d_grad_output + p_start * 3,
            d_px + p_start, d_py + p_start,
            d_cx, d_cy, d_rx, d_ry, d_angle,
            d_colors, d_opacity, d_bg_color,
            d_sina, d_cosa,
            d_alpha_scratch, d_r_norm_scratch,
            d_grad_cx, d_grad_cy,
            d_grad_rx, d_grad_ry,
            d_grad_angle, d_grad_colors, d_grad_opacity,
            d_active_indices, na, N, p_count,
            d_grad_alpha ? d_grad_alpha + p_start : nullptr);
    }
}

// Per-tile forward→backward (re-forwards for scratch, then backward)
// Scratch buffer is sized for one tile; must process tile-by-tile.
void cuda_render_tile_forward_backward_per_tile(
    const float* d_px, const float* d_py,
    const float* d_cx, const float* d_cy,
    const float* d_rx, const float* d_ry,
    const float* d_angle,
    const float* d_colors,
    const float* d_opacity,
    const float* d_bg_color,
    const float* d_grad_output,
    const float* d_sina, const float* d_cosa,
    float* d_alpha_scratch,
    float* d_r_norm_scratch,
    float* d_grad_cx, float* d_grad_cy,
    float* d_grad_rx, float* d_grad_ry,
    float* d_grad_angle,
    float* d_grad_colors,
    float* d_grad_opacity,
    const float* h_cx, const float* h_cy,
    const float* h_rx, const float* h_ry,
    const float* h_px, const float* h_py,
    int* d_active_indices,
    int N, int total_pixels, int tile_pixels,
    cudaStream_t stream,
    cudaTextureObject_t alpha_tex,
    const float* tile_tx1, const float* tile_tx2,
    const float* tile_ty1, const float* tile_ty2,
    int num_tiles,
    const float* d_grad_alpha)
{
    const int threads = 256;

    std::vector<float> shape_R(N);
    for (int i = 0; i < N; i++) {
        shape_R[i] = std::max(h_rx[i], h_ry[i]);
    }
    std::vector<int> active_vec;
    active_vec.reserve(N);

    for (int t = 0; t < num_tiles; t++) {
        int p_start = t * tile_pixels;
        int p_count = (p_start + tile_pixels <= total_pixels) ? tile_pixels : (total_pixels - p_start);

        float tx1 = tile_tx1[t], tx2 = tile_tx2[t];
        float ty1 = tile_ty1[t], ty2 = tile_ty2[t];

        active_vec.clear();
        for (int i = 0; i < N; i++) {
            float R = shape_R[i];
            if (h_cx[i] + R > tx1 && h_cx[i] - R < tx2 &&
                h_cy[i] + R > ty1 && h_cy[i] - R < ty2) {
                active_vec.push_back(i);
            }
        }

        int na = static_cast<int>(active_vec.size());
        if (na == 0) continue;

        cudaMemcpyAsync(d_active_indices, active_vec.data(),
                        na * sizeof(int), cudaMemcpyHostToDevice, stream);

        int blocks = (p_count + threads - 1) / threads;

        // Tile forward: populate scratch for this tile
        render_tile_forward_kernel<<<blocks, threads, 0, stream>>>(
            d_px + p_start, d_py + p_start,
            d_cx, d_cy, d_rx, d_ry, d_angle,
            d_colors, d_opacity,
            d_sina, d_cosa,
            d_alpha_scratch, d_r_norm_scratch,
            d_active_indices, na, N, p_count,
            alpha_tex,
            nullptr, // output not needed here
            d_bg_color,
            nullptr); // rendered_alpha not needed in backward pass

        // Tile backward: read scratch, accumulate gradients
        render_backward_kernel<<<blocks, threads, 0, stream>>>(
            d_grad_output + p_start * 3,
            d_px + p_start, d_py + p_start,
            d_cx, d_cy, d_rx, d_ry, d_angle,
            d_colors, d_opacity, d_bg_color,
            d_sina, d_cosa,
            d_alpha_scratch, d_r_norm_scratch,
            d_grad_cx, d_grad_cy,
            d_grad_rx, d_grad_ry,
            d_grad_angle, d_grad_colors, d_grad_opacity,
            d_active_indices, na, N, p_count,
            d_grad_alpha ? d_grad_alpha + p_start : nullptr);
    }
}

// ============================================================
// Per-tile visibility accumulation kernel (for relocation)
// ============================================================

__global__ void render_tile_visibility_kernel(
    const float* __restrict__ alpha_scratch,
    const float* __restrict__ opacity,
    const int* __restrict__ active_indices,
    int num_active, int N, int tile_count,
    float* __restrict__ visibility)
{
    int local_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_idx >= tile_count) return;

    float T = 1.0f;
    for (int j = num_active - 1; j >= 0; j--) {
        int i = active_indices[j];
        float ar = alpha_scratch[(long long)i * tile_count + local_idx];
        if (ar <= 0.0f) continue;
        float ae = ar * opacity[i];
        if (ae > 0.0f) {
            atomicAdd(&visibility[i], ae * T);
            T *= (1.0f - ae);
            if (T < 1e-6f) break;
        }
    }
}

void cuda_render_tile_visibility(
    const float* d_px, const float* d_py,
    const float* d_cx, const float* d_cy,
    const float* d_rx, const float* d_ry,
    const float* d_angle,
    const float* d_colors,
    const float* d_opacity,
    const float* d_bg_color,
    const float* d_sina, const float* d_cosa,
    float* d_alpha_scratch,
    float* d_r_norm_scratch,
    float* d_visibility,
    const float* h_cx, const float* h_cy,
    const float* h_rx, const float* h_ry,
    const float* h_px, const float* h_py,
    int* d_active_indices,
    int N, int total_pixels, int tile_pixels,
    cudaStream_t stream,
    cudaTextureObject_t alpha_tex,
    const float* tile_tx1, const float* tile_tx2,
    const float* tile_ty1, const float* tile_ty2,
    int num_tiles)
{
    const int threads = 256;

    std::vector<float> shape_R(N);
    for (int i = 0; i < N; i++) {
        shape_R[i] = std::max(h_rx[i], h_ry[i]);
    }
    std::vector<int> active_vec;
    active_vec.reserve(N);

    for (int t = 0; t < num_tiles; t++) {
        int p_start = t * tile_pixels;
        int p_count = (p_start + tile_pixels <= total_pixels) ? tile_pixels : (total_pixels - p_start);

        float tx1 = tile_tx1[t], tx2 = tile_tx2[t];
        float ty1 = tile_ty1[t], ty2 = tile_ty2[t];

        active_vec.clear();
        for (int i = 0; i < N; i++) {
            float R = shape_R[i];
            if (h_cx[i] + R > tx1 && h_cx[i] - R < tx2 &&
                h_cy[i] + R > ty1 && h_cy[i] - R < ty2) {
                active_vec.push_back(i);
            }
        }

        int na = static_cast<int>(active_vec.size());
        if (na == 0) continue;

        cudaMemcpyAsync(d_active_indices, active_vec.data(),
                        na * sizeof(int), cudaMemcpyHostToDevice, stream);

        int blocks = (p_count + threads - 1) / threads;

        render_tile_forward_kernel<<<blocks, threads, 0, stream>>>(
            d_px + p_start, d_py + p_start,
            d_cx, d_cy, d_rx, d_ry, d_angle,
            d_colors, d_opacity,
            d_sina, d_cosa,
            d_alpha_scratch, d_r_norm_scratch,
            d_active_indices, na, N, p_count,
            alpha_tex,
            nullptr, d_bg_color,
            nullptr); // rendered_alpha not needed for visibility

        render_tile_visibility_kernel<<<blocks, threads, 0, stream>>>(
            d_alpha_scratch, d_opacity,
            d_active_indices, na, N, p_count,
            d_visibility);
    }
}

} // extern "C"
