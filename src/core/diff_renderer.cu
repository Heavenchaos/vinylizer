#include "core/diff_renderer.h"
#include "common/logger.h"
#include "common/alpha_228.h"

#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include <cstring>

// Forward declarations for CUDA kernel entry points (in render_kernel.cu)
extern "C" {

void cuda_compute_sin_cos(
    const float* d_angle,
    float* d_sina, float* d_cosa,
    int N, cudaStream_t stream);

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
    cudaTextureObject_t alpha_tex);

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
    float* d_rendered_alpha);

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
    const float* d_grad_alpha);

// Per-tile forward→backward (re-forward for scratch, then backward)
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
    const float* d_grad_alpha);

// Per-tile forward + visibility accumulation (for relocation scan)
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
    int num_tiles);
}

// Simple CUDA kernel for MSE gradient computation
// RGBA version: includes alpha channel loss for transparency fitting
__global__ void mse_grad_kernel(
    const float* __restrict__ output,     // [C, 3]
    const float* __restrict__ target,     // [C, 3]
    const float* __restrict__ mask,       // [C]
    float* __restrict__ grad_output,      // [C, 3]
    float* __restrict__ mse_sum,          // [1] atomic result
    int C,
    const float* __restrict__ rendered_alpha,  // [C] rendered alpha (0-255)
    const float* __restrict__ target_alpha,    // [C] target alpha (0 or 255)
    float* __restrict__ grad_alpha)            // [C] alpha gradient output
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C) return;

    float m = mask[idx];

    // RGB gradient: masked (transparent pixels have no meaningful RGB target)
    if (m < 0.5f) {
        grad_output[idx * 3 + 0] = 0.0f;
        grad_output[idx * 3 + 1] = 0.0f;
        grad_output[idx * 3 + 2] = 0.0f;
    } else {
        float dr = output[idx * 3 + 0] - target[idx * 3 + 0];
        float dg = output[idx * 3 + 1] - target[idx * 3 + 1];
        float db = output[idx * 3 + 2] - target[idx * 3 + 2];

        float sq_err = dr * dr + dg * dg + db * db;
        atomicAdd(mse_sum, sq_err);

        float scale = 2.0f / (C * 4.0f);  // 4 channels (RGBA)
        grad_output[idx * 3 + 0] = dr * scale;
        grad_output[idx * 3 + 1] = dg * scale;
        grad_output[idx * 3 + 2] = db * scale;
    }

    // Alpha gradient: always computed (not masked by opaque mask)
    if (rendered_alpha && target_alpha && grad_alpha) {
        float da = rendered_alpha[idx] - target_alpha[idx];
        atomicAdd(mse_sum, da * da);
        float scale = 2.0f / (C * 4.0f);  // same scale as RGB
        grad_alpha[idx] = da * scale;
    } else if (grad_alpha) {
        grad_alpha[idx] = 0.0f;
    }
}

// Convert float RGB [0-255] canvas to uint8 RGB
__global__ void float_to_uint8_kernel(
    const float* __restrict__ input,   // [total_pixels * 3] float RGB
    uint8_t* __restrict__ output,      // [total_pixels * 3] uint8 RGB
    int total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    output[idx] = static_cast<uint8_t>(min(255.0f, max(0.0f, input[idx])));
}

// Pack SoA gradients into AoS layout [N*9] + optionally compute per-shape geometric norm
__global__ void pack_and_norm_grads_kernel(
    const float* __restrict__ grad_cx,
    const float* __restrict__ grad_cy,
    const float* __restrict__ grad_rx,
    const float* __restrict__ grad_ry,
    const float* __restrict__ grad_angle,
    const float* __restrict__ grad_colors,  // [N*3]
    const float* __restrict__ grad_opacity,
    float* __restrict__ grads_packed,       // [N*9]
    int N,
    float* grad_history,        // [window * N], nullptr = skip
    int write_idx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float gcx = grad_cx[i];
    float gcy = grad_cy[i];
    float grx = grad_rx[i];
    float gry = grad_ry[i];
    grads_packed[i * 9 + 0] = gcx;
    grads_packed[i * 9 + 1] = gcy;
    grads_packed[i * 9 + 2] = grx;
    grads_packed[i * 9 + 3] = gry;
    grads_packed[i * 9 + 4] = grad_angle[i];
    grads_packed[i * 9 + 5] = grad_colors[i * 3 + 0];
    grads_packed[i * 9 + 6] = grad_colors[i * 3 + 1];
    grads_packed[i * 9 + 7] = grad_colors[i * 3 + 2];
    grads_packed[i * 9 + 8] = grad_opacity[i];
    if (grad_history) {
        grad_history[write_idx * N + i] = sqrtf(gcx*gcx + gcy*gcy + grx*grx + gry*gry);
    }
}

// Fused Adam update + gradient clip + parameter clamp + STE quantize
// Operates on GPU-resident parameter and Adam state buffers
__global__ void fused_adam_kernel(
    // Gradient (input, may be modified in-place for clipping)
    float* __restrict__ grads_packed,       // [N*9]
    // Parameters (input/output)
    float* __restrict__ d_cx, float* __restrict__ d_cy,
    float* __restrict__ d_rx, float* __restrict__ d_ry,
    float* __restrict__ d_angle,
    float* __restrict__ d_colors,           // [N*3]
    float* __restrict__ d_opacity,
    // Adam state (input/output)
    float* __restrict__ m_cx, float* __restrict__ m_cy,
    float* __restrict__ m_rx, float* __restrict__ m_ry,
    float* __restrict__ m_angle,
    float* __restrict__ m_cr, float* __restrict__ m_cg, float* __restrict__ m_cb, float* __restrict__ m_ca,
    float* __restrict__ v_cx, float* __restrict__ v_cy,
    float* __restrict__ v_rx, float* __restrict__ v_ry,
    float* __restrict__ v_angle,
    float* __restrict__ v_cr, float* __restrict__ v_cg, float* __restrict__ v_cb, float* __restrict__ v_ca,
    // STE output (for rendering)
    float* __restrict__ ste_cx, float* __restrict__ ste_cy,
    float* __restrict__ ste_rx, float* __restrict__ ste_ry,
    float* __restrict__ ste_angle,
    float* __restrict__ ste_colors,         // [N*3]
    float* __restrict__ ste_opacity,
    // Config
    int N, int adam_step,
    float lr, float beta1, float beta2, float eps,
    float max_grad_norm,
    int W, int H,
    // Frozen mask (optional, nullptr if not used)
    const int* __restrict__ frozen_mask)    // [N] 1=frozen, 0=active
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Check if frozen
    if (frozen_mask && frozen_mask[i]) return;

    float* g = &grads_packed[i * 9];

    // Gradient clipping is done globally (separate kernel), skip per-shape here

    // Adam update
    float bias1 = 1.0f - powf(beta1, (float)adam_step);
    float bias2 = 1.0f - powf(beta2, (float)adam_step);

    auto adam_update = [&](float& param, float grad, float& m, float& v, float lr_factor) {
        m = beta1 * m + (1.0f - beta1) * grad;
        v = beta2 * v + (1.0f - beta2) * grad * grad;
        float m_hat = m / bias1;
        float v_hat = v / bias2;
        param -= lr * lr_factor * m_hat / (sqrtf(v_hat) + eps);
    };

    adam_update(d_cx[i],    g[0], m_cx[i],    v_cx[i],    1.0f);
    adam_update(d_cy[i],    g[1], m_cy[i],    v_cy[i],    1.0f);
    adam_update(d_rx[i],    g[2], m_rx[i],    v_rx[i],    1.0f);
    adam_update(d_ry[i],    g[3], m_ry[i],    v_ry[i],    1.0f);
    adam_update(d_angle[i], g[4], m_angle[i], v_angle[i], 1.0f);
    adam_update(d_colors[i*3+0], g[5], m_cr[i], v_cr[i], 1.0f);
    adam_update(d_colors[i*3+1], g[6], m_cg[i], v_cg[i], 1.0f);
    adam_update(d_colors[i*3+2], g[7], m_cb[i], v_cb[i], 1.0f);
    adam_update(d_opacity[i],     g[8], m_ca[i], v_ca[i], 1.0f);

    // Parameter clamping
    d_cx[i] = fmaxf(0.0f, fminf(d_cx[i], (float)(W - 1)));
    d_cy[i] = fmaxf(0.0f, fminf(d_cy[i], (float)(H - 1)));
    d_rx[i] = fmaxf(1.0f, fminf(d_rx[i], (float)W));
    d_ry[i] = fmaxf(1.0f, fminf(d_ry[i], (float)H));
    d_angle[i] = fmodf(d_angle[i], 360.0f);
    if (d_angle[i] < 0) d_angle[i] += 360.0f;
    d_colors[i*3+0] = fmaxf(0.0f, fminf(d_colors[i*3+0], 255.0f));
    d_colors[i*3+1] = fmaxf(0.0f, fminf(d_colors[i*3+1], 255.0f));
    d_colors[i*3+2] = fmaxf(0.0f, fminf(d_colors[i*3+2], 255.0f));
    d_opacity[i] = fmaxf(0.0f, fminf(d_opacity[i], 1.0f));

    // STE quantize: copy clamped params, round colors
    ste_cx[i] = d_cx[i];
    ste_cy[i] = d_cy[i];
    ste_rx[i] = d_rx[i];
    ste_ry[i] = d_ry[i];
    ste_angle[i] = d_angle[i];
    ste_colors[i*3+0] = roundf(d_colors[i*3+0]);
    ste_colors[i*3+1] = roundf(d_colors[i*3+1]);
    ste_colors[i*3+2] = roundf(d_colors[i*3+2]);
    ste_opacity[i] = roundf(d_opacity[i] * 255.0f) / 255.0f;
}

// Global gradient norm clipping kernel
__global__ void clip_grad_norm_kernel_v2(
    float* __restrict__ grads,     // [N*8]
    int total_elements,
    float max_norm,
    float* __restrict__ norm_sq)   // [1] atomic result
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    float g = grads[idx];
    atomicAdd(norm_sq, g * g);
}

__global__ void scale_grads_kernel_v2(
    float* __restrict__ grads,
    int total_elements,
    float scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    grads[idx] *= scale;
}

namespace vinylizer {

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            VIN_ERROR("CUDA error at %s:%d: %s", __FILE__, __LINE__, \
                      cudaGetErrorString(err)); \
        } \
    } while (0)

#ifdef _DEBUG
#define CUDA_KERNEL_CHECK() \
    do { \
        cudaError_t _err = cudaGetLastError(); \
        if (_err != cudaSuccess) { \
            VIN_ERROR("CUDA kernel error at %s:%d: %s", __FILE__, __LINE__, \
                      cudaGetErrorString(_err)); \
        } \
    } while (0)
#else
#define CUDA_KERNEL_CHECK() ((void)0)
#endif

DiffRenderer::~DiffRenderer() {
    if (!initialized_) return;

    auto& s = scratch_;
    cudaStreamDestroy(s.stream);

    // Free device memory
    cudaFree(s.d_px); cudaFree(s.d_py);
    cudaFree(s.d_cx); cudaFree(s.d_cy);
    cudaFree(s.d_rx); cudaFree(s.d_ry);
    cudaFree(s.d_angle); cudaFree(s.d_colors);
    cudaFree(s.d_opacity); cudaFree(s.d_sina); cudaFree(s.d_cosa);
    cudaFree(s.d_bg_color); cudaFree(s.d_output);
    cudaFree(s.d_grad_cx); cudaFree(s.d_grad_cy);
    cudaFree(s.d_grad_rx); cudaFree(s.d_grad_ry);
    cudaFree(s.d_grad_angle); cudaFree(s.d_grad_colors);
    cudaFree(s.d_grad_opacity); cudaFree(s.d_grad_output);
    cudaFree(s.d_grads_packed);

    // GPU-resident Adam state
    cudaFree(s.d_m_cx); cudaFree(s.d_m_cy);
    cudaFree(s.d_m_rx); cudaFree(s.d_m_ry);
    cudaFree(s.d_m_angle);
    cudaFree(s.d_m_cr); cudaFree(s.d_m_cg); cudaFree(s.d_m_cb); cudaFree(s.d_m_ca);
    cudaFree(s.d_v_cx); cudaFree(s.d_v_cy);
    cudaFree(s.d_v_rx); cudaFree(s.d_v_ry);
    cudaFree(s.d_v_angle);
    cudaFree(s.d_v_cr); cudaFree(s.d_v_cg); cudaFree(s.d_v_cb); cudaFree(s.d_v_ca);

    // STE output
    cudaFree(s.d_ste_cx); cudaFree(s.d_ste_cy);
    cudaFree(s.d_ste_rx); cudaFree(s.d_ste_ry);
    cudaFree(s.d_ste_angle);
    cudaFree(s.d_ste_colors); cudaFree(s.d_ste_opacity);

    // Frozen mask and gradient norm accumulator
    cudaFree(s.d_frozen_mask);
    cudaFree(s.d_grad_norm_sq);
    cudaFree(s.d_alpha_scratch); cudaFree(s.d_r_norm_scratch);
    cudaFree(s.d_active_indices);
    cudaFree(s.d_active_indices_per_tile);
    cudaFree(s.d_active_count_per_tile);
    cudaFree(s.d_target); cudaFree(s.d_mask);
    cudaFree(s.d_target_alpha); cudaFree(s.d_rendered_alpha); cudaFree(s.d_grad_alpha);
    cudaFree(s.d_visibility);
    cudaFree(s.d_canvas_u8);
    cudaFree(s.d_mse_sum);

    // Alpha LUT texture cleanup
    if (s.alpha_tex) cudaDestroyTextureObject(s.alpha_tex);
    if (s.alpha_array) cudaFreeArray(s.alpha_array);

    // Free pinned host memory
    cudaFreeHost(s.h_cx); cudaFreeHost(s.h_cy);
    cudaFreeHost(s.h_rx); cudaFreeHost(s.h_ry);
    cudaFreeHost(s.h_px); cudaFreeHost(s.h_py);
}

void DiffRenderer::init(int canvas_w, int canvas_h, int max_shapes) {
    canvas_w_ = canvas_w;
    canvas_h_ = canvas_h;
    max_shapes_ = max_shapes;

    auto& s = scratch_;
    s.total_pixels = canvas_w * canvas_h;

    // Smart VRAM-aware tile allocation
    size_t free_bytes, total_bytes;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
        size_t scratch_budget = static_cast<size_t>(static_cast<double>(free_bytes) * 0.60);
        long long needed_for_full = static_cast<long long>(max_shapes) * s.total_pixels * 8LL;
        if (needed_for_full <= static_cast<long long>(scratch_budget)) {
            s.tile_pixels = s.total_pixels;
            full_scratch_ = true;
            VIN_INFO("Full-size scratch: %lld MB (VRAM free: %zu MB, budget: %zu MB)",
                     needed_for_full / (1024 * 1024), free_bytes / (1024 * 1024),
                     scratch_budget / (1024 * 1024));
        } else {
            s.tile_pixels = static_cast<int>(scratch_budget / (max_shapes * 8LL));
            s.tile_pixels = std::max(s.tile_pixels, 256);
            s.tile_pixels = std::min(s.tile_pixels, s.total_pixels);
            VIN_INFO("Tiled scratch: %d pixels/tile (VRAM free: %zu MB, budget: %zu MB)",
                     s.tile_pixels, free_bytes / (1024 * 1024), scratch_budget / (1024 * 1024));
        }
    } else {
        // Fallback: conservative 4GB limit
        long long max_scratch = 4LL * 1024 * 1024 * 1024;
        s.tile_pixels = static_cast<int>(max_scratch / (max_shapes * 8LL));
        s.tile_pixels = std::max(s.tile_pixels, 256);
        s.tile_pixels = std::min(s.tile_pixels, s.total_pixels);
        VIN_INFO("Fallback tile: %d pixels/tile (cudaMemGetInfo failed)", s.tile_pixels);
    }

    // Create CUDA stream
    CUDA_CHECK(cudaStreamCreate(&s.stream));

    // Allocate pixel coordinates (constant)
    CUDA_CHECK(cudaMalloc(&s.d_px, s.total_pixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_py, s.total_pixels * sizeof(float)));

    // Initialize pixel coordinates on host and upload
    std::vector<float> h_px(s.total_pixels), h_py(s.total_pixels);
    for (int y = 0; y < canvas_h; ++y) {
        for (int x = 0; x < canvas_w; ++x) {
            int idx = y * canvas_w + x;
            h_px[idx] = static_cast<float>(x);
            h_py[idx] = static_cast<float>(y);
        }
    }
    CUDA_CHECK(cudaMemcpy(s.d_px, h_px.data(), s.total_pixels * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.d_py, h_py.data(), s.total_pixels * sizeof(float), cudaMemcpyHostToDevice));

    // Also keep CPU copies for AABB filtering
    h_px_ = std::move(h_px);
    h_py_ = std::move(h_py);

    // Allocate shape parameter buffers
    CUDA_CHECK(cudaMalloc(&s.d_cx, max_shapes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_cy, max_shapes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_rx, max_shapes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_ry, max_shapes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_angle, max_shapes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_colors, max_shapes * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_opacity, max_shapes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_sina, max_shapes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_cosa, max_shapes * sizeof(float)));

    // Background color
    CUDA_CHECK(cudaMalloc(&s.d_bg_color, 3 * sizeof(float)));
    float bg[3] = {0.0f, 0.0f, 0.0f};  // black background
    CUDA_CHECK(cudaMemcpy(s.d_bg_color, bg, 3 * sizeof(float), cudaMemcpyHostToDevice));

    // Output buffer
    CUDA_CHECK(cudaMalloc(&s.d_output, s.total_pixels * 3 * sizeof(float)));

    // Gradient buffers
    CUDA_CHECK(cudaMalloc(&s.d_grad_cx, max_shapes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_grad_cy, max_shapes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_grad_rx, max_shapes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_grad_ry, max_shapes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_grad_angle, max_shapes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_grad_colors, max_shapes * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_grad_opacity, max_shapes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_grad_output, s.total_pixels * 3 * sizeof(float)));

    // Packed gradient buffer for single-download
    CUDA_CHECK(cudaMalloc(&s.d_grads_packed, max_shapes * 9 * sizeof(float)));

    // GPU-resident Adam state
    auto alloc_adam_buf = [&](float** ptr) {
        CUDA_CHECK(cudaMalloc(ptr, max_shapes * sizeof(float)));
        CUDA_CHECK(cudaMemset(*ptr, 0, max_shapes * sizeof(float)));
    };
    alloc_adam_buf(&s.d_m_cx); alloc_adam_buf(&s.d_m_cy);
    alloc_adam_buf(&s.d_m_rx); alloc_adam_buf(&s.d_m_ry);
    alloc_adam_buf(&s.d_m_angle);
    alloc_adam_buf(&s.d_m_cr); alloc_adam_buf(&s.d_m_cg);
    alloc_adam_buf(&s.d_m_cb); alloc_adam_buf(&s.d_m_ca);
    alloc_adam_buf(&s.d_v_cx); alloc_adam_buf(&s.d_v_cy);
    alloc_adam_buf(&s.d_v_rx); alloc_adam_buf(&s.d_v_ry);
    alloc_adam_buf(&s.d_v_angle);
    alloc_adam_buf(&s.d_v_cr); alloc_adam_buf(&s.d_v_cg);
    alloc_adam_buf(&s.d_v_cb); alloc_adam_buf(&s.d_v_ca);

    // STE quantized output buffers
    alloc_adam_buf(&s.d_ste_cx); alloc_adam_buf(&s.d_ste_cy);
    alloc_adam_buf(&s.d_ste_rx); alloc_adam_buf(&s.d_ste_ry);
    alloc_adam_buf(&s.d_ste_angle);
    CUDA_CHECK(cudaMalloc(&s.d_ste_colors, max_shapes * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemset(s.d_ste_colors, 0, max_shapes * 3 * sizeof(float)));
    alloc_adam_buf(&s.d_ste_opacity);

    // Frozen mask
    CUDA_CHECK(cudaMalloc(&s.d_frozen_mask, max_shapes * sizeof(int)));
    CUDA_CHECK(cudaMemset(s.d_frozen_mask, 0, max_shapes * sizeof(int)));

    // Gradient clip norm accumulator
    CUDA_CHECK(cudaMalloc(&s.d_grad_norm_sq, sizeof(float)));

    // Scratch buffers for tile forward
    CUDA_CHECK(cudaMalloc(&s.d_alpha_scratch, (size_t)max_shapes * s.tile_pixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_r_norm_scratch, (size_t)max_shapes * s.tile_pixels * sizeof(float)));

    // Active indices
    CUDA_CHECK(cudaMalloc(&s.d_active_indices, max_shapes * sizeof(int)));

    // Target and mask
    CUDA_CHECK(cudaMalloc(&s.d_target, s.total_pixels * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_mask, s.total_pixels * sizeof(float)));

    // Target alpha, rendered alpha, and alpha gradient
    CUDA_CHECK(cudaMalloc(&s.d_target_alpha, s.total_pixels * sizeof(float)));
    CUDA_CHECK(cudaMemset(s.d_target_alpha, 0, s.total_pixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_rendered_alpha, s.total_pixels * sizeof(float)));
    CUDA_CHECK(cudaMemset(s.d_rendered_alpha, 0, s.total_pixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_grad_alpha, s.total_pixels * sizeof(float)));
    CUDA_CHECK(cudaMemset(s.d_grad_alpha, 0, s.total_pixels * sizeof(float)));

    // Per-shape visibility (for relocation scan)
    CUDA_CHECK(cudaMalloc(&s.d_visibility, max_shapes * sizeof(float)));

    // Canvas uint8 buffer for async visualization
    CUDA_CHECK(cudaMalloc(&s.d_canvas_u8, s.total_pixels * 3 * sizeof(uint8_t)));

    // MSE sum accumulator (pre-allocated)
    CUDA_CHECK(cudaMalloc(&s.d_mse_sum, sizeof(float)));

    // Alpha LUT texture (from shared LUT, 4096 entries with hardware interpolation)
    {
        const auto& shared_lut = alpha_228_lut();

        cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
        CUDA_CHECK(cudaMallocArray(&s.alpha_array, &desc, ALPHA_LUT_SIZE, 1));
        CUDA_CHECK(cudaMemcpyToArray(s.alpha_array, 0, 0,
                    shared_lut.data(), ALPHA_LUT_SIZE * sizeof(float),
                    cudaMemcpyHostToDevice));

        cudaResourceDesc resDesc = {};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = s.alpha_array;

        cudaTextureDesc texDesc = {};
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 1;

        CUDA_CHECK(cudaCreateTextureObject(&s.alpha_tex, &resDesc, &texDesc, nullptr));
    }

    // Pinned host memory for AABB filtering
    CUDA_CHECK(cudaMallocHost(&s.h_cx, max_shapes * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&s.h_cy, max_shapes * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&s.h_rx, max_shapes * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&s.h_ry, max_shapes * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&s.h_px, s.tile_pixels * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&s.h_py, s.tile_pixels * sizeof(float)));

    // CPU-side vectors for AABB filtering
    h_cx_.resize(max_shapes);
    h_cy_.resize(max_shapes);
    h_rx_.resize(max_shapes);
    h_ry_.resize(max_shapes);

    // Pre-allocate upload buffers
    h_angle_buf_.resize(max_shapes);
    h_colors_buf_.resize(max_shapes * 3);
    h_opacity_buf_.resize(max_shapes);

    // Precompute tile AABBs (pixel coordinates are constant)
    s.num_tiles = (s.total_pixels + s.tile_pixels - 1) / s.tile_pixels;
    s.tile_tx1.resize(s.num_tiles);
    s.tile_tx2.resize(s.num_tiles);
    s.tile_ty1.resize(s.num_tiles);
    s.tile_ty2.resize(s.num_tiles);
    for (int t = 0; t < s.num_tiles; t++) {
        int p_start = t * s.tile_pixels;
        int p_count = std::min(s.tile_pixels, s.total_pixels - p_start);
        float tx1 = 1e30f, tx2 = -1e30f, ty1 = 1e30f, ty2 = -1e30f;
        for (int k = 0; k < p_count; k++) {
            float x = h_px_[p_start + k], y = h_py_[p_start + k];
            if (x < tx1) tx1 = x; if (x > tx2) tx2 = x;
            if (y < ty1) ty1 = y; if (y > ty2) ty2 = y;
        }
        s.tile_tx1[t] = tx1;
        s.tile_tx2[t] = tx2;
        s.tile_ty1[t] = ty1;
        s.tile_ty2[t] = ty2;
    }

    // GPU buffers for per-tile active indices
    CUDA_CHECK(cudaMalloc(&s.d_active_indices_per_tile, (size_t)s.num_tiles * max_shapes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&s.d_active_count_per_tile, s.num_tiles * sizeof(int)));

    initialized_ = true;
    VIN_INFO("DiffRenderer initialized: %dx%d, max_shapes=%d, tile_pixels=%d",
             canvas_w, canvas_h, max_shapes, s.tile_pixels);
}

void DiffRenderer::set_target(const float* h_target, const float* h_mask, const float* h_target_alpha) {
    auto& s = scratch_;
    CUDA_CHECK(cudaMemcpy(s.d_target, h_target, s.total_pixels * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.d_mask, h_mask, s.total_pixels * sizeof(float), cudaMemcpyHostToDevice));
    if (h_target_alpha) {
        CUDA_CHECK(cudaMemcpy(s.d_target_alpha, h_target_alpha, s.total_pixels * sizeof(float), cudaMemcpyHostToDevice));
    } else {
        // Default: all opaque (target_alpha = 255)
        std::vector<float> ones(s.total_pixels, 255.0f);
        CUDA_CHECK(cudaMemcpy(s.d_target_alpha, ones.data(), s.total_pixels * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void DiffRenderer::set_bg_color(float r, float g, float b) {
    auto& s = scratch_;
    float bg[3] = {r, g, b};
    if (s.d_bg_color) {
        CUDA_CHECK(cudaMemcpy(s.d_bg_color, bg, 3 * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void DiffRenderer::compute_visibility(int N) {
    auto& s = scratch_;
    if (!initialized_ || N == 0) return;

    CUDA_CHECK(cudaMemsetAsync(s.d_visibility, 0, N * sizeof(float), s.stream));

    cuda_render_tile_visibility(
        s.d_px, s.d_py, s.d_ste_cx, s.d_ste_cy, s.d_ste_rx, s.d_ste_ry, s.d_ste_angle,
        s.d_ste_colors, s.d_ste_opacity, s.d_bg_color,
        s.d_sina, s.d_cosa,
        s.d_alpha_scratch, s.d_r_norm_scratch,
        s.d_visibility,
        s.h_cx, s.h_cy, s.h_rx, s.h_ry,
        h_px_.data(), h_py_.data(),
        s.d_active_indices, N, s.total_pixels, s.tile_pixels,
        s.stream,
        s.alpha_tex,
        s.tile_tx1.data(), s.tile_tx2.data(),
        s.tile_ty1.data(), s.tile_ty2.data(),
        s.num_tiles);
}

void DiffRenderer::download_visibility(float* h_visibility, int N) {
    auto& s = scratch_;
    if (!initialized_ || !s.d_visibility) return;
    CUDA_CHECK(cudaMemcpyAsync(h_visibility, s.d_visibility,
                N * sizeof(float), cudaMemcpyDeviceToHost, s.stream));
    CUDA_CHECK(cudaStreamSynchronize(s.stream));
}

void DiffRenderer::render_forward(const ShapeParams* shapes, int N,
                                   float* h_output) {
    if (!initialized_ || N == 0) return;

    auto& s = scratch_;
    upload_params_gpu(shapes, N);

    // Compute sin/cos on GPU
    cuda_compute_sin_cos(s.d_angle, s.d_sina, s.d_cosa, N, s.stream);

    // Launch forward kernel using STE buffers
    cuda_render_forward(
        s.d_px, s.d_py, s.d_ste_cx, s.d_ste_cy, s.d_ste_rx, s.d_ste_ry, s.d_ste_angle,
        s.d_ste_colors, s.d_ste_opacity, s.d_bg_color, s.d_output,
        s.d_sina, s.d_cosa, N, s.total_pixels, s.stream,
        s.alpha_tex);

    // Download result
    CUDA_CHECK(cudaMemcpyAsync(h_output, s.d_output, s.total_pixels * 3 * sizeof(float),
                               cudaMemcpyDeviceToHost, s.stream));
    CUDA_CHECK(cudaStreamSynchronize(s.stream));
}

void DiffRenderer::download_output(float* h_output) {
    if (!initialized_) return;
    auto& s = scratch_;
    CUDA_CHECK(cudaMemcpyAsync(h_output, s.d_output, s.total_pixels * 3 * sizeof(float),
                               cudaMemcpyDeviceToHost, s.stream));
    CUDA_CHECK(cudaStreamSynchronize(s.stream));
}

float DiffRenderer::compute_mse_and_grad(int N) {
    auto& s = scratch_;

    // Use pre-allocated MSE sum buffer
    CUDA_CHECK(cudaMemsetAsync(s.d_mse_sum, 0, sizeof(float), s.stream));

    int threads = 256;
    int blocks = (s.total_pixels + threads - 1) / threads;
    mse_grad_kernel<<<blocks, threads, 0, s.stream>>>(
        s.d_output, s.d_target, s.d_mask,
        s.d_grad_output, s.d_mse_sum, s.total_pixels,
        s.d_rendered_alpha, s.d_target_alpha, s.d_grad_alpha);
    CUDA_KERNEL_CHECK();

    // Read MSE value
    float mse;
    CUDA_CHECK(cudaMemcpyAsync(&mse, s.d_mse_sum, sizeof(float), cudaMemcpyDeviceToHost, s.stream));
    CUDA_CHECK(cudaStreamSynchronize(s.stream));
    mse /= (s.total_pixels * 4.0f);  // 4 channels (RGBA)

    return mse;
}

float DiffRenderer::render_forward_backward(int N, float* d_grad_history, int write_idx) {
    if (!initialized_ || N == 0) return 0.0f;

    auto& s = scratch_;

    // Compute sin/cos on GPU (from d_angle which was set by fused_adam_update)
    cuda_compute_sin_cos(s.d_angle, s.d_sina, s.d_cosa, N, s.stream);

    // Zero gradient buffers
    CUDA_CHECK(cudaMemsetAsync(s.d_grad_cx, 0, N * sizeof(float), s.stream));
    CUDA_CHECK(cudaMemsetAsync(s.d_grad_cy, 0, N * sizeof(float), s.stream));
    CUDA_CHECK(cudaMemsetAsync(s.d_grad_rx, 0, N * sizeof(float), s.stream));
    CUDA_CHECK(cudaMemsetAsync(s.d_grad_ry, 0, N * sizeof(float), s.stream));
    CUDA_CHECK(cudaMemsetAsync(s.d_grad_angle, 0, N * sizeof(float), s.stream));
    CUDA_CHECK(cudaMemsetAsync(s.d_grad_colors, 0, N * 3 * sizeof(float), s.stream));
    CUDA_CHECK(cudaMemsetAsync(s.d_grad_opacity, 0, N * sizeof(float), s.stream));

    // Step 1: Tile forward all tiles (produces d_output + d_rendered_alpha + scratch)
    cuda_render_tile_forward(
        s.d_px, s.d_py, s.d_ste_cx, s.d_ste_cy, s.d_ste_rx, s.d_ste_ry, s.d_ste_angle,
        s.d_ste_colors, s.d_ste_opacity, s.d_bg_color,
        s.d_sina, s.d_cosa,
        s.d_alpha_scratch, s.d_r_norm_scratch,
        s.d_output,
        s.h_cx, s.h_cy, s.h_rx, s.h_ry,
        h_px_.data(), h_py_.data(),
        s.d_active_indices, N, s.total_pixels, s.tile_pixels,
        s.stream,
        s.alpha_tex,
        s.tile_tx1.data(), s.tile_tx2.data(),
        s.tile_ty1.data(), s.tile_ty2.data(),
        s.num_tiles,
        s.d_rendered_alpha);

    // Step 2: Compute MSE and grad_output from d_output
    float mse = compute_mse_and_grad(N);

    // Step 3: Backward pass
    if (full_scratch_) {
        // Full-size scratch: reuse data from step 1, no re-forward needed
        cuda_render_tile_backward(
            s.d_px, s.d_py, s.d_ste_cx, s.d_ste_cy, s.d_ste_rx, s.d_ste_ry, s.d_ste_angle,
            s.d_ste_colors, s.d_ste_opacity, s.d_bg_color, s.d_grad_output,
            s.d_sina, s.d_cosa,
            s.d_alpha_scratch, s.d_r_norm_scratch,
            s.d_grad_cx, s.d_grad_cy, s.d_grad_rx, s.d_grad_ry,
            s.d_grad_angle, s.d_grad_colors, s.d_grad_opacity,
            s.h_cx, s.h_cy, s.h_rx, s.h_ry,
            h_px_.data(), h_py_.data(),
            s.d_active_indices, N, s.total_pixels, s.tile_pixels,
            s.stream,
            s.tile_tx1.data(), s.tile_tx2.data(),
            s.tile_ty1.data(), s.tile_ty2.data(),
            s.num_tiles,
            s.d_grad_alpha);
    } else {
        // Tiled: re-forward per-tile for scratch, then backward
        cuda_render_tile_forward_backward_per_tile(
            s.d_px, s.d_py, s.d_ste_cx, s.d_ste_cy, s.d_ste_rx, s.d_ste_ry, s.d_ste_angle,
            s.d_ste_colors, s.d_ste_opacity, s.d_bg_color, s.d_grad_output,
            s.d_sina, s.d_cosa,
            s.d_alpha_scratch, s.d_r_norm_scratch,
            s.d_grad_cx, s.d_grad_cy, s.d_grad_rx, s.d_grad_ry,
            s.d_grad_angle, s.d_grad_colors, s.d_grad_opacity,
            s.h_cx, s.h_cy, s.h_rx, s.h_ry,
            h_px_.data(), h_py_.data(),
            s.d_active_indices, N, s.total_pixels, s.tile_pixels,
            s.stream,
            s.alpha_tex,
            s.tile_tx1.data(), s.tile_tx2.data(),
            s.tile_ty1.data(), s.tile_ty2.data(),
            s.num_tiles,
            s.d_grad_alpha);
    }

    // Step 4: Pack gradients (stays in d_grads_packed for fused_adam_update)
    {
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        pack_and_norm_grads_kernel<<<blocks, threads, 0, s.stream>>>(
            s.d_grad_cx, s.d_grad_cy, s.d_grad_rx, s.d_grad_ry,
            s.d_grad_angle, s.d_grad_colors, s.d_grad_opacity,
            s.d_grads_packed, N, d_grad_history, write_idx);
        CUDA_KERNEL_CHECK();
    }

    return mse;
}

void DiffRenderer::async_copy_canvas(uint8_t* h_pinned_canvas, cudaEvent_t& event) {
    if (!initialized_) return;
    auto& s = scratch_;
    int total_elements = s.total_pixels * 3;

    // Convert float d_output to uint8 on GPU
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    float_to_uint8_kernel<<<blocks, threads, 0, s.stream>>>(
        s.d_output, s.d_canvas_u8, total_elements);

    // Async copy to pinned host memory
    CUDA_CHECK(cudaMemcpyAsync(h_pinned_canvas, s.d_canvas_u8,
                               total_elements * sizeof(uint8_t),
                               cudaMemcpyDeviceToHost, s.stream));

    // Record event so we can synchronize later
    CUDA_CHECK(cudaEventRecord(event, s.stream));
}

void DiffRenderer::async_snapshot_params(float* h_pinned, cudaEvent_t event, int N) {
    auto& s = scratch_;
    CUDA_CHECK(cudaMemcpyAsync(h_pinned + 0 * N, s.d_cx, N * sizeof(float), cudaMemcpyDeviceToHost, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(h_pinned + 1 * N, s.d_cy, N * sizeof(float), cudaMemcpyDeviceToHost, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(h_pinned + 2 * N, s.d_rx, N * sizeof(float), cudaMemcpyDeviceToHost, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(h_pinned + 3 * N, s.d_ry, N * sizeof(float), cudaMemcpyDeviceToHost, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(h_pinned + 4 * N, s.d_angle, N * sizeof(float), cudaMemcpyDeviceToHost, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(h_pinned + 5 * N, s.d_colors, N * 3 * sizeof(float), cudaMemcpyDeviceToHost, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(h_pinned + 8 * N, s.d_opacity, N * sizeof(float), cudaMemcpyDeviceToHost, s.stream));
    CUDA_CHECK(cudaEventRecord(event, s.stream));
}

void DiffRenderer::upload_params_gpu(const ShapeParams* shapes, int N) {
    auto& s = scratch_;
    // Unpack and upload to GPU-resident param buffers
    for (int i = 0; i < N; ++i) {
        h_cx_[i] = shapes[i].cx;
        h_cy_[i] = shapes[i].cy;
        h_rx_[i] = shapes[i].rx;
        h_ry_[i] = shapes[i].ry;
        h_angle_buf_[i] = shapes[i].angle;
        h_colors_buf_[i * 3 + 0] = shapes[i].color_r;
        h_colors_buf_[i * 3 + 1] = shapes[i].color_g;
        h_colors_buf_[i * 3 + 2] = shapes[i].color_b;
        h_opacity_buf_[i] = shapes[i].opacity;
    }
    CUDA_CHECK(cudaMemcpyAsync(s.d_cx, h_cx_.data(), N * sizeof(float), cudaMemcpyHostToDevice, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(s.d_cy, h_cy_.data(), N * sizeof(float), cudaMemcpyHostToDevice, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(s.d_rx, h_rx_.data(), N * sizeof(float), cudaMemcpyHostToDevice, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(s.d_ry, h_ry_.data(), N * sizeof(float), cudaMemcpyHostToDevice, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(s.d_angle, h_angle_buf_.data(), N * sizeof(float), cudaMemcpyHostToDevice, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(s.d_colors, h_colors_buf_.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(s.d_opacity, h_opacity_buf_.data(), N * sizeof(float), cudaMemcpyHostToDevice, s.stream));

    // Also copy to pinned memory for AABB filtering
    memcpy(s.h_cx, h_cx_.data(), N * sizeof(float));
    memcpy(s.h_cy, h_cy_.data(), N * sizeof(float));
    memcpy(s.h_rx, h_rx_.data(), N * sizeof(float));
    memcpy(s.h_ry, h_ry_.data(), N * sizeof(float));

    // Initial STE: copy params to STE buffers
    CUDA_CHECK(cudaMemcpyAsync(s.d_ste_cx, s.d_cx, N * sizeof(float), cudaMemcpyDeviceToDevice, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(s.d_ste_cy, s.d_cy, N * sizeof(float), cudaMemcpyDeviceToDevice, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(s.d_ste_rx, s.d_rx, N * sizeof(float), cudaMemcpyDeviceToDevice, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(s.d_ste_ry, s.d_ry, N * sizeof(float), cudaMemcpyDeviceToDevice, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(s.d_ste_angle, s.d_angle, N * sizeof(float), cudaMemcpyDeviceToDevice, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(s.d_ste_colors, s.d_colors, N * 3 * sizeof(float), cudaMemcpyDeviceToDevice, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(s.d_ste_opacity, s.d_opacity, N * sizeof(float), cudaMemcpyDeviceToDevice, s.stream));
}

void DiffRenderer::fused_adam_update(
    int N, int adam_step,
    float lr, float beta1, float beta2, float eps,
    float max_grad_norm,
    int W, int H)
{
    auto& s = scratch_;
    int total_grad_elements = N * 9;

    // Step 1: Global gradient norm clipping
    if (max_grad_norm > 0) {
        CUDA_CHECK(cudaMemsetAsync(s.d_grad_norm_sq, 0, sizeof(float), s.stream));
        int threads = 256;
        int blocks = (total_grad_elements + threads - 1) / threads;
        clip_grad_norm_kernel_v2<<<blocks, threads, 0, s.stream>>>(
            s.d_grads_packed, total_grad_elements, max_grad_norm, s.d_grad_norm_sq);
        CUDA_KERNEL_CHECK();

        // Read norm and compute scale on CPU (small data, negligible cost)
        float norm_sq;
        CUDA_CHECK(cudaMemcpyAsync(&norm_sq, s.d_grad_norm_sq, sizeof(float),
                                   cudaMemcpyDeviceToHost, s.stream));
        CUDA_CHECK(cudaStreamSynchronize(s.stream));

        float global_norm = sqrtf(norm_sq);
        if (global_norm > max_grad_norm) {
            float scale = max_grad_norm / global_norm;
            scale_grads_kernel_v2<<<blocks, threads, 0, s.stream>>>(
                s.d_grads_packed, total_grad_elements, scale);
            CUDA_KERNEL_CHECK();
        }
    }

    // Step 2: Fused Adam + clamp + STE
    {
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        fused_adam_kernel<<<blocks, threads, 0, s.stream>>>(
            s.d_grads_packed,
            s.d_cx, s.d_cy, s.d_rx, s.d_ry, s.d_angle, s.d_colors, s.d_opacity,
            s.d_m_cx, s.d_m_cy, s.d_m_rx, s.d_m_ry, s.d_m_angle,
            s.d_m_cr, s.d_m_cg, s.d_m_cb, s.d_m_ca,
            s.d_v_cx, s.d_v_cy, s.d_v_rx, s.d_v_ry, s.d_v_angle,
            s.d_v_cr, s.d_v_cg, s.d_v_cb, s.d_v_ca,
            s.d_ste_cx, s.d_ste_cy, s.d_ste_rx, s.d_ste_ry,
            s.d_ste_angle, s.d_ste_colors, s.d_ste_opacity,
            N, adam_step, lr, beta1, beta2, eps, max_grad_norm,
            W, H, s.d_frozen_mask);
        CUDA_KERNEL_CHECK();
    }

    // Step 3: Update pinned AABB data from GPU params
    CUDA_CHECK(cudaMemcpyAsync(s.h_cx, s.d_cx, N * sizeof(float), cudaMemcpyDeviceToHost, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(s.h_cy, s.d_cy, N * sizeof(float), cudaMemcpyDeviceToHost, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(s.h_rx, s.d_rx, N * sizeof(float), cudaMemcpyDeviceToHost, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(s.h_ry, s.d_ry, N * sizeof(float), cudaMemcpyDeviceToHost, s.stream));
    CUDA_CHECK(cudaStreamSynchronize(s.stream));
}

void DiffRenderer::download_params_gpu(ShapeParams* shapes, int N) {
    auto& s = scratch_;
    // Download from GPU-resident param buffers
    CUDA_CHECK(cudaMemcpyAsync(h_cx_.data(), s.d_cx, N * sizeof(float), cudaMemcpyDeviceToHost, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(h_cy_.data(), s.d_cy, N * sizeof(float), cudaMemcpyDeviceToHost, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(h_rx_.data(), s.d_rx, N * sizeof(float), cudaMemcpyDeviceToHost, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(h_ry_.data(), s.d_ry, N * sizeof(float), cudaMemcpyDeviceToHost, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(h_angle_buf_.data(), s.d_angle, N * sizeof(float), cudaMemcpyDeviceToHost, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(h_colors_buf_.data(), s.d_colors, N * 3 * sizeof(float), cudaMemcpyDeviceToHost, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(h_opacity_buf_.data(), s.d_opacity, N * sizeof(float), cudaMemcpyDeviceToHost, s.stream));
    CUDA_CHECK(cudaStreamSynchronize(s.stream));

    for (int i = 0; i < N; ++i) {
        shapes[i].cx = h_cx_[i];
        shapes[i].cy = h_cy_[i];
        shapes[i].rx = h_rx_[i];
        shapes[i].ry = h_ry_[i];
        shapes[i].angle = h_angle_buf_[i];
        shapes[i].color_r = h_colors_buf_[i * 3 + 0];
        shapes[i].color_g = h_colors_buf_[i * 3 + 1];
        shapes[i].color_b = h_colors_buf_[i * 3 + 2];
        shapes[i].opacity = h_opacity_buf_[i];
    }
}

void DiffRenderer::set_frozen_mask(const int* frozen, int N) {
    auto& s = scratch_;
    CUDA_CHECK(cudaMemcpyAsync(s.d_frozen_mask, frozen, N * sizeof(int),
                               cudaMemcpyHostToDevice, s.stream));
}

void DiffRenderer::download_grads_packed(float* h_grads, int N) {
    auto& s = scratch_;
    CUDA_CHECK(cudaMemcpyAsync(h_grads, s.d_grads_packed, N * 9 * sizeof(float),
                               cudaMemcpyDeviceToHost, s.stream));
    CUDA_CHECK(cudaStreamSynchronize(s.stream));
}

void DiffRenderer::reset_adam_state_gpu(const std::vector<int>& indices) {
    auto& s = scratch_;
    for (int i : indices) {
        float zero = 0.0f;
        CUDA_CHECK(cudaMemcpyAsync(s.d_m_cx + i, &zero, sizeof(float), cudaMemcpyHostToDevice, s.stream));
        CUDA_CHECK(cudaMemcpyAsync(s.d_m_cy + i, &zero, sizeof(float), cudaMemcpyHostToDevice, s.stream));
        CUDA_CHECK(cudaMemcpyAsync(s.d_m_rx + i, &zero, sizeof(float), cudaMemcpyHostToDevice, s.stream));
        CUDA_CHECK(cudaMemcpyAsync(s.d_m_ry + i, &zero, sizeof(float), cudaMemcpyHostToDevice, s.stream));
        CUDA_CHECK(cudaMemcpyAsync(s.d_m_angle + i, &zero, sizeof(float), cudaMemcpyHostToDevice, s.stream));
        CUDA_CHECK(cudaMemcpyAsync(s.d_m_cr + i, &zero, sizeof(float), cudaMemcpyHostToDevice, s.stream));
        CUDA_CHECK(cudaMemcpyAsync(s.d_m_cg + i, &zero, sizeof(float), cudaMemcpyHostToDevice, s.stream));
        CUDA_CHECK(cudaMemcpyAsync(s.d_m_cb + i, &zero, sizeof(float), cudaMemcpyHostToDevice, s.stream));
        CUDA_CHECK(cudaMemcpyAsync(s.d_m_ca + i, &zero, sizeof(float), cudaMemcpyHostToDevice, s.stream));
        CUDA_CHECK(cudaMemcpyAsync(s.d_v_cx + i, &zero, sizeof(float), cudaMemcpyHostToDevice, s.stream));
        CUDA_CHECK(cudaMemcpyAsync(s.d_v_cy + i, &zero, sizeof(float), cudaMemcpyHostToDevice, s.stream));
        CUDA_CHECK(cudaMemcpyAsync(s.d_v_rx + i, &zero, sizeof(float), cudaMemcpyHostToDevice, s.stream));
        CUDA_CHECK(cudaMemcpyAsync(s.d_v_ry + i, &zero, sizeof(float), cudaMemcpyHostToDevice, s.stream));
        CUDA_CHECK(cudaMemcpyAsync(s.d_v_angle + i, &zero, sizeof(float), cudaMemcpyHostToDevice, s.stream));
        CUDA_CHECK(cudaMemcpyAsync(s.d_v_cr + i, &zero, sizeof(float), cudaMemcpyHostToDevice, s.stream));
        CUDA_CHECK(cudaMemcpyAsync(s.d_v_cg + i, &zero, sizeof(float), cudaMemcpyHostToDevice, s.stream));
        CUDA_CHECK(cudaMemcpyAsync(s.d_v_cb + i, &zero, sizeof(float), cudaMemcpyHostToDevice, s.stream));
        CUDA_CHECK(cudaMemcpyAsync(s.d_v_ca + i, &zero, sizeof(float), cudaMemcpyHostToDevice, s.stream));
    }
}

void DiffRenderer::reset_adam_state_all_gpu(int N) {
    auto& s = scratch_;
    CUDA_CHECK(cudaMemsetAsync(s.d_m_cx, 0, N * sizeof(float), s.stream));
    CUDA_CHECK(cudaMemsetAsync(s.d_m_cy, 0, N * sizeof(float), s.stream));
    CUDA_CHECK(cudaMemsetAsync(s.d_m_rx, 0, N * sizeof(float), s.stream));
    CUDA_CHECK(cudaMemsetAsync(s.d_m_ry, 0, N * sizeof(float), s.stream));
    CUDA_CHECK(cudaMemsetAsync(s.d_m_angle, 0, N * sizeof(float), s.stream));
    CUDA_CHECK(cudaMemsetAsync(s.d_m_cr, 0, N * sizeof(float), s.stream));
    CUDA_CHECK(cudaMemsetAsync(s.d_m_cg, 0, N * sizeof(float), s.stream));
    CUDA_CHECK(cudaMemsetAsync(s.d_m_cb, 0, N * sizeof(float), s.stream));
    CUDA_CHECK(cudaMemsetAsync(s.d_m_ca, 0, N * sizeof(float), s.stream));
    CUDA_CHECK(cudaMemsetAsync(s.d_v_cx, 0, N * sizeof(float), s.stream));
    CUDA_CHECK(cudaMemsetAsync(s.d_v_cy, 0, N * sizeof(float), s.stream));
    CUDA_CHECK(cudaMemsetAsync(s.d_v_rx, 0, N * sizeof(float), s.stream));
    CUDA_CHECK(cudaMemsetAsync(s.d_v_ry, 0, N * sizeof(float), s.stream));
    CUDA_CHECK(cudaMemsetAsync(s.d_v_angle, 0, N * sizeof(float), s.stream));
    CUDA_CHECK(cudaMemsetAsync(s.d_v_cr, 0, N * sizeof(float), s.stream));
    CUDA_CHECK(cudaMemsetAsync(s.d_v_cg, 0, N * sizeof(float), s.stream));
    CUDA_CHECK(cudaMemsetAsync(s.d_v_cb, 0, N * sizeof(float), s.stream));
    CUDA_CHECK(cudaMemsetAsync(s.d_v_ca, 0, N * sizeof(float), s.stream));
}

void DiffRenderer::sync_stream() {
    if (!initialized_) return;
    CUDA_CHECK(cudaStreamSynchronize(scratch_.stream));
}

} // namespace vinylizer
