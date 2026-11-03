#include "core/diff_renderer.h"
#include "common/logger.h"
#include "common/alpha_228.h"
#include "cuda/template_sampling.cuh"

#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "common/stamp_loader.h"
#include "common/path_utils.h"

// CUDA-OpenGL interop 函数声明（避免引入 cuda_gl_interop.h 依赖 GL/gl.h）
// 这些函数实际上在 cuda_runtime.h 中也可用，只需要前向声明
struct cudaGraphicsResource;
extern "C" cudaError_t cudaGraphicsMapResources(int count, cudaGraphicsResource** resources, cudaStream_t stream);
extern "C" cudaError_t cudaGraphicsUnmapResources(int count, cudaGraphicsResource** resources, cudaStream_t stream);
extern "C" cudaError_t cudaGraphicsSubResourceGetMappedArray(cudaArray** array, cudaGraphicsResource* resource, unsigned int arrayIndex, unsigned int mipLevel);

// Forward declarations for CUDA kernel entry points (in render_kernel.cu)
extern "C" {

void cuda_compute_sin_cos(
    const float* d_angle,
    float* d_sina, float* d_cosa,
    int N, cudaStream_t stream);

void cuda_compute_softmax(
    const float* d_logits,
    float* d_softmax_w,
    int N, float temperature, cudaStream_t stream,
    int num_types);

void cuda_update_locked_types(
    const float* d_softmax_w,
    int* d_shape_types,
    int N, cudaStream_t stream,
    int num_types,
    bool force_lock);

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
}

// Simple CUDA kernel for MSE gradient computation
// MSE gradient kernel with dynamic channel count
// channels=3: RGB only (alpha-loss off, gradient scale 2/(C*3))
// channels=4: RGBA (alpha-loss on, gradient scale 2/(C*4))
__global__ void mse_grad_kernel(
    const float* __restrict__ output,     // [C, 3]
    const float* __restrict__ target,     // [C, 3]
    const float* __restrict__ mask,       // [C]
    float* __restrict__ grad_output,      // [C, 3]
    float* __restrict__ mse_sum,          // [1] atomic result
    int C,
    int channels,                              // 3 or 4
    const float* __restrict__ rendered_alpha,  // [C] rendered alpha (0-255)
    const float* __restrict__ target_alpha,    // [C] target alpha (0 or 255)
    float* __restrict__ grad_alpha)            // [C] alpha gradient output
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C) return;

    float m = mask[idx];
    float scale = 2.0f / (C * (float)channels);

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

        grad_output[idx * 3 + 0] = dr * scale;
        grad_output[idx * 3 + 1] = dg * scale;
        grad_output[idx * 3 + 2] = db * scale;
    }

    // Alpha gradient: only computed when alpha-loss is on (channels==4)
    if (channels == 4 && rendered_alpha && target_alpha && grad_alpha) {
        float da = rendered_alpha[idx] - target_alpha[idx];
        atomicAdd(mse_sum, da * da);
        grad_alpha[idx] = da * scale;
    } else if (grad_alpha) {
        grad_alpha[idx] = 0.0f;
    }
}

// === Gradient Loss（残差拉普拉斯）===
// g'(p) = g(p) - λ·Δg(p)，Δ 为 3×3 拉普拉斯 [0,1,0;1,-4,1;0,1,0]，RGB 三通道独立。
// 数学依据：DᵀD=-Δ ⇒ ∂L_grad/∂R = -2λ·Δ(R-T)，故 ∂L_total/∂R ∝ r - λ·Δr；
// g(p) 已是归一化的 2r/(C·3)，Δ 为线性算子，直接作用于 g 再减 λ·Δg 即得正确梯度。
// 边界处理：图像边界与 mask 边界均 clamp——越界或掩码外的邻居用中心值替代
//（等价于残差在边界处复制外延，避免 mask 边沿产生虚假二阶差分）。
// grad_in 与 grad_out 必须指向不同 buffer。
__global__ void grad_loss_laplacian_kernel(
    const float* __restrict__ grad_in,   // [H*W*3] 原始像素梯度 g
    const float* __restrict__ mask,      // [H*W]
    float* __restrict__ grad_out,        // [H*W*3] 输出 g - λ·Δg
    int H, int W, float lambda)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int idx = y * W + x;

    if (mask[idx] < 0.5f) {
        grad_out[idx * 3 + 0] = 0.0f;
        grad_out[idx * 3 + 1] = 0.0f;
        grad_out[idx * 3 + 2] = 0.0f;
        return;
    }

    // 4 邻居索引（越界或掩码外 -> -1，取值时 clamp 为中心值）
    int n_l = (x > 0     && mask[idx - 1] >= 0.5f) ? (idx - 1) : -1;
    int n_r = (x < W - 1 && mask[idx + 1] >= 0.5f) ? (idx + 1) : -1;
    int n_u = (y > 0     && mask[idx - W] >= 0.5f) ? (idx - W) : -1;
    int n_d = (y < H - 1 && mask[idx + W] >= 0.5f) ? (idx + W) : -1;

    for (int c = 0; c < 3; ++c) {
        float gc = grad_in[idx * 3 + c];
        float gl = (n_l >= 0) ? grad_in[n_l * 3 + c] : gc;
        float gr = (n_r >= 0) ? grad_in[n_r * 3 + c] : gc;
        float gu = (n_u >= 0) ? grad_in[n_u * 3 + c] : gc;
        float gd = (n_d >= 0) ? grad_in[n_d * 3 + c] : gc;
        float lap = gl + gr + gu + gd - 4.0f * gc;
        grad_out[idx * 3 + c] = gc - lambda * lap;
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

// ──────────────────────────────────────────────────────────────
// Multi-scale pyramid loss kernels (方案C: pure CUDA pyramid)
// ──────────────────────────────────────────────────────────────

// 2x downsample via 2x2 average pooling
// input: [H, W, 3], output: [H/2, W/2, 3]
__global__ void downsample2x_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int H, int W)
{
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    int oh = H / 2, ow = W / 2;
    if (ox >= ow || oy >= oh) return;

    for (int c = 0; c < 3; c++) {
        float sum = input[((2*oy+0)*W + (2*ox+0))*3 + c]
                  + input[((2*oy+0)*W + (2*ox+1))*3 + c]
                  + input[((2*oy+1)*W + (2*ox+0))*3 + c]
                  + input[((2*oy+1)*W + (2*ox+1))*3 + c];
        output[(oy*ow + ox)*3 + c] = sum * 0.25f;
    }
}

// Compute MSE + gradient at a pyramid level and accumulate loss
// grad_level: [H*W*3] gradient at this level (will be upsampled to full res later)
__global__ void pyramid_mse_grad_kernel(
    const float* __restrict__ rendered,
    const float* __restrict__ target,
    float* __restrict__ grad_level,
    float* __restrict__ loss_sum,
    int H, int W, int full_pixels, float level_weight)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= H * W) return;

    float dr = rendered[idx*3+0] - target[idx*3+0];
    float dg = rendered[idx*3+1] - target[idx*3+1];
    float db = rendered[idx*3+2] - target[idx*3+2];
    float sq = dr*dr + dg*dg + db*db;
    float wsq = sq * level_weight;
    if (wsq > 0.0f) atomicAdd(loss_sum, wsq);

    // gradient for this level (will be upsampled to full-res and accumulated)
    float scale = 2.0f * level_weight / (full_pixels * 3.0f);
    grad_level[idx*3+0] = dr * scale;
    grad_level[idx*3+1] = dg * scale;
    grad_level[idx*3+2] = db * scale;
}

// Upsample gradient 2x via nearest neighbor and add to output grad buffer
// grad_in: [H, W, 3], grad_out_full: [full_H, full_W, 3] (write target, additive)
__global__ void upsample_add_grad_kernel(
    const float* __restrict__ grad_in,
    float* __restrict__ grad_out_full,
    int H, int W, int full_W)
{
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    int ow = W * 2, oh = H * 2;
    if (ox >= ow || oy >= oh) return;

    int sx = ox / 2, sy = oy / 2;
    int full_idx = (oy * full_W + ox) * 3;
    int in_idx = (sy * W + sx) * 3;
    grad_out_full[full_idx + 0] += grad_in[in_idx + 0];
    grad_out_full[full_idx + 1] += grad_in[in_idx + 1];
    grad_out_full[full_idx + 2] += grad_in[in_idx + 2];
}

// ──────────────────────────────────────────────────────────────
// Opacity 熵惩罚（二值化）：L_bin = weight · Σ_i opi_i·(1-opi_i)
// opi=0 或 1 时惩罚为 0，opi=0.5 时最大（0.25）。驱动 opacity 趋向双峰。
// 梯度（对 logit）：∂L_bin/∂lo = weight·(1-2·opi)·opi·(1-opi)
//   = weight·(1-2·o)·sigmoid_deriv_from_s(o)   [sum，不除 N]
// 用 atomicAdd 累加到 grad_opacity（与 MSE 梯度叠加，per-shape 仅注入一次）。
// ──────────────────────────────────────────────────────────────
__global__ void alpha_sparsity_kernel(
    const float* __restrict__ opacities,
    float* __restrict__ grad_opacity,
    float* __restrict__ loss_sum,   // 可为 nullptr（仅日志用）
    int N, float weight)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float o = sigmoid_logit(opacities[idx]);
    float penalty = o * (1.0f - o);  // 0 when o=0 or o=1, max 0.25 at o=0.5
    if (loss_sum != nullptr && penalty * weight > 0.0f)
        atomicAdd(loss_sum, penalty * weight);

    // ∂L_bin/∂logit = weight · (1-2o) · o(1-o)，sum（不除 N）
    float g = (1.0f - 2.0f * o) * sigmoid_deriv_from_s(o) * weight;
    atomicAdd(&grad_opacity[idx], g);
}

// Uniform scale constraint kernel: copy rx → ry (rx=ry)
__global__ void uniform_scale_kernel(
    const float* __restrict__ d_rx,
    float* __restrict__ d_ry,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    d_ry[idx] = d_rx[idx];
}

// Clamp opacity logit to [logit_min, +inf). Applied every step after Adam update
// so rendered opacity sigmoid(logit/tau) >= opacity_min. Both d_opacity (param
// buffer) and d_ste_opacity (forward-read STE buffer) are kept in sync.
// logit_min is computed on host with the current tau.
__global__ void clamp_opacity_kernel(
    float* __restrict__ d_opacity,
    float* __restrict__ d_ste_opacity,
    float logit_min,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float v = d_opacity[idx];
    if (v < logit_min) {
        d_opacity[idx] = logit_min;
        d_ste_opacity[idx] = logit_min;
    }
}

// Compute per-pixel error map: (rendered - target)^2 summed over RGB
// Non-mask pixels get 0 error (excluded from relocate top-K sampling)
__global__ void compute_error_map_kernel(
    const float* __restrict__ rendered,  // [total_pixels * 3]
    const float* __restrict__ target,    // [total_pixels * 3]
    const float* __restrict__ mask,      // [total_pixels]
    float* __restrict__ error_map,       // [total_pixels]
    int total_pixels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pixels) return;
    if (mask[idx] < 0.5f) { error_map[idx] = 0.0f; return; }
    float dr = rendered[idx * 3 + 0] - target[idx * 3 + 0];
    float dg = rendered[idx * 3 + 1] - target[idx * 3 + 1];
    float db = rendered[idx * 3 + 2] - target[idx * 3 + 2];
    error_map[idx] = dr * dr + dg * dg + db * db;
}

// EM-YMP: block 归约 sum(error_map) → atomicAdd 到全局标量 d_error_sum。
// 每 block 先做 shared-memory 归约，再由 thread0 做一次 atomicAdd（与 mse_grad_kernel 同模式）。
__global__ void sum_error_map_kernel(
    const float* __restrict__ error_map,   // [total_pixels]
    float* __restrict__ d_error_sum,       // [1]
    int total_pixels)
{
    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = 0.0f;
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float partial = 0.0f;
    for (int p = idx; p < total_pixels; p += stride) {
        partial += error_map[p];
    }
    atomicAdd(&s_sum, partial);
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(d_error_sum, s_sum);
}

// EM-YMP: mean_err = error_sum / max(valid_count, 1)。单线程，1 个标量。
__global__ void compute_mean_err_kernel(
    const float* __restrict__ d_error_sum,  // [1]
    float* __restrict__ d_mean_err,         // [1]
    int valid_count)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float cnt = fmaxf((float)valid_count, 1.0f);
        *d_mean_err = *d_error_sum / cnt;
    }
}

// === Error Potential Field ===
// Bilinear sample a [H*W] float field at continuous (x, y). Edge-clamped.
__device__ __forceinline__ float pf_bilinear(
    const float* __restrict__ field, float x, float y, int W, int H)
{
    x = fminf(fmaxf(x, 0.0f), (float)(W - 1));
    y = fminf(fmaxf(y, 0.0f), (float)(H - 1));
    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);
    int x1 = min(x0 + 1, W - 1);
    int y1 = min(y0 + 1, H - 1);
    float fx = x - (float)x0;
    float fy = y - (float)y0;
    float v00 = field[y0 * W + x0];
    float v01 = field[y0 * W + x1];
    float v10 = field[y1 * W + x0];
    float v11 = field[y1 * W + x1];
    return (v00 * (1.0f - fx) + v01 * fx) * (1.0f - fy)
         + (v10 * (1.0f - fx) + v11 * fx) * fy;
}

// Apply external force from the (normalized) potential gradient field to positions.
// One thread per shape. Displacement = lambda * (1 - gate) * grad_U(center),
// gate = sigmoid((cov - gate_k) / gate_temp), cov = rendered alpha [0,1] at center.
// Force direction: gradient ASCENT of U = blur(E)（往误差强度高的区域走）。
// 软门控按形状中心覆盖率衰减：可见/覆盖好的形状几乎不受力（不打扰正常拟合），
// 被遮挡的死形状受强力（逃出局部最优，获得全局视野）。防扎堆靠门控而非 U 自身。
// Bypasses Adam by design; updates both param and STE (pass-through) buffers.
__global__ void potential_force_kernel(
    float* __restrict__ d_cx, float* __restrict__ d_cy,
    float* __restrict__ d_ste_cx, float* __restrict__ d_ste_cy,
    const float* __restrict__ grad_x, const float* __restrict__ grad_y,
    const float* __restrict__ rendered_alpha,   // [H*W], 0-255
    int N, int W, int H,
    float lambda, float gate_k, float gate_temp)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float cx = d_cx[i];
    float cy = d_cy[i];

    float cov = pf_bilinear(rendered_alpha, cx, cy, W, H) * (1.0f / 255.0f);
    float gate = 1.0f / (1.0f + expf(-(cov - gate_k) / gate_temp));
    float lam = lambda * (1.0f - gate);

    float gx = pf_bilinear(grad_x, cx, cy, W, H);
    float gy = pf_bilinear(grad_y, cx, cy, W, H);

    cx = fminf(fmaxf(cx + lam * gx, 0.0f), (float)(W - 1));
    cy = fminf(fmaxf(cy + lam * gy, 0.0f), (float)(H - 1));

    d_cx[i] = cx;
    d_cy[i] = cy;
    d_ste_cx[i] = cx;
    d_ste_cy[i] = cy;
}

// === Frequency-Hole Potential Field (FH-PF) ===
// vis 门控 kernel: gate_i = 1 - sigmoid((vis_i - thresh)/temp) = 1 / (1 + exp((vis-thresh)/temp))
// vis < thresh → gate≈1（死形状，强受力）；vis > thresh → gate≈0（有用形状，不受扰）。
// 必须在 compute_visibility 之后调用（读取 d_visibility）。
__global__ void fh_vis_gate_kernel(
    const float* __restrict__ vis,   // [N] d_visibility
    float* __restrict__ gate,        // [N] d_fh_vis_gate
    int N, float thresh, float temp)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float v = vis[i];
    // 防溢出：vis 可能很大（全可见大形状），用 clamp
    float x = (v - thresh) / temp;
    if (x > 60.0f) x = 60.0f;
    if (x < -60.0f) x = -60.0f;
    gate[i] = 1.0f / (1.0f + expf(x));   // = 1 - sigmoid(x)
}

// FH-PF 外力 kernel: pos += lambda * gate_i * grad_U(center)
// grad_U 来自高频空洞势场 U=blur(Var·(1−C)) 的归一化梯度（梯度上升→漂向高频空洞）。
// gate_i 来自 d_fh_vis_gate（vis 软门控）。直接改 d_cx/d_cy 及 STE 直通缓冲，绕过 Adam。
__global__ void fh_potential_force_kernel(
    float* __restrict__ d_cx, float* __restrict__ d_cy,
    float* __restrict__ d_ste_cx, float* __restrict__ d_ste_cy,
    const float* __restrict__ grad_x, const float* __restrict__ grad_y,
    const float* __restrict__ vis_gate,   // [N]
    int N, int W, int H, float lambda)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float cx = d_cx[i];
    float cy = d_cy[i];
    float lam = lambda * vis_gate[i];

    float gx = pf_bilinear(grad_x, cx, cy, W, H);
    float gy = pf_bilinear(grad_y, cx, cy, W, H);

    cx = fminf(fmaxf(cx + lam * gx, 0.0f), (float)(W - 1));
    cy = fminf(fmaxf(cy + lam * gy, 0.0f), (float)(H - 1));

    d_cx[i] = cx;
    d_cy[i] = cy;
    d_ste_cx[i] = cx;
    d_ste_cy[i] = cy;
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
    const float* __restrict__ grad_logits,  // [N*num_types]
    float* __restrict__ grads_packed,       // [N*(9+num_types)]
    int N,
    float* grad_history,        // [window * N], nullptr = skip
    int write_idx,
    int num_types)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float gcx = grad_cx[i];
    float gcy = grad_cy[i];
    float grx = grad_rx[i];
    float gry = grad_ry[i];
    int stride = 9 + num_types;
    grads_packed[i * stride + 0] = gcx;
    grads_packed[i * stride + 1] = gcy;
    grads_packed[i * stride + 2] = grx;
    grads_packed[i * stride + 3] = gry;
    grads_packed[i * stride + 4] = grad_angle[i];
    grads_packed[i * stride + 5] = grad_colors[i * 3 + 0];
    grads_packed[i * stride + 6] = grad_colors[i * 3 + 1];
    grads_packed[i * stride + 7] = grad_colors[i * 3 + 2];
    grads_packed[i * stride + 8] = grad_opacity[i];
    for (int j = 0; j < num_types; j++) {
        grads_packed[i * stride + 9 + j] = grad_logits[i * num_types + j];
    }
    if (grad_history) {
        grad_history[write_idx * N + i] = sqrtf(gcx*gcx + gcy*gcy + grx*grx + gry*gry);
    }
}

// Batch reset Adam state for specified indices using GPU kernel
__global__ void reset_adam_state_indices_kernel(
    const int* __restrict__ indices, int num_indices,
    float* __restrict__ m_cx, float* __restrict__ m_cy,
    float* __restrict__ m_rx, float* __restrict__ m_ry,
    float* __restrict__ m_angle,
    float* __restrict__ m_cr, float* __restrict__ m_cg,
    float* __restrict__ m_cb, float* __restrict__ m_ca,
    float* __restrict__ v_cx, float* __restrict__ v_cy,
    float* __restrict__ v_rx, float* __restrict__ v_ry,
    float* __restrict__ v_angle,
    float* __restrict__ v_cr, float* __restrict__ v_cg,
    float* __restrict__ v_cb, float* __restrict__ v_ca,
    float* __restrict__ m_logits, float* __restrict__ v_logits,
    int logits_stride, int num_types)
{
    int ji = blockIdx.x * blockDim.x + threadIdx.x;
    if (ji >= num_indices) return;
    int i = indices[ji];
    m_cx[i] = 0.0f; m_cy[i] = 0.0f;
    m_rx[i] = 0.0f; m_ry[i] = 0.0f;
    m_angle[i] = 0.0f;
    m_cr[i] = 0.0f; m_cg[i] = 0.0f;
    m_cb[i] = 0.0f; m_ca[i] = 0.0f;
    v_cx[i] = 0.0f; v_cy[i] = 0.0f;
    v_rx[i] = 0.0f; v_ry[i] = 0.0f;
    v_angle[i] = 0.0f;
    v_cr[i] = 0.0f; v_cg[i] = 0.0f;
    v_cb[i] = 0.0f; v_ca[i] = 0.0f;
    for (int j = 0; j < num_types; j++) {
        m_logits[i * logits_stride + j] = 0.0f;
        v_logits[i * logits_stride + j] = 0.0f;
    }
}

// Fused Adam update + gradient clip + parameter clamp + STE quantize
// Operates on GPU-resident parameter and Adam state buffers
__global__ void fused_adam_kernel(
    // Gradient (input, may be modified in-place for clipping)
    float* __restrict__ grads_packed,       // [N*(9+num_types)]
    // Parameters (input/output)
    float* __restrict__ d_cx, float* __restrict__ d_cy,
    float* __restrict__ d_rx, float* __restrict__ d_ry,
    float* __restrict__ d_angle,
    float* __restrict__ d_colors,           // [N*3]
    float* __restrict__ d_opacity,
    float* __restrict__ d_type_logits,      // [N*num_types]
    // Adam state (input/output)
    float* __restrict__ m_cx, float* __restrict__ m_cy,
    float* __restrict__ m_rx, float* __restrict__ m_ry,
    float* __restrict__ m_angle,
    float* __restrict__ m_cr, float* __restrict__ m_cg, float* __restrict__ m_cb, float* __restrict__ m_ca,
    float* __restrict__ v_cx, float* __restrict__ v_cy,
    float* __restrict__ v_rx, float* __restrict__ v_ry,
    float* __restrict__ v_angle,
    float* __restrict__ v_cr, float* __restrict__ v_cg, float* __restrict__ v_cb, float* __restrict__ v_ca,
    float* __restrict__ m_logits, float* __restrict__ v_logits,
    int logits_stride,
    // STE output (for rendering)
    float* __restrict__ ste_cx, float* __restrict__ ste_cy,
    float* __restrict__ ste_rx, float* __restrict__ ste_ry,
    float* __restrict__ ste_angle,
    float* __restrict__ ste_colors,         // [N*3]
    float* __restrict__ ste_opacity,
    float* __restrict__ ste_type_logits,    // [N*num_types]
    // Config
    int N, int adam_step,
    float lr, float beta1, float beta2, float eps,
    float max_grad_norm,
    int W, int H,
    // Frozen mask (optional, nullptr if not used)
    const int* __restrict__ frozen_mask,    // [N] 1=frozen, 0=active
    // v8 面积硬上限（解耦实验）：area0 非空且 max_area_ratio>0 时，
    // π·rx·ry > max_area_ratio·area0_i → 等比缩放回边界（投影式 clamp，幂等）。
    // 修复 v5-v7 的结构性失控：颜色相位 cx/cy 冻结时 MSE 只能推 rx/ry 扩张
    // （实测 area×11 vs baseline×2.65 → 互叠 → vis 5% → 死形×3）。
    const float* __restrict__ area0,        // [N] 初始面积（nullptr=关）
    float max_area_ratio,
    float lr_factor_xy, float lr_factor_r, float lr_factor_angle,
    float lr_factor_rgb, float lr_factor_a,
    float logits_lr_factor,
    int num_types,
    int update_mask,              // bit0=位置(cx/cy/angle), bit1=RGBA, bit2=logits, bit3=尺度(rx/ry), bit4=仅opacity(R*模式, bit1关闭时生效)（默认 15=全量）
    bool uniform_scale)           // true=uniform模式：合并rx/ry梯度，使单一r接收完整双向(u+v)梯度
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Check if frozen
    if (frozen_mask && frozen_mask[i]) return;

    int grad_stride = 9 + num_types;
    float* g = &grads_packed[i * grad_stride];

    // Gradient clipping is done globally (separate kernel), skip per-shape here

    // Adam update: compute bias correction once per block in shared memory
    __shared__ float s_bias1, s_bias2;
    if (threadIdx.x == 0) {
        s_bias1 = 1.0f - expf(logf(beta1) * (float)adam_step);
        s_bias2 = 1.0f - expf(logf(beta2) * (float)adam_step);
    }
    __syncthreads();

    auto adam_update = [&](float& param, float grad, float& m, float& v, float lr_factor) {
        m = beta1 * m + (1.0f - beta1) * grad;
        v = beta2 * v + (1.0f - beta2) * grad * grad;
        float m_hat = m / s_bias1;
        float v_hat = v / s_bias2;
        param -= lr * lr_factor * m_hat / (sqrtf(v_hat) + eps);
    };

    // 参数组屏蔽（几何-RGBA 解耦）：被屏蔽组跳过 Adam 更新，m/v 保持不变。
    // v5 拆分：bit0=cx/cy/angle（位置组），bit3=rx/ry（尺度组）。原因（实测证据）：
    // v3/v4 把 rx/ry 划给位置损失（尺度不变归一化 → 面积零梯度方向）+ λ_scale 锚定
    // → 面积冻结在初始值（535→525），覆盖仅 baseline 的 40%（4.2 vs 10.6 层）→
    // PSNR 6.49 dB。而 baseline 证明 MSE 的 rx/ry 梯度是强有效信号（前 50 iter
    // 面积翻倍，最终 ×2.7 铺满画面）。正确分工：cx/cy/angle 由位置损失管（MSE
    // 位移梯度 SNR 低，九/十轮证伪），rx/ry 由 MSE 管（面积信号一致有效）。
    // 默认 update_mask=15 时全部执行，与 baseline 严格等价。
    if (update_mask & 1) {  // 位置组：cx, cy, angle
        adam_update(d_cx[i],    g[0], m_cx[i],    v_cx[i],    lr_factor_xy);
        adam_update(d_cy[i],    g[1], m_cy[i],    v_cy[i],    lr_factor_xy);
        adam_update(d_angle[i], g[4], m_angle[i], v_angle[i], lr_factor_angle);
    }
    if (update_mask & 8) {  // 尺度组：rx, ry（v5 起归颜色相位，由 MSE 驱动）
        if (uniform_scale) {
            // uniform模式：rx只收u方向梯度、ry只收v方向梯度（template_backward 的
            // dadrx=dadu*(-u/r)、dadry=dadv*(-v/r)）。合并后 g_r=grad_rx+grad_ry
            // = -(u·dadu + v·dadv)/r，等价于 diffbmp 标量 r 的完整各向同性梯度。
            float g_r = g[2] + g[3];
            adam_update(d_rx[i], g_r, m_rx[i], v_rx[i], lr_factor_r);
            adam_update(d_ry[i], g_r, m_ry[i], v_ry[i], lr_factor_r);
        } else {
            adam_update(d_rx[i],    g[2], m_rx[i],    v_rx[i],    lr_factor_r);
            adam_update(d_ry[i],    g[3], m_ry[i],    v_ry[i],    lr_factor_r);
        }
    }
    if (update_mask & 2) {  // RGBA 组：colors + opacity
        adam_update(d_colors[i*3+0], g[5], m_cr[i], v_cr[i], lr_factor_rgb);
        adam_update(d_colors[i*3+1], g[6], m_cg[i], v_cg[i], lr_factor_rgb);
        adam_update(d_colors[i*3+2], g[7], m_cb[i], v_cb[i], lr_factor_rgb);
        adam_update(d_opacity[i],     g[8], m_ca[i], v_ca[i], lr_factor_a);
    } else if (update_mask & 16) {  // R* 模式（第十二轮）：仅 opacity（A logit）组。
        // RGB 非优化参数（每轮由 R* 闭式解重解），bit1 关闭时走此分支。
        // baseline 的 update_mask=15 含 bit1 → 永不进入此分支，严格等价不变。
        adam_update(d_opacity[i],     g[8], m_ca[i], v_ca[i], lr_factor_a);
    }

    if (update_mask & 4) {  // type logits 组
        for (int j = 0; j < num_types; j++) {
            adam_update(d_type_logits[i * num_types + j], g[9 + j],
                        m_logits[i * logits_stride + j], v_logits[i * logits_stride + j],
                        logits_lr_factor);
            d_type_logits[i * num_types + j] = fminf(fmaxf(d_type_logits[i * num_types + j], -10.0f), 10.0f);
        }
    }

    // Parameter clamping
    d_cx[i] = fmaxf(0.0f, fminf(d_cx[i], (float)(W - 1)));
    d_cy[i] = fmaxf(0.0f, fminf(d_cy[i], (float)(H - 1)));
    if (uniform_scale) {
        // uniform 模式：半径上限对齐 diffbmp r_max = min(H,W)//4
        float r_max = floorf(fminf((float)W, (float)H) * 0.25f);
        d_rx[i] = fmaxf(1.0f, fminf(d_rx[i], r_max));
        d_ry[i] = fmaxf(1.0f, fminf(d_ry[i], r_max));
    } else {
        // baseline：保留原 clamp [1.0, W/H]，与历史数据 bit-exact 一致
        d_rx[i] = fmaxf(1.0f, fminf(d_rx[i], (float)W));
        d_ry[i] = fmaxf(1.0f, fminf(d_ry[i], (float)H));
    }
    // v8 面积硬上限：投影式等比缩放（在 rx/ry 各自 clamp 之后执行，幂等）
    if (area0 && max_area_ratio > 0.0f) {
        float amax = max_area_ratio * area0[i];
        float a = 3.14159265358979323846f * d_rx[i] * d_ry[i];
        if (a > amax) {
            float s = sqrtf(amax / a);
            d_rx[i] = fmaxf(1.0f, d_rx[i] * s);
            d_ry[i] = fmaxf(1.0f, d_ry[i] * s);
        }
    }
    d_angle[i] = fmodf(d_angle[i], 360.0f);
    if (d_angle[i] < 0) d_angle[i] += 360.0f;
    if (!isfinite(d_angle[i])) d_angle[i] = 0.0f;
    // Logit parameterization: colors/opacity stored in logit space.
    // Clamp to [-10, 10] to avoid sigmoid saturation (sigmoid(±10) ≈ 0/1, gradient ≈ 0).
    d_colors[i*3+0] = fmaxf(-10.0f, fminf(d_colors[i*3+0], 10.0f));
    d_colors[i*3+1] = fmaxf(-10.0f, fminf(d_colors[i*3+1], 10.0f));
    d_colors[i*3+2] = fmaxf(-10.0f, fminf(d_colors[i*3+2], 10.0f));
    d_opacity[i] = fmaxf(-10.0f, fminf(d_opacity[i], 10.0f));

    // Pass-through: use continuous values directly (no STE quantization)
    // Colors and opacity are logits (clamped above), copied as-is to STE buffers
    ste_cx[i] = d_cx[i];
    ste_cy[i] = d_cy[i];
    ste_rx[i] = d_rx[i];
    ste_ry[i] = d_ry[i];
    ste_angle[i] = d_angle[i];
    ste_colors[i*3+0] = d_colors[i*3+0];
    ste_colors[i*3+1] = d_colors[i*3+1];
    ste_colors[i*3+2] = d_colors[i*3+2];
    ste_opacity[i] = d_opacity[i];
    for (int j = 0; j < num_types; j++) {
        ste_type_logits[i * num_types + j] = d_type_logits[i * num_types + j];
    }
}

// === Soft depth z Adam kernel ===
// 独立更新 z_i（与 fused_adam_kernel 分离，避免破坏 grads_packed 布局）。
// z 不参与全局梯度裁剪（其尺度与位置参数不同），但可在此处做简单的 per-param clip。
__global__ void fused_adam_z_kernel(
    float* __restrict__ d_z,            // [N] learnable depth
    const float* __restrict__ d_grad_z, // [N] gradient (already accumulated)
    float* __restrict__ m_z,            // [N] Adam first moment
    float* __restrict__ v_z,            // [N] Adam second moment
    float* __restrict__ ste_z,          // [N] STE copy (z 无量化，直接复制)
    const int* __restrict__ frozen_mask,// [N] 1=frozen (nullptr=无)
    int N, int adam_step,
    float lr, float beta1, float beta2, float eps,
    float lr_factor_z)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (frozen_mask && frozen_mask[i]) {
        ste_z[i] = d_z[i];
        return;
    }
    float g = d_grad_z[i];
    // Per-param clip: 防止 tau_z 很小时 z 梯度爆炸（z 梯度 ∝ 1/tau_z）。
    // z 不参与全局梯度裁剪，但需要自身的 clip 保持数值稳定。
    // 限制梯度幅度为 1.0（z 初始范围 [-1,1]，单步更新不应超过此量级）
    float g_abs = fabsf(g);
    if (g_abs > 1.0f) g = copysignf(1.0f, g);
    m_z[i] = beta1 * m_z[i] + (1.0f - beta1) * g;
    v_z[i] = beta2 * v_z[i] + (1.0f - beta2) * g * g;
    float bias1 = 1.0f - expf(logf(beta1) * (float)adam_step);
    float bias2 = 1.0f - expf(logf(beta2) * (float)adam_step);
    float m_hat = m_z[i] / bias1;
    float v_hat = v_z[i] / bias2;
    d_z[i] -= lr * lr_factor_z * m_hat / (sqrtf(v_hat) + eps);
    // z 不需要 clamp（无界），但限制范围避免数值问题
    d_z[i] = fmaxf(-30.0f, fminf(d_z[i], 30.0f));
    ste_z[i] = d_z[i];
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

// GPU-fused gradient clipping: compute scale from norm_sq on GPU (avoids CPU sync)
__global__ void compute_clip_scale_kernel(
    const float* __restrict__ norm_sq_ptr,
    float* __restrict__ scale_ptr,
    float max_grad_norm)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float norm_sq = *norm_sq_ptr;
        float global_norm = sqrtf(norm_sq);
        float scale = (global_norm > max_grad_norm)
                    ? (max_grad_norm / fmaxf(global_norm, 1e-8f)) : 1.0f;
        *scale_ptr = scale;
    }
}

// GPU-fused gradient scaling: read scale from GPU memory (no CPU sync)
__global__ void scale_grads_auto_kernel(
    float* __restrict__ grads,
    int total_elements,
    const float* __restrict__ scale_ptr)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    grads[idx] *= *scale_ptr;
}

// ============================================================
// R23: Deterministic reduction kernels (two-pass, bit-exact)
// ============================================================

// MSE gradient kernel (deterministic variant): writes sq_err to per-pixel buffer
// instead of atomicAdd. Gradient computation is identical to mse_grad_kernel.
__global__ void mse_grad_deterministic_kernel(
    const float* __restrict__ output,
    const float* __restrict__ target,
    const float* __restrict__ mask,
    float* __restrict__ grad_output,
    float* __restrict__ sq_err_buffer,  // [C] per-pixel squared error
    int C,
    int channels,
    const float* __restrict__ rendered_alpha,
    const float* __restrict__ target_alpha,
    float* __restrict__ grad_alpha)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C) return;

    float m = mask[idx];
    float scale = 2.0f / (C * (float)channels);
    float sq_err = 0.0f;

    if (m < 0.5f) {
        grad_output[idx * 3 + 0] = 0.0f;
        grad_output[idx * 3 + 1] = 0.0f;
        grad_output[idx * 3 + 2] = 0.0f;
    } else {
        float dr = output[idx * 3 + 0] - target[idx * 3 + 0];
        float dg = output[idx * 3 + 1] - target[idx * 3 + 1];
        float db = output[idx * 3 + 2] - target[idx * 3 + 2];
        sq_err = dr * dr + dg * dg + db * db;
        grad_output[idx * 3 + 0] = dr * scale;
        grad_output[idx * 3 + 1] = dg * scale;
        grad_output[idx * 3 + 2] = db * scale;
    }

    if (channels == 4 && rendered_alpha && target_alpha && grad_alpha) {
        float da = rendered_alpha[idx] - target_alpha[idx];
        sq_err += da * da;
        grad_alpha[idx] = da * scale;
    } else if (grad_alpha) {
        grad_alpha[idx] = 0.0f;
    }

    sq_err_buffer[idx] = sq_err;
}

// Two-pass scalar reduction: Pass 1 (block tree reduction → partials)
__global__ void scalar_block_reduce_kernel(
    const float* __restrict__ input,
    float* __restrict__ partials,
    int C)
{
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < C) ? input[i] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partials[blockIdx.x] = sdata[0];
}

// Two-pass scalar reduction: Pass 2 (reduce partials → single output)
__global__ void scalar_final_reduce_kernel(
    const float* __restrict__ partials,
    float* __restrict__ output,
    int num_blocks)
{
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < num_blocks; i += blockDim.x)
        sum += partials[i];
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) output[0] = sdata[0];
}

// Deterministic clip_grad_norm: Pass 1 (write per-element g*g to buffer)
__global__ void clip_grad_norm_det_pass1_kernel(
    const float* __restrict__ grads,
    float* __restrict__ gg_buffer,
    int total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    float g = grads[idx];
    gg_buffer[idx] = g * g;
}

// Per-shape gradient reduction: sum partials across blocks for each shape
// Each thread handles one shape, reads [num_blocks] partials in fixed order.
__global__ void reduce_grad_partials_kernel(
    const float* __restrict__ partials,  // [num_blocks * N * 9]
    float* __restrict__ grad_cx,
    float* __restrict__ grad_cy,
    float* __restrict__ grad_rx,
    float* __restrict__ grad_ry,
    float* __restrict__ grad_angle,
    float* __restrict__ grad_colors,     // [N*3]
    float* __restrict__ grad_opacity,    // [N]
    int N, int num_blocks)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0;
    float s5 = 0, s6 = 0, s7 = 0, s8 = 0;
    for (int b = 0; b < num_blocks; b++) {
        int base = (b * N + i) * 9;
        s0 += partials[base + 0]; s1 += partials[base + 1];
        s2 += partials[base + 2]; s3 += partials[base + 3];
        s4 += partials[base + 4]; s5 += partials[base + 5];
        s6 += partials[base + 6]; s7 += partials[base + 7];
        s8 += partials[base + 8];
    }
    grad_cx[i] += s0; grad_cy[i] += s1;
    grad_rx[i] += s2; grad_ry[i] += s3;
    grad_angle[i] += s4;
    grad_colors[i * 3 + 0] += s5; grad_colors[i * 3 + 1] += s6;
    grad_colors[i * 3 + 2] += s7; grad_opacity[i] += s8;
}

// Per-shape logits gradient reduction: sum partials across blocks
__global__ void reduce_logits_partials_kernel(
    const float* __restrict__ partials,  // [num_blocks * N * num_types]
    float* __restrict__ grad_logits,     // [N * num_types]
    int N, int num_types, int num_blocks)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * num_types;
    if (idx >= total) return;
    int i = idx / num_types;
    int t = idx % num_types;
    float sum = 0.0f;
    for (int b = 0; b < num_blocks; b++)
        sum += partials[(b * N + i) * num_types + t];
    grad_logits[idx] += sum;
}

namespace vinylizer {

// Host-side sigmoid/logit helpers for upload/download parameter conversion.
// GPU d_colors/d_opacity are stored in LOGIT space; ShapeParams (host-facing)
// keeps legacy linear ranges (color [0,255], opacity [0,1]).
static inline float host_sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}
static inline float host_logit(float p) {
    float pc = std::clamp(p, 1e-5f, 1.0f - 1e-5f);
    return std::log(pc / (1.0f - pc));
}

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
    // 等待 viz stream 完成再释放
    cudaStreamSynchronize(s.viz_stream);
    cudaStreamDestroy(s.viz_stream);
    cudaEventDestroy(s.compute_done_event);
    cudaStreamDestroy(s.stream);

    // Free device memory
    cudaFree(s.d_px); cudaFree(s.d_py);
    cudaFree(s.d_cx); cudaFree(s.d_cy);
    cudaFree(s.d_rx); cudaFree(s.d_ry);
    cudaFree(s.d_angle); cudaFree(s.d_colors);
    cudaFree(s.d_opacity); cudaFree(s.d_sina); cudaFree(s.d_cosa);
    cudaFree(s.d_bg_color); cudaFree(s.d_output);
    // d_grad_cx 等是 d_grad_unpacked 的偏移指针，仅释放主 buffer（避免 double-free）
    cudaFree(s.d_grad_unpacked);
    cudaFree(s.d_grad_output);
    cudaFree(s.d_grad_lap_tmp);
    cudaFree(s.d_grads_packed);

    // GPU-resident Adam state (合并 buffer: d_adam_m_buffer + d_adam_v_buffer)
    cudaFree(s.d_adam_m_buffer);
    cudaFree(s.d_adam_v_buffer);

    cudaFree(s.d_type_logits); cudaFree(s.d_ste_type_logits);
    cudaFree(s.d_softmax_w); cudaFree(s.d_grad_logits);

    cudaFree(s.d_shape_types);

    // STE output
    cudaFree(s.d_ste_cx); cudaFree(s.d_ste_cy);
    cudaFree(s.d_ste_rx); cudaFree(s.d_ste_ry);
    cudaFree(s.d_ste_angle);
    cudaFree(s.d_ste_colors); cudaFree(s.d_ste_opacity);

    // Soft depth buffers
    cudaFree(s.d_z); cudaFree(s.d_ste_z); cudaFree(s.d_grad_z);
    cudaFree(s.d_m_z); cudaFree(s.d_v_z);

    // Frozen mask and gradient norm accumulator
    cudaFree(s.d_frozen_mask);
    cudaFree(s.d_grad_norm_sq);
    cudaFree(s.d_batch_scratch);
    cudaFree(s.d_all_active_indices);
    cudaFree(s.d_templates); cudaFree(s.d_blurred_templates); cudaFree(s.d_template_tmp);
    cudaFree(s.d_blur_kernel);
    cudaFree(s.d_reset_indices);
    cudaFree(s.d_target); cudaFree(s.d_mask);
    if (s.d_target_blurred) cudaFree(s.d_target_blurred);
    if (s.d_target_blur_tmp) cudaFree(s.d_target_blur_tmp);
    cudaFree(s.d_target_alpha); cudaFree(s.d_rendered_alpha); cudaFree(s.d_grad_alpha);
    cudaFree(s.d_visibility);
    if (s.d_visibility_full) cudaFree(s.d_visibility_full);
    if (s.d_grad_rgb_quadrant) cudaFree(s.d_grad_rgb_quadrant);
    cudaFree(s.d_canvas_u8);
    cudaFree(s.d_error_map_scratch);
    cudaFree(s.d_error_sum);
    cudaFree(s.d_mean_err);
    cudaFree(s.d_coverage_count);
    // SP-Loss buffers
    cudaFree(s.d_sp_S);
    cudaFree(s.d_sp_G);
    cudaFree(s.d_k_allowed);
    cudaFree(s.d_pf_grad_x);
    cudaFree(s.d_pf_grad_y);
    cudaFree(s.d_fh_vis_gate);
    // SP-GEO buffers
    cudaFree(s.d_sp_geo_field);
    cudaFree(s.d_sp_geo_grad_x);
    cudaFree(s.d_sp_geo_grad_y);
    cudaFree(s.d_aux_gate);
    cudaFree(s.d_mse_sum);

    // FA-Loss buffers
    cudaFree(s.d_fa_stats);
    cudaFree(s.d_fa_mu);
    cudaFree(s.d_fa_globals);

    // 双向渲染反向 buffer（仅在启用时分配，可能为 nullptr，cudaFree(nullptr) 是安全的）
    cudaFree(s.d_output_rev);
    cudaFree(s.d_rendered_alpha_rev);
    cudaFree(s.d_grad_output_rev);
    cudaFree(s.d_grad_alpha_rev);
    cudaFree(s.d_mse_sum_rev);

    // Free pyramid buffers
    cudaFree(s.d_pyr_rendered_1);
    cudaFree(s.d_pyr_rendered_2);
    cudaFree(s.d_pyr_target_1);
    cudaFree(s.d_pyr_target_2);
    cudaFree(s.d_pyr_grad_tmp);

    // Free tile acceleration structures
    cudaFree(s.d_tile_shape_lists);
    cudaFree(s.d_tile_shape_counts);
    cudaFree(s.d_tile_overflow);

    // Alpha LUT texture cleanup
    if (s.alpha_tex) cudaDestroyTextureObject(s.alpha_tex);
    if (s.alpha_array) cudaFreeArray(s.alpha_array);

    // R23: Free deterministic reduction buffers
    cudaFree(s.d_sq_err_buffer);
    cudaFree(s.d_gg_buffer);
    cudaFree(s.d_scalar_partials);
    cudaFree(s.d_grad_partials);
    cudaFree(s.d_grad_logits_partials);

    // Free async MSE readback resources
    if (h_mse_pinned_) cudaFreeHost(h_mse_pinned_);
    if (mse_event_) cudaEventDestroy(mse_event_);
    if (d_clip_scale_) cudaFree(d_clip_scale_);

#ifndef __NODEBUG__
    // Free async debug statistics buffers
    for (int b = 0; b < 2; ++b) {
        if (h_debug_logits_pinned_[b]) cudaFreeHost(h_debug_logits_pinned_[b]);
        if (h_debug_grads_pinned_[b]) cudaFreeHost(h_debug_grads_pinned_[b]);
        if (debug_events_[b]) cudaEventDestroy(debug_events_[b]);
    }
#endif
}

void DiffRenderer::init(int canvas_w, int canvas_h, int max_shapes, int num_template_types, int num_builtins) {
    canvas_w_ = canvas_w;
    canvas_h_ = canvas_h;
    max_shapes_ = max_shapes;

    auto& s = scratch_;
    s.num_template_types = num_template_types;
    s.num_builtins = num_builtins;
    s.total_pixels = canvas_w * canvas_h;

    // VRAM budget check for fp16 alpha scratch (2 bytes per shape-pixel)
    size_t free_bytes, total_bytes;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
        size_t fixed_alloc = static_cast<size_t>(max_shapes) * 324ULL
                           + static_cast<size_t>(s.total_pixels) * 75ULL
                           + 3145728ULL + 1024ULL;
        fixed_alloc = static_cast<size_t>(fixed_alloc * 1.1);
        size_t budget_80 = static_cast<size_t>(static_cast<double>(free_bytes) * 0.80);
        size_t scratch_needed = static_cast<size_t>(max_shapes) * s.total_pixels * 2ULL;
        if (scratch_needed > (budget_80 > fixed_alloc ? budget_80 - fixed_alloc : 0ULL)) {
            VIN_WARN("VRAM may be insufficient for fp16 scratch: need %zu MB, have %zu MB budget",
                     scratch_needed / (1024 * 1024),
                     (budget_80 > fixed_alloc ? budget_80 - fixed_alloc : 0ULL) / (1024 * 1024));
        }
    }

    // Create CUDA stream
    CUDA_CHECK(cudaStreamCreate(&s.stream));
    // 独立 viz stream：D2H copy 不阻塞 compute stream
    CUDA_CHECK(cudaStreamCreate(&s.viz_stream));
    CUDA_CHECK(cudaEventCreate(&s.compute_done_event));

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
    float bg[3] = {255.0f, 255.0f, 255.0f};  // white background (align with diffbmp)
    CUDA_CHECK(cudaMemcpy(s.d_bg_color, bg, 3 * sizeof(float), cudaMemcpyHostToDevice));

    // Output buffer
    CUDA_CHECK(cudaMalloc(&s.d_output, s.total_pixels * 3 * sizeof(float)));

    // Gradient buffers: 合并为单个 SoA buffer [max_shapes*9]，偏移指针指向各段
    // 布局: cx(max_shapes), cy(max_shapes), rx(max_shapes), ry(max_shapes),
    //       angle(max_shapes), colors(max_shapes*3), opacity(max_shapes)
    CUDA_CHECK(cudaMalloc(&s.d_grad_unpacked, (size_t)max_shapes * 9 * sizeof(float)));
    s.d_grad_cx      = s.d_grad_unpacked + 0;
    s.d_grad_cy      = s.d_grad_unpacked + max_shapes;
    s.d_grad_rx      = s.d_grad_unpacked + max_shapes * 2;
    s.d_grad_ry      = s.d_grad_unpacked + max_shapes * 3;
    s.d_grad_angle   = s.d_grad_unpacked + max_shapes * 4;
    s.d_grad_colors  = s.d_grad_unpacked + max_shapes * 5;
    s.d_grad_opacity = s.d_grad_unpacked + max_shapes * 8;
    CUDA_CHECK(cudaMalloc(&s.d_grad_output, s.total_pixels * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_grad_lap_tmp, s.total_pixels * 3 * sizeof(float)));

    // Packed gradient buffer for single-download
    CUDA_CHECK(cudaMalloc(&s.d_grads_packed, (size_t)max_shapes * (9 + num_template_types) * sizeof(float)));

    // GPU-resident Adam state (合并为 2 个 buffer: m 和 v)
    auto alloc_adam_buf = [&](float** ptr) {
        CUDA_CHECK(cudaMalloc(ptr, max_shapes * sizeof(float)));
        CUDA_CHECK(cudaMemset(*ptr, 0, max_shapes * sizeof(float)));
    };
    {
        size_t adam_buf_size = (size_t)max_shapes * (9 + num_template_types) * sizeof(float);
        CUDA_CHECK(cudaMalloc(&s.d_adam_m_buffer, adam_buf_size));
        CUDA_CHECK(cudaMemset(s.d_adam_m_buffer, 0, adam_buf_size));
        CUDA_CHECK(cudaMalloc(&s.d_adam_v_buffer, adam_buf_size));
        CUDA_CHECK(cudaMemset(s.d_adam_v_buffer, 0, adam_buf_size));

        // 设置偏移指针（m buffer: cx=0, cy=1, rx=2, ry=3, angle=4, cr=5, cg=6, cb=7, ca=8）
        s.d_m_cx    = s.d_adam_m_buffer + max_shapes * 0;
        s.d_m_cy    = s.d_adam_m_buffer + max_shapes * 1;
        s.d_m_rx    = s.d_adam_m_buffer + max_shapes * 2;
        s.d_m_ry    = s.d_adam_m_buffer + max_shapes * 3;
        s.d_m_angle = s.d_adam_m_buffer + max_shapes * 4;
        s.d_m_cr    = s.d_adam_m_buffer + max_shapes * 5;
        s.d_m_cg    = s.d_adam_m_buffer + max_shapes * 6;
        s.d_m_cb    = s.d_adam_m_buffer + max_shapes * 7;
        s.d_m_ca    = s.d_adam_m_buffer + max_shapes * 8;

        // 设置 logits Adam state 基指针 (替代 d_m_l0..d_m_l3)
        s.d_m_logits = s.d_adam_m_buffer + max_shapes * 9;
        s.d_v_logits = s.d_adam_v_buffer + max_shapes * 9;

        // 设置偏移指针（v buffer）
        s.d_v_cx    = s.d_adam_v_buffer + max_shapes * 0;
        s.d_v_cy    = s.d_adam_v_buffer + max_shapes * 1;
        s.d_v_rx    = s.d_adam_v_buffer + max_shapes * 2;
        s.d_v_ry    = s.d_adam_v_buffer + max_shapes * 3;
        s.d_v_angle = s.d_adam_v_buffer + max_shapes * 4;
        s.d_v_cr    = s.d_adam_v_buffer + max_shapes * 5;
        s.d_v_cg    = s.d_adam_v_buffer + max_shapes * 6;
        s.d_v_cb    = s.d_adam_v_buffer + max_shapes * 7;
        s.d_v_ca    = s.d_adam_v_buffer + max_shapes * 8;
    }

    // STE quantized output buffers
    alloc_adam_buf(&s.d_ste_cx); alloc_adam_buf(&s.d_ste_cy);
    alloc_adam_buf(&s.d_ste_rx); alloc_adam_buf(&s.d_ste_ry);
    alloc_adam_buf(&s.d_ste_angle);
    CUDA_CHECK(cudaMalloc(&s.d_ste_colors, max_shapes * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemset(s.d_ste_colors, 0, max_shapes * 3 * sizeof(float)));
    alloc_adam_buf(&s.d_ste_opacity);

    // === Soft depth buffers (always allocated; used only when soft_depth_enabled) ===
    // d_z / d_ste_z: learnable depth scalar per shape (z 大 = 底层)
    // d_grad_z: gradient accumulator (atomic, zeroed each iter)
    // d_m_z / d_v_z: Adam first/second moments
    CUDA_CHECK(cudaMalloc(&s.d_z, max_shapes * sizeof(float)));
    CUDA_CHECK(cudaMemset(s.d_z, 0, max_shapes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_ste_z, max_shapes * sizeof(float)));
    CUDA_CHECK(cudaMemset(s.d_ste_z, 0, max_shapes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_grad_z, max_shapes * sizeof(float)));
    CUDA_CHECK(cudaMemset(s.d_grad_z, 0, max_shapes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_m_z, max_shapes * sizeof(float)));
    CUDA_CHECK(cudaMemset(s.d_m_z, 0, max_shapes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_v_z, max_shapes * sizeof(float)));
    CUDA_CHECK(cudaMemset(s.d_v_z, 0, max_shapes * sizeof(float)));

    // Frozen mask
    CUDA_CHECK(cudaMalloc(&s.d_frozen_mask, max_shapes * sizeof(int)));
    CUDA_CHECK(cudaMemset(s.d_frozen_mask, 0, max_shapes * sizeof(int)));

    // Gradient clip norm accumulator
    CUDA_CHECK(cudaMalloc(&s.d_grad_norm_sq, sizeof(float)));

    // Tile-based rendering acceleration (compact batch scratch replaces full-size per-shape alpha cache)
    {
        std::vector<int> h_all_active(max_shapes);
        for (int i = 0; i < max_shapes; i++) h_all_active[i] = i;
        CUDA_CHECK(cudaMalloc(&s.d_all_active_indices, max_shapes * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(s.d_all_active_indices, h_all_active.data(), max_shapes * sizeof(int), cudaMemcpyHostToDevice));

        s.tile_size = 16;
        s.max_shapes_per_tile = 2048;
        s.num_tiles_x = (canvas_w + s.tile_size - 1) / s.tile_size;
        s.num_tiles_y = (canvas_h + s.tile_size - 1) / s.tile_size;
        s.num_tiles = s.num_tiles_x * s.num_tiles_y;

        CUDA_CHECK(cudaMalloc(&s.d_tile_shape_lists,
            (size_t)s.num_tiles * s.max_shapes_per_tile * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&s.d_tile_shape_counts, s.num_tiles * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&s.d_tile_overflow, sizeof(int)));

        h_tile_counts_.resize(s.num_tiles);
        batch_descs_.reserve(s.num_tiles);

        VIN_INFO("Tile-based: %dx%d tiles, tile lists %.1f MB",
                 s.num_tiles_x, s.num_tiles_y,
                 (float)s.num_tiles * s.max_shapes_per_tile * sizeof(int) / (1024 * 1024));
    }

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
    CUDA_CHECK(cudaMalloc(&s.d_visibility_full, max_shapes * sizeof(float)));

    // Per-shape per-quadrant RGB gradient buffer [N*4*3] (for split analysis)
    CUDA_CHECK(cudaMalloc(&s.d_grad_rgb_quadrant,
        (size_t)max_shapes * 4 * 3 * sizeof(float)));

    // Canvas uint8 buffer for async visualization
    CUDA_CHECK(cudaMalloc(&s.d_canvas_u8, s.total_pixels * 3 * sizeof(uint8_t)));

    // Per-pixel error map scratch (for relocate GPU path & EM-YMP)
    CUDA_CHECK(cudaMalloc(&s.d_error_map_scratch, s.total_pixels * sizeof(float)));

    // EM-YMP: 误差归约与均值标量
    CUDA_CHECK(cudaMalloc(&s.d_error_sum, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_mean_err, sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(s.d_mean_err, 0, sizeof(float), s.stream));

    // Per-pixel coverage count (diagnosis only)
    CUDA_CHECK(cudaMalloc(&s.d_coverage_count, s.total_pixels * sizeof(int)));
    CUDA_CHECK(cudaMemset(s.d_coverage_count, 0, s.total_pixels * sizeof(int)));

    // SP-Loss: S(p) 累加器 [total_pixels] + K_allowed(p) 预计算 [total_pixels]
    CUDA_CHECK(cudaMalloc(&s.d_sp_S, s.total_pixels * sizeof(float)));
    CUDA_CHECK(cudaMemset(s.d_sp_S, 0, s.total_pixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_sp_G, s.total_pixels * sizeof(float)));
    CUDA_CHECK(cudaMemset(s.d_sp_G, 0, s.total_pixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_k_allowed, s.total_pixels * sizeof(float)));
    CUDA_CHECK(cudaMemset(s.d_k_allowed, 0, s.total_pixels * sizeof(float)));

    // Error potential field gradient buffers (normalized grad U, uploaded per refresh)
    CUDA_CHECK(cudaMalloc(&s.d_pf_grad_x, s.total_pixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_pf_grad_y, s.total_pixels * sizeof(float)));
    CUDA_CHECK(cudaMemset(s.d_pf_grad_x, 0, s.total_pixels * sizeof(float)));
    CUDA_CHECK(cudaMemset(s.d_pf_grad_y, 0, s.total_pixels * sizeof(float)));

    // FH-PF vis gate buffer [N] (per-shape, recomputed every fh_pf_update_interval)
    CUDA_CHECK(cudaMalloc(&s.d_fh_vis_gate, max_shapes * sizeof(float)));
    CUDA_CHECK(cudaMemset(s.d_fh_vis_gate, 0, max_shapes * sizeof(float)));

    // SP-GEO buffers: d(p) field + normalized gradient (host-computed, uploaded per refresh)
    CUDA_CHECK(cudaMalloc(&s.d_sp_geo_field, s.total_pixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_sp_geo_grad_x, s.total_pixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_sp_geo_grad_y, s.total_pixels * sizeof(float)));
    CUDA_CHECK(cudaMemset(s.d_sp_geo_field, 0, s.total_pixels * sizeof(float)));
    CUDA_CHECK(cudaMemset(s.d_sp_geo_grad_x, 0, s.total_pixels * sizeof(float)));
    CUDA_CHECK(cudaMemset(s.d_sp_geo_grad_y, 0, s.total_pixels * sizeof(float)));

    // Aux-Vis gate buffer [N] (per-shape g_i, recomputed every iter from d_visibility)
    CUDA_CHECK(cudaMalloc(&s.d_aux_gate, max_shapes * sizeof(float)));
    CUDA_CHECK(cudaMemset(s.d_aux_gate, 0, max_shapes * sizeof(float)));

    // MSE sum accumulator (pre-allocated)
    CUDA_CHECK(cudaMalloc(&s.d_mse_sum, sizeof(float)));

    // === FA-Loss buffers ===
    // d_fa_stats: [max_shapes*5] = {sum_w, sum_wI_r, sum_wI_g, sum_wI_b, sum_wI2}
    // d_fa_mu:    [max_shapes*3] 每形状 μ_i 颜色（backward kernel 读取）
    // d_fa_globals: [3] = {fa_scale=λ/v, fa_mean_t=T/s, fa_vis_weight}（finalize 后写入）
    //   量级修正：fa_scale=λ/v（去 s 归一化，原 λ/(v·s) 使梯度小 ~1e9 倍）
    //   可见性修正：fa_vis_weight=1 时权重乘 T_k(p)，被遮挡形状不被误伤
    CUDA_CHECK(cudaMalloc(&s.d_fa_stats, (size_t)max_shapes * 5 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_fa_mu, (size_t)max_shapes * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_fa_globals, 3 * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(s.d_fa_globals, 0, 3 * sizeof(float), s.stream));

    // 双向渲染反向 buffer（仅在启用时分配）
    if (bidir_lambda_ > 0.0f) {
        CUDA_CHECK(cudaMalloc(&s.d_output_rev, s.total_pixels * 3 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s.d_rendered_alpha_rev, s.total_pixels * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s.d_grad_output_rev, s.total_pixels * 3 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s.d_grad_alpha_rev, s.total_pixels * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s.d_mse_sum_rev, sizeof(float)));
        VIN_INFO("Bidirectional rendering enabled: lambda=%.4f", bidir_lambda_);
    }

    // Pre-allocate pyramid buffers for multi-scale perceptual loss
    // Level 1: H/2 * W/2 * 3, Level 2: H/4 * W/4 * 3
    {
        int h1 = canvas_h / 2, w1 = canvas_w / 2;
        int h2 = h1 / 2, w2 = w1 / 2;
        CUDA_CHECK(cudaMalloc(&s.d_pyr_rendered_1, h1 * w1 * 3 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s.d_pyr_rendered_2, h2 * w2 * 3 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s.d_pyr_target_1, h1 * w1 * 3 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s.d_pyr_target_2, h2 * w2 * 3 * sizeof(float)));
        // Grad temp buffer sized for largest pyramid level (level 1 = H/2 * W/2 * 3)
        CUDA_CHECK(cudaMalloc(&s.d_pyr_grad_tmp, h1 * w1 * 3 * sizeof(float)));
    }

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

    // CPU-side vectors for param upload/download
    h_cx_.resize(max_shapes);
    h_cy_.resize(max_shapes);
    h_rx_.resize(max_shapes);
    h_ry_.resize(max_shapes);

    // Pre-allocate upload buffers
    h_angle_buf_.resize(max_shapes);
    h_colors_buf_.resize(max_shapes * 3);
    h_opacity_buf_.resize(max_shapes);

    // Pre-allocated buffer for batch Adam reset indices
    CUDA_CHECK(cudaMalloc(&s.d_reset_indices, max_shapes * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&s.d_type_logits, (size_t)max_shapes * num_template_types * sizeof(float)));
    CUDA_CHECK(cudaMemset(s.d_type_logits, 0, (size_t)max_shapes * num_template_types * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_ste_type_logits, (size_t)max_shapes * num_template_types * sizeof(float)));
    CUDA_CHECK(cudaMemset(s.d_ste_type_logits, 0, (size_t)max_shapes * num_template_types * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_softmax_w, (size_t)max_shapes * num_template_types * sizeof(float)));
    CUDA_CHECK(cudaMemset(s.d_softmax_w, 0, (size_t)max_shapes * num_template_types * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.d_grad_logits, (size_t)max_shapes * num_template_types * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&s.d_shape_types, max_shapes * sizeof(int)));
    std::vector<int> default_types(max_shapes, DARTS_MODE);
    CUDA_CHECK(cudaMemcpy(s.d_shape_types, default_types.data(), max_shapes * sizeof(int), cudaMemcpyHostToDevice));

    // Template sampling buffers
    {
        const int tmpl_size = num_template_types * TEMPLATE_SIZE * TEMPLATE_SIZE;
        CUDA_CHECK(cudaMalloc(&s.d_templates, tmpl_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s.d_blurred_templates, tmpl_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s.d_template_tmp, tmpl_size * sizeof(float)));
        // Pre-allocate Gaussian kernel buffer (max ksize = 2*ceil(3*42)+1 = 255，支持双模糊 sigma_max=16+)
        CUDA_CHECK(cudaMalloc(&s.d_blur_kernel, 256 * sizeof(float)));
        VIN_INFO("Template sampling buffers allocated: %d templates, %dx%d", num_template_types, TEMPLATE_SIZE, TEMPLATE_SIZE);
    }

    // 初始化内置模板（仅当 num_builtins > 0 时）
    if (num_builtins == 4) {
        cuda_template_init(s.d_templates);
    }

    // === R23: 确定性归约缓冲区分配（仅 deterministic_reduce_ 开启时）===
    if (deterministic_reduce_) {
        s.deterministic_reduce = 1;
        // MSE 逐像素 sq_err 缓冲
        CUDA_CHECK(cudaMalloc(&s.d_sq_err_buffer, s.total_pixels * sizeof(float)));
        // norm_sq 逐元素 g*g 缓冲（d_grads_packed 大小 = N*13）
        CUDA_CHECK(cudaMalloc(&s.d_gg_buffer, (size_t)max_shapes * 13 * sizeof(float)));
        // 标量归约 partials（足够大覆盖 total_pixels / 256 blocks）
        s.max_blocks_scalar = (s.total_pixels + 255) / 256;
        CUDA_CHECK(cudaMalloc(&s.d_scalar_partials, s.max_blocks_scalar * sizeof(float)));
        // 梯度归约 partials（num_tiles blocks × N × 9）
        s.max_blocks_grad = s.num_tiles;
        CUDA_CHECK(cudaMalloc(&s.d_grad_partials,
            (size_t)s.max_blocks_grad * max_shapes * 9 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s.d_grad_logits_partials,
            (size_t)s.max_blocks_grad * max_shapes * s.num_template_types * sizeof(float)));
        VIN_INFO("R23: Deterministic reduction enabled (partial buffers allocated)");
    }

    initialized_ = true;

    // Async MSE readback resources (pinned memory + event for DMA, lag 1 iter)
    CUDA_CHECK(cudaMallocHost(&h_mse_pinned_, sizeof(float)));
    CUDA_CHECK(cudaEventCreate(&mse_event_));
    // Allocate GPU-resident scale factor for gradient clipping (avoids CPU sync)
    CUDA_CHECK(cudaMalloc(&d_clip_scale_, sizeof(float)));

#ifndef __NODEBUG__
    // Allocate async debug statistics buffers (double-buffered pinned memory + events)
    {
        size_t logits_bytes = (size_t)max_shapes * s.num_template_types * sizeof(float);
        size_t grads_bytes = (size_t)max_shapes * (9 + s.num_template_types) * sizeof(float);
        for (int b = 0; b < 2; ++b) {
            CUDA_CHECK(cudaMallocHost(&h_debug_logits_pinned_[b], logits_bytes));
            CUDA_CHECK(cudaMallocHost(&h_debug_grads_pinned_[b], grads_bytes));
            CUDA_CHECK(cudaEventCreate(&debug_events_[b]));
            debug_first_[b] = true;
        }
        debug_buf_idx_ = 0;
    }
#endif
    VIN_INFO("DiffRenderer initialized: %dx%d, max_shapes=%d",
             canvas_w, canvas_h, max_shapes);
}

void DiffRenderer::load_external_stamps(
    const std::vector<std::string>& paths)
{
    auto& s = scratch_;
    const int S = TEMPLATE_SIZE;
    const int S2 = S * S;

    // 内置模板已在 init() 中通过 cuda_template_init 加载，此函数仅加载外部图章
    int external_count = (int)paths.size();
    int builtin_count = s.num_builtins;
    int total = builtin_count + external_count;

    if (total > MAX_TEMPLATE_TYPES) {
        VIN_ERROR("Too many templates: %d > MAX_TEMPLATE_TYPES (%d)", total, MAX_TEMPLATE_TYPES);
        return;
    }

    // 仅加载外部图章，从内置模板之后的位置开始
    std::vector<float> h_stamps(external_count * S2, 0.0f);

    for (int i = 0; i < external_count; i++) {
        // 步骤1: 读取图片
        cv::Mat img = imread_utf8(paths[i], cv::IMREAD_UNCHANGED);
        if (img.empty()) {
            VIN_ERROR("Failed to load stamp (file not found, unsupported format, or path encoding error): %s", paths[i].c_str());
            continue;
        }

        // 步骤2: 强制 RGBA + 保持宽高比缩放 + 补零到正方形
        cv::Mat padded = resize_stamp_keep_aspect(img, S);

        // 步骤3: 提取 alpha（pydiffbmp 逻辑）
        std::vector<float> alpha = extract_stamp_alpha(padded);

        // 复制到 h_stamps
        std::copy(alpha.begin(), alpha.end(), h_stamps.data() + i * S2);
    }

    // 写入内置模板之后的位置
    cudaMemcpy(s.d_templates + builtin_count * S2, h_stamps.data(),
               external_count * S2 * sizeof(float), cudaMemcpyHostToDevice);
    VIN_INFO("External stamps loaded: %d builtin + %d external = %d total",
             builtin_count, external_count, total);
}

void DiffRenderer::set_target(const float* h_target, const float* h_mask, const float* h_target_alpha) {
    auto& s = scratch_;
    CUDA_CHECK(cudaMemcpy(s.d_target, h_target, s.total_pixels * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.d_mask, h_mask, s.total_pixels * sizeof(float), cudaMemcpyHostToDevice));
    // EM-YMP: 统计 mask 内有效像素数（用于 mean_err = sum(error_map)/valid_count）
    valid_pixel_count_ = 0;
    for (int p = 0; p < s.total_pixels; ++p) {
        if (h_mask[p] > 0.5f) valid_pixel_count_++;
    }
    if (valid_pixel_count_ <= 0) valid_pixel_count_ = s.total_pixels;  // 防除零兜底

    // === FA-Loss: 预计算全图目标方差 σ²_global ===
    // μ_global = Σ_p mask(p)·I(p) / Σ_p mask(p)
    // σ²_global = Σ_p mask(p)·||I(p)-μ_global||² / Σ_p mask(p)
    // 这是像素级 RGB 方差均值，用于归一化每形状的局部方差 σ²_i。
    // 在 host 一次性计算（O(total_pixels)，与 mask 统计同量级）。
    if (s.fa_lambda > 0.0f) {
        double sum_r = 0.0, sum_g = 0.0, sum_b = 0.0;
        double sum_sq = 0.0;
        long long cnt = 0;
        for (int p = 0; p < s.total_pixels; ++p) {
            if (h_mask[p] < 0.5f) continue;
            float ir = h_target[p * 3 + 0];
            float ig = h_target[p * 3 + 1];
            float ib = h_target[p * 3 + 2];
            sum_r += ir; sum_g += ig; sum_b += ib;
            sum_sq += (double)ir * ir + (double)ig * ig + (double)ib * ib;
            cnt++;
        }
        if (cnt > 0) {
            double mr = sum_r / cnt, mg = sum_g / cnt, mb = sum_b / cnt;
            double e2 = sum_sq / cnt;
            double mu_sq = mr * mr + mg * mg + mb * mb;
            double var = e2 - mu_sq;
            if (var < 0.0) var = 0.0;
            s.fa_sigma2_global = (float)var;
        } else {
            s.fa_sigma2_global = 1.0f;
        }
        VIN_INFO("FA-Loss: sigma2_global = %.2f (valid_pixels=%lld, lambda=%.4f)",
                 s.fa_sigma2_global, cnt, s.fa_lambda);
    }
    if (h_target_alpha) {
        CUDA_CHECK(cudaMemcpy(s.d_target_alpha, h_target_alpha, s.total_pixels * sizeof(float), cudaMemcpyHostToDevice));
    } else {
        // Default: all opaque (target_alpha = 255)
        std::vector<float> ones(s.total_pixels, 255.0f);
        CUDA_CHECK(cudaMemcpy(s.d_target_alpha, ones.data(), s.total_pixels * sizeof(float), cudaMemcpyHostToDevice));
    }

    // === SP-Loss: 预计算 K_allowed(p) = K_base + β · Var_7x7(p) / mean(Var_7x7) ===
    // 7×7 局部方差图（RGB 三通道平均），一次性预计算上传。
    // 高复杂度区（边缘/纹理）局部方差大 → K_allowed 大 → 允许更多堆叠；
    // 平滑区方差小 → K_allowed ≈ K_base → 惩罚堆叠。
    // SP-GEO 也需要 K_allowed（host 端算 d(p)=max(0,S-C-K)），故 sp_geo_active 时同样预计算。
    if ((s.sp_lambda > 0.0f || s.sp_geo_active || s.sp_geo_loss_active) && s.d_k_allowed) {
        const int W = canvas_w_, H = canvas_h_;
        const int r = 3;  // 7×7 窗口半径
        std::vector<float> local_var(s.total_pixels, 0.0f);
        double sum_var = 0.0;
        int valid_cnt = 0;
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                int p = y * W + x;
                if (h_mask[p] < 0.5f) { local_var[p] = 0.0f; continue; }
                double sum_r = 0, sum_g = 0, sum_b = 0;
                double sum_r2 = 0, sum_g2 = 0, sum_b2 = 0;
                int cnt = 0;
                for (int dy = -r; dy <= r; dy++) {
                    int ny = std::min(std::max(y + dy, 0), H - 1);
                    for (int dx = -r; dx <= r; dx++) {
                        int nx = std::min(std::max(x + dx, 0), W - 1);
                        int np = ny * W + nx;
                        if (h_mask[np] < 0.5f) continue;
                        float ir = h_target[np * 3 + 0];
                        float ig = h_target[np * 3 + 1];
                        float ib = h_target[np * 3 + 2];
                        sum_r += ir; sum_g += ig; sum_b += ib;
                        sum_r2 += (double)ir * ir;
                        sum_g2 += (double)ig * ig;
                        sum_b2 += (double)ib * ib;
                        cnt++;
                    }
                }
                if (cnt > 1) {
                    double mr = sum_r / cnt, mg = sum_g / cnt, mb = sum_b / cnt;
                    double vr = sum_r2 / cnt - mr * mr;
                    double vg = sum_g2 / cnt - mg * mg;
                    double vb = sum_b2 / cnt - mb * mb;
                    if (vr < 0) vr = 0;
                    if (vg < 0) vg = 0;
                    if (vb < 0) vb = 0;
                    local_var[p] = (float)((vr + vg + vb) / 3.0);
                    sum_var += local_var[p];
                    valid_cnt++;
                }
            }
        }
        float mean_var = (valid_cnt > 0) ? (float)(sum_var / valid_cnt) : 1e-6f;
        if (mean_var < 1e-6f) mean_var = 1e-6f;
        s.sp_mean_var = mean_var;
        // K_allowed(p) = K_base + β · Var(p) / mean_var
        std::vector<float> k_allowed(s.total_pixels, 0.0f);
        for (int p = 0; p < s.total_pixels; p++) {
            float ratio = local_var[p] / mean_var;
            k_allowed[p] = s.sp_k_base + s.sp_beta * ratio;
        }
        CUDA_CHECK(cudaMemcpy(s.d_k_allowed, k_allowed.data(),
                              s.total_pixels * sizeof(float), cudaMemcpyHostToDevice));
        VIN_INFO("SP-Loss: mean_var=%.2f, K_base=%.2f, beta=%.2f, lambda=%.4f (valid=%d)",
                 mean_var, s.sp_k_base, s.sp_beta, s.sp_lambda, valid_cnt);
    }
}

void DiffRenderer::set_bg_color(float r, float g, float b) {
    auto& s = scratch_;
    float bg[3] = {r, g, b};
    if (s.d_bg_color) {
        CUDA_CHECK(cudaMemcpy(s.d_bg_color, bg, 3 * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void DiffRenderer::set_blur_sigma(float sigma) {
    blur_sigma_ = sigma;
}

void DiffRenderer::set_target_blur_enabled(bool enabled) {
    auto& s = scratch_;
    target_blur_enabled_ = enabled;
    if (enabled && initialized_ && !s.d_target_blurred) {
        CUDA_CHECK(cudaMalloc(&s.d_target_blurred, s.total_pixels * 3 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s.d_target_blur_tmp, s.total_pixels * 3 * sizeof(float)));
        VIN_INFO("Target blur enabled: allocated %d x 3 floats x2 (W=%d, H=%d)",
                 s.total_pixels, canvas_w_, canvas_h_);
    }
}

void DiffRenderer::apply_uniform_scale(int N) {
    auto& s = scratch_;
    if (N <= 0 || !s.d_rx || !s.d_ry) return;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    uniform_scale_kernel<<<blocks, threads, 0, s.stream>>>(
        s.d_rx, s.d_ry, N);
}

void DiffRenderer::compute_visibility(int N) {
    auto& s = scratch_;
    if (!initialized_ || N == 0) return;

    CUDA_CHECK(cudaMemsetAsync(s.d_visibility, 0, N * sizeof(float), s.stream));
    CUDA_CHECK(cudaMemsetAsync(s.d_visibility_full, 0, N * sizeof(float), s.stream));

    cuda_fused_template_visibility(
        s.d_px, s.d_py, s.d_ste_cx, s.d_ste_cy, s.d_ste_rx, s.d_ste_ry,
        s.d_sina, s.d_cosa,
        s.d_shape_types, s.d_ste_opacity,
        s.d_blurred_templates,
        s.d_visibility,
        s.d_visibility_full,
        s.d_all_active_indices, N, N, s.total_pixels,
        s.d_softmax_w, s.render_mode, blur_sigma_,
        s.stream, s.num_template_types, s.num_builtins);
}

void DiffRenderer::download_visibility(float* h_visibility, int N) {
    auto& s = scratch_;
    if (!initialized_ || !s.d_visibility) return;
    CUDA_CHECK(cudaMemcpyAsync(h_visibility, s.d_visibility,
                N * sizeof(float), cudaMemcpyDeviceToHost, s.stream));
    CUDA_CHECK(cudaStreamSynchronize(s.stream));
}

void DiffRenderer::download_visibility_full(float* h_visibility_full, int N) {
    auto& s = scratch_;
    if (!initialized_ || !s.d_visibility_full) {
        std::memset(h_visibility_full, 0, N * sizeof(float));
        return;
    }
    CUDA_CHECK(cudaMemcpyAsync(h_visibility_full, s.d_visibility_full,
                N * sizeof(float), cudaMemcpyDeviceToHost, s.stream));
    CUDA_CHECK(cudaStreamSynchronize(s.stream));
}

void DiffRenderer::download_grad_rgb_quadrant(float* h_grad, int N) {
    auto& s = scratch_;
    if (!initialized_ || !s.d_grad_rgb_quadrant) return;
    CUDA_CHECK(cudaMemcpyAsync(h_grad, s.d_grad_rgb_quadrant,
                N * 4 * 3 * sizeof(float), cudaMemcpyDeviceToHost, s.stream));
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

void DiffRenderer::set_loss_config(float perceptual_weight, int pyramid_levels,
                                    float alpha_loss_weight, float alpha_sparsity_weight) {
    auto& s = scratch_;
    s.perceptual_weight = perceptual_weight;
    s.pyramid_levels = pyramid_levels;
    s.alpha_loss_weight = alpha_loss_weight;
    s.alpha_sparsity_weight = alpha_sparsity_weight;
    VIN_INFO("Loss config: perceptual_weight=%.2f pyramid_levels=%d alpha_loss=%.2f alpha_sparse=%.2f",
             perceptual_weight, pyramid_levels, alpha_loss_weight, alpha_sparsity_weight);
}

float DiffRenderer::compute_mse_and_grad(int N) {
    auto& s = scratch_;

    int channels = s.alpha_loss_enabled ? 4 : 3;

    // 双模糊退火：训练损失用模糊目标；未开启时用原始目标（baseline 行为不变）
    const float* d_target_eff =
        (target_blur_enabled_ && s.d_target_blurred) ? s.d_target_blurred : s.d_target;

    // Use pre-allocated MSE sum buffer
    CUDA_CHECK(cudaMemsetAsync(s.d_mse_sum, 0, sizeof(float), s.stream));

    int threads = 256;
    int blocks = (s.total_pixels + threads - 1) / threads;
    if (deterministic_reduce_) {
        // R23: 确定性 MSE 归约 — 写逐像素 sq_err 到缓冲，再两遍归约
        mse_grad_deterministic_kernel<<<blocks, threads, 0, s.stream>>>(
            s.d_output, d_target_eff, s.d_mask,
            s.d_grad_output, s.d_sq_err_buffer, s.total_pixels, channels,
            s.d_rendered_alpha, s.d_target_alpha, s.d_grad_alpha);
        CUDA_KERNEL_CHECK();
        // Two-pass reduction: sq_err_buffer → scalar_partials → d_mse_sum
        scalar_block_reduce_kernel<<<blocks, threads, 0, s.stream>>>(
            s.d_sq_err_buffer, s.d_scalar_partials, s.total_pixels);
        CUDA_KERNEL_CHECK();
        int reduce_blocks = (blocks + 255) / 256;
        scalar_final_reduce_kernel<<<1, 256, 0, s.stream>>>(
            s.d_scalar_partials, s.d_mse_sum, blocks);
        CUDA_KERNEL_CHECK();
    } else {
        mse_grad_kernel<<<blocks, threads, 0, s.stream>>>(
            s.d_output, d_target_eff, s.d_mask,
            s.d_grad_output, s.d_mse_sum, s.total_pixels, channels,
            s.d_rendered_alpha, s.d_target_alpha, s.d_grad_alpha);
        CUDA_KERNEL_CHECK();
    }

    // === Gradient Loss（残差拉普拉斯）：λ>0 时 g(p) ← g(p) - λ·Δg(p) ===
    // 仅替换 backward 的像素梯度缓冲；MSE 数值（d_mse_sum）与 forward 渲染不受影响。
    // 在 pyramid 感知损失之前应用：拉普拉斯项只作用于全分辨率基础残差（数学定义如此）。
    if (s.grad_lambda > 0.0f) {
        dim3 lap_blk(16, 16);
        dim3 lap_grd((canvas_w_ + 15) / 16, (canvas_h_ + 15) / 16);
        grad_loss_laplacian_kernel<<<lap_grd, lap_blk, 0, s.stream>>>(
            s.d_grad_output, s.d_mask, s.d_grad_lap_tmp,
            canvas_h_, canvas_w_, s.grad_lambda);
        CUDA_KERNEL_CHECK();
        CUDA_CHECK(cudaMemcpyAsync(s.d_grad_output, s.d_grad_lap_tmp,
                                   s.total_pixels * 3 * sizeof(float),
                                   cudaMemcpyDeviceToDevice, s.stream));
    }

    // ── Multi-scale pyramid perceptual loss (low weight to avoid overwhelming base MSE) ──
    if (s.perceptual_weight > 0.0f && s.pyramid_levels > 0 && s.total_pixels > 0) {
        int H = canvas_h_, W = canvas_w_;

        // Level 1: downsample 2x → compute MSE + grad → upsample + accumulate
        if (s.pyramid_levels >= 1 && H >= 2 && W >= 2) {
            int h1 = H / 2, w1 = W / 2;
            dim3 ds_blk(16, 16);
            dim3 ds_grd((w1 + 15) / 16, (h1 + 15) / 16);

            // Downsample rendered
            downsample2x_kernel<<<ds_grd, ds_blk, 0, s.stream>>>(
                s.d_output, s.d_pyr_rendered_1, H, W);
            // Downsample target
            downsample2x_kernel<<<ds_grd, ds_blk, 0, s.stream>>>(
                d_target_eff, s.d_pyr_target_1, H, W);

            // Compute MSE + grad at level 1
            int pyr_blocks = (h1 * w1 + 255) / 256;
            pyramid_mse_grad_kernel<<<pyr_blocks, 256, 0, s.stream>>>(
                s.d_pyr_rendered_1, s.d_pyr_target_1,
                s.d_pyr_grad_tmp, s.d_mse_sum,
                h1, w1, s.total_pixels, s.perceptual_weight * 0.4f);
            CUDA_KERNEL_CHECK();

            // Upsample grad from level 1 → accumulate into d_grad_output
            dim3 us_blk(16, 16);
            dim3 us_grd((W + 15) / 16, (H + 15) / 16);
            upsample_add_grad_kernel<<<us_grd, us_blk, 0, s.stream>>>(
                s.d_pyr_grad_tmp, s.d_grad_output, h1, w1, W);
        }

        // Level 2: downsample 4x → compute MSE + grad
        if (s.pyramid_levels >= 2 && H >= 4 && W >= 4) {
            int h2 = H / 4, w2 = W / 4;
            dim3 ds2_blk(16, 16);
            dim3 ds2_grd((w2 + 15) / 16, (h2 + 15) / 16);

            // Downsample from level 1 to level 2
            downsample2x_kernel<<<ds2_grd, ds2_blk, 0, s.stream>>>(
                s.d_pyr_rendered_1, s.d_pyr_rendered_2, H/2, W/2);
            downsample2x_kernel<<<ds2_grd, ds2_blk, 0, s.stream>>>(
                s.d_pyr_target_1, s.d_pyr_target_2, H/2, W/2);

            // Compute MSE + grad at level 2
            int pyr2_blocks = (h2 * w2 + 255) / 256;
            pyramid_mse_grad_kernel<<<pyr2_blocks, 256, 0, s.stream>>>(
                s.d_pyr_rendered_2, s.d_pyr_target_2,
                s.d_pyr_grad_tmp, s.d_mse_sum,
                h2, w2, s.total_pixels, s.perceptual_weight * 0.2f);
            CUDA_KERNEL_CHECK();

            // Upsample level 2 grad to level 1 space.
            // CRITICAL: Zero d_pyr_rendered_1 first — it still contains the
            // downsampled rendered image (values 0-255), which would completely
            // overwhelm the actual gradient signal if used as-is.
            int h1 = H / 2, w1 = W / 2;
            int h1_pixels = h1 * w1;
            CUDA_CHECK(cudaMemsetAsync(s.d_pyr_rendered_1, 0,
                                       h1_pixels * 3 * sizeof(float), s.stream));
            dim3 us2_blk(16, 16);
            dim3 us2_grd(((W/2) + 15) / 16, ((H/2) + 15) / 16);
            upsample_add_grad_kernel<<<us2_grd, us2_blk, 0, s.stream>>>(
                s.d_pyr_grad_tmp, s.d_pyr_rendered_1, h2, w2, W/2);

            // Now upsample the accumulated grad in pyr_rendered_1 (level 1 → full res)
            dim3 us1_blk(16, 16);
            dim3 us1_grd((W + 15) / 16, (H + 15) / 16);
            upsample_add_grad_kernel<<<us1_grd, us1_blk, 0, s.stream>>>(
                s.d_pyr_rendered_1, s.d_grad_output, H/2, W/2, W);
        }
    }

    // Async DMA readback of MSE value (pinned memory + event, lag 1 iter to avoid CPU sync)
    // 返回上次 iter 的 MSE，同时启动本次 MSE 的异步拷贝
    float current_mse;
    float norm = s.total_pixels * (float)channels;
    if (mse_first_iter_) {
        // 首次迭代：同步等待，确保有可用值
        CUDA_CHECK(cudaMemcpyAsync(h_mse_pinned_, s.d_mse_sum, sizeof(float),
                                   cudaMemcpyDeviceToHost, s.stream));
        CUDA_CHECK(cudaEventRecord(mse_event_, s.stream));
        CUDA_CHECK(cudaEventSynchronize(mse_event_));
        current_mse = *h_mse_pinned_ / norm;
        last_mse_ = current_mse;
        mse_first_iter_ = false;
    } else {
        // 非首次：先检查上次的 event 是否完成（应该已完成，过了整个 backward+adam 周期）
        if (cudaEventQuery(mse_event_) == cudaSuccess) {
            last_mse_ = *h_mse_pinned_ / norm;
        }
        // 启动本次的异步拷贝（覆盖 pinned buffer 和 event）
        CUDA_CHECK(cudaMemcpyAsync(h_mse_pinned_, s.d_mse_sum, sizeof(float),
                                   cudaMemcpyDeviceToHost, s.stream));
        CUDA_CHECK(cudaEventRecord(mse_event_, s.stream));
        // 返回上次的 MSE（lag 1 iter）
        current_mse = last_mse_;
    }

    return current_mse;
}

void DiffRenderer::compute_mse_and_grad_rev(int N) {
    (void)N;
    auto& s = scratch_;

    // 反向 MSE + grad（简化版：仅基本 MSE，不做 perceptual pyramid）
    // channels 与正向一致：alpha_loss_enabled 时为 4（含 alpha 通道），否则为 3
    int channels = s.alpha_loss_enabled ? 4 : 3;

    CUDA_CHECK(cudaMemsetAsync(s.d_mse_sum_rev, 0, sizeof(float), s.stream));

    int threads = 256;
    int blocks = (s.total_pixels + threads - 1) / threads;
    mse_grad_kernel<<<blocks, threads, 0, s.stream>>>(
        s.d_output_rev, s.d_target, s.d_mask,
        s.d_grad_output_rev, s.d_mse_sum_rev, s.total_pixels, channels,
        s.d_rendered_alpha_rev, s.d_target_alpha, s.d_grad_alpha_rev);
    CUDA_KERNEL_CHECK();

    // 如果 alpha loss 未启用，清零反向 alpha 梯度（与正向行为一致）
    if (!s.alpha_loss_enabled) {
        CUDA_CHECK(cudaMemsetAsync(s.d_grad_alpha_rev, 0, s.total_pixels * sizeof(float), s.stream));
    }
}

void DiffRenderer::build_batch_descriptors(int N) {
    (void)N;
    auto& s = scratch_;
    batch_descs_.clear();

    size_t free_bytes, total_bytes;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        free_bytes = 512ULL * 1024 * 1024;
    }
    size_t budget = static_cast<size_t>(free_bytes * 0.80);
    const size_t overhead = 512ULL * 1024 * 1024;
    size_t scratch_budget = (budget > overhead) ? (budget - overhead) : 0;
    if (scratch_budget < 1024 * 1024) scratch_budget = 1024 * 1024;

    struct TileInfo { int idx, pixels, shapes; };
    std::vector<TileInfo> tiles(s.num_tiles);
    for (int i = 0; i < s.num_tiles; i++) {
        int tx = i % s.num_tiles_x, ty = i / s.num_tiles_x;
        int tw = min(s.tile_size, canvas_w_ - tx * s.tile_size);
        int th = min(s.tile_size, canvas_h_ - ty * s.tile_size);
        tiles[i] = {i, tw * th, h_tile_counts_[i]};
    }

    int tile_ptr = 0;
    while (tile_ptr < s.num_tiles) {
        TileBatchDesc batch;
        batch.tile_start = tile_ptr;
        batch.total_pixels = 0;
        batch.max_shapes = 0;

        int shape_cap = std::max(1, tiles[tile_ptr].shapes);
        batch.max_shapes = shape_cap;
        size_t max_pixels_for_cap = scratch_budget / (shape_cap * sizeof(__half));

        while (tile_ptr < s.num_tiles) {
            int candidate_shapes = tiles[tile_ptr].shapes;
            if (candidate_shapes > shape_cap) {
                size_t new_need = (size_t)candidate_shapes *
                    (batch.total_pixels + tiles[tile_ptr].pixels) * sizeof(__half);
                if (new_need <= scratch_budget) {
                    shape_cap = candidate_shapes;
                    batch.max_shapes = candidate_shapes;
                    max_pixels_for_cap = scratch_budget / (shape_cap * sizeof(__half));
                } else {
                    break;
                }
            }
            if ((size_t)(batch.total_pixels + tiles[tile_ptr].pixels) > max_pixels_for_cap) {
                if (batch.total_pixels == 0) {
                    shape_cap = tiles[tile_ptr].shapes;
                    batch.max_shapes = shape_cap;
                } else {
                    break;
                }
            }
            batch.total_pixels += tiles[tile_ptr].pixels;
            batch.max_shapes = shape_cap;
            tile_ptr++;
        }

        batch.tile_count = tile_ptr - batch.tile_start;
        batch_descs_.push_back(batch);
    }

    VIN_INFO("Tile-batch: %zu batches, scratch_budget=%zu MB",
             batch_descs_.size(), scratch_budget / (1024 * 1024));
}

float DiffRenderer::render_forward_backward(int N, float* d_grad_history, int write_idx, float temperature, bool output_quadrant) {
    if (!initialized_ || N == 0) return 0.0f;

    auto& s = scratch_;

    cuda_compute_sin_cos(s.d_angle, s.d_sina, s.d_cosa, N, s.stream);
    cuda_compute_softmax(s.d_type_logits, s.d_softmax_w, N, temperature, s.stream, s.num_template_types);
    // DARTS mode (render_mode != 0): preserve DARTS_MODE sentinel (-1) on shapes
    // that haven't been explicitly locked. PURE mode (render_mode == 0): force-lock all.
    bool force_lock = (s.render_mode == 0);
    cuda_update_locked_types(s.d_softmax_w, s.d_shape_types, N, s.stream, s.num_template_types, force_lock);

    CUDA_CHECK(cudaMemsetAsync(s.d_grad_unpacked, 0, (size_t)N * 9 * sizeof(float), s.stream));
    CUDA_CHECK(cudaMemsetAsync(s.d_grad_logits, 0, N * s.num_template_types * sizeof(float), s.stream));
    CUDA_CHECK(cudaMemsetAsync(s.d_rendered_alpha, 0, s.total_pixels * sizeof(float), s.stream));
    if (s.soft_depth_enabled && s.d_grad_z) {
        CUDA_CHECK(cudaMemsetAsync(s.d_grad_z, 0, N * sizeof(float), s.stream));
    }
    if (output_quadrant && s.d_grad_rgb_quadrant) {
        CUDA_CHECK(cudaMemsetAsync(s.d_grad_rgb_quadrant, 0,
            (size_t)N * 4 * 3 * sizeof(float), s.stream));
    }

    // R23: 确定性归约 — 清零 partials（grad_partials 直接写但未必覆盖所有 tile-shape 对；
    // grad_logits_partials 用 atomicAdd 累加）。两者都必须从 0 开始。
    if (deterministic_reduce_) {
        CUDA_CHECK(cudaMemsetAsync(s.d_grad_partials, 0,
            (size_t)s.max_blocks_grad * max_shapes_ * 9 * sizeof(float), s.stream));
        CUDA_CHECK(cudaMemsetAsync(s.d_grad_logits_partials, 0,
            (size_t)s.max_blocks_grad * max_shapes_ * s.num_template_types * sizeof(float), s.stream));
    }

#ifdef SHARP_FORWARD
    // === 无模糊改造：强制全程锐利（sigma=0 等效）===
    // 把原始锐利模板 d_templates 直接复制到 d_blurred_templates，
    // 使前向渲染与反向传播都使用锐利模板（前后向共享同一路径，不可分离）。
    // sigma annealing 仍由 optimizer 计算，但此处对其结果无视。
    if (last_blur_sigma_ > -0.5f) {  // 仅首帧触发一次复制（之后模板内容不变）
        CUDA_CHECK(cudaMemcpyAsync(s.d_blurred_templates, s.d_templates,
            s.num_template_types * TEMPLATE_SIZE * TEMPLATE_SIZE * sizeof(float),
            cudaMemcpyDeviceToDevice, s.stream));
        last_blur_sigma_ = -1.0f;  // 标记已切换为锐利模式，后续不再重复复制
    }
#elif defined(ASYMM_SHARP_FWD)
    // === 不对称渲染：前向锐利 + 反向模糊 ===
    // 反向仍用 d_blurred_templates（保持 sigma annealing 模糊），
    // 前向用 d_templates（锐利）。此处保持 d_blurred_templates 的模糊更新不变。
    if (fabsf(blur_sigma_ - last_blur_sigma_) > 0.01f) {
        cuda_template_blur(s.d_templates, s.d_blurred_templates, s.d_template_tmp,
                           blur_sigma_, s.stream, s.d_blur_kernel, s.num_template_types);
        last_blur_sigma_ = blur_sigma_;
    }
#else
    if (fabsf(blur_sigma_ - last_blur_sigma_) > 0.01f) {
        cuda_template_blur(s.d_templates, s.d_blurred_templates, s.d_template_tmp,
                           blur_sigma_, s.stream, s.d_blur_kernel, s.num_template_types);
        last_blur_sigma_ = blur_sigma_;
    }
#endif

    // === 双模糊退火：对称模糊路径下，目标图按同一 sigma(t) 模糊 ===
    // 训练损失对模糊目标计算 MSE；原始 d_target 不动，诊断/PSNR 仍用原图。
    if (target_blur_enabled_ && s.d_target_blurred) {
        if (fabsf(blur_sigma_ - last_target_blur_sigma_) > 0.01f) {
            cuda_blur_image(s.d_target, s.d_target_blurred, s.d_target_blur_tmp,
                            canvas_w_, canvas_h_, 3, blur_sigma_, s.stream, s.d_blur_kernel);
            last_target_blur_sigma_ = blur_sigma_;
        }
    }

    // === Phase 0: Culling + Batch划分 ===
    cuda_cull_shapes_to_tiles(
        s.d_ste_cx, s.d_ste_cy, s.d_ste_rx, s.d_ste_ry,
        s.d_sina, s.d_cosa, s.d_shape_types,
        N, canvas_w_, canvas_h_,
        s.tile_size, s.num_tiles_x, s.num_tiles_y, s.max_shapes_per_tile,
        s.d_tile_shape_lists, s.d_tile_shape_counts, s.d_tile_overflow,
        h_tile_counts_.data(),
        blur_sigma_, s.stream, s.num_template_types, s.num_builtins,
        /*reverse=*/false);

#ifndef __NODEBUG__
    {
        static int cull_dbg = 0;
        if (++cull_dbg % 30 == 0) {
            int tl = s.num_tiles - 1;
            int tr = s.num_tiles - s.num_tiles_x;
            int bl = s.num_tiles_x - 1;
            long long sum_all = 0;
            for (int i = 0; i < s.num_tiles; i++) sum_all += h_tile_counts_[i];
            VIN_INFO("  [cull-dbg] iter %d tile_counts: all_sum=%lld topL=%d topR=%d botL=%d botR=%d avg=%.1f",
                     cull_dbg, sum_all,
                     h_tile_counts_[0], h_tile_counts_[tr],
                     h_tile_counts_[bl], h_tile_counts_[tl],
                     (double)sum_all / s.num_tiles);
        }
    }
#endif

    build_batch_descriptors(N);

    size_t max_scratch = 0;
        for (auto& desc : batch_descs_) {
        size_t need = (size_t)desc.total_pixels * desc.max_shapes * sizeof(__half);
        if (need > max_scratch) max_scratch = need;
    }
    if (max_scratch > s.batch_scratch_capacity) {
        if (s.d_batch_scratch) cudaFree(s.d_batch_scratch);
        CUDA_CHECK(cudaMalloc(&s.d_batch_scratch, max_scratch));
        s.batch_scratch_capacity = max_scratch;
    }

    // === Phase 1: Forward ===
    CUDA_CHECK(cudaMemsetAsync(s.d_output, 0, s.total_pixels * 3 * sizeof(float), s.stream));
    CUDA_CHECK(cudaMemsetAsync(s.d_rendered_alpha, 0, s.total_pixels * sizeof(float), s.stream));

    // Soft depth 参数：启用时传 d_ste_z / current_tau_z，否则传 nullptr/0
    const float* sd_dz = s.soft_depth_enabled ? s.d_ste_z : nullptr;
    float sd_tau_z = s.soft_depth_enabled ? s.current_tau_z : 0.0f;
    int sd_max_m = s.soft_depth_enabled ? s.soft_depth_max_m : 0;
    float* sd_grad_z = s.soft_depth_enabled ? s.d_grad_z : nullptr;

    // ASYMM_SHARP_FWD: 前向用锐利模板 d_templates，反向保持模糊 d_blurred_templates
    const float* d_fwd_templates =
#ifdef ASYMM_SHARP_FWD
        s.d_templates;
#else
        s.d_blurred_templates;
#endif

    // === 运行时探针：一次性验证 ASYMM_SHARP_FWD 生效（前向锐利/反向模糊分离）===
    // 已验证链路正确（fwd_is_sharp=1, fwd_eq_blur=0），探针停用。
    /*
    {
        static int probe_cnt = 0;
        if (probe_cnt < 3 && (probe_cnt == 0 || probe_cnt == 1 || probe_cnt == 2)) {
            probe_cnt++;
            float t0_sharp = 0.0f, t0_blur = 0.0f, t0_fwd = 0.0f;
            // 拷贝首元素到 host 比较（d_templates[0], d_blurred_templates[0], d_fwd_templates[0]）
            cudaMemcpy(&t0_sharp, s.d_templates, sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&t0_blur,  s.d_blurred_templates, sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&t0_fwd,   d_fwd_templates,       sizeof(float), cudaMemcpyDeviceToHost);
            VIN_INFO("[ASYMM_PROBE #%d] iter_ptr=%p | d_templates=%p val[0]=%.6f | "
                     "d_blurred_templates=%p val[0]=%.6f | d_fwd_templates=%p val[0]=%.6f | "
                     "fwd_is_sharp=%d | fwd_eq_blur=%d | blur_sigma=%.3f",
                     probe_cnt, (void*)d_fwd_templates,
                     (void*)s.d_templates, t0_sharp,
                     (void*)s.d_blurred_templates, t0_blur,
                     (void*)d_fwd_templates, t0_fwd,
                     (d_fwd_templates == s.d_templates) ? 1 : 0,
                     (d_fwd_templates == s.d_blurred_templates) ? 1 : 0,
                     blur_sigma_);
        }
    }
    */

    for (auto& desc : batch_descs_) {
        cuda_fused_tile_forward(
            s.d_px, s.d_py, s.d_ste_cx, s.d_ste_cy, s.d_ste_rx, s.d_ste_ry,
            s.d_sina, s.d_cosa, s.d_shape_types, s.d_ste_colors, s.d_ste_opacity,
            s.d_bg_color, d_fwd_templates,
            s.d_output, s.d_rendered_alpha,
            /*d_sp_S=*/(s.sp_lambda > 0.0f || s.sp_geo_active) ? s.d_sp_S : nullptr,
            /*d_sp_G=*/s.sp_geo_loss_active ? s.d_sp_G : nullptr,
            s.d_batch_scratch,
            s.d_tile_shape_lists, s.d_tile_shape_counts,
            s.num_tiles_x, s.num_tiles_y, s.tile_size, s.max_shapes_per_tile,
            desc.tile_start, desc.tile_count, desc.total_pixels, desc.max_shapes,
            N, canvas_w_, canvas_h_, s.total_pixels,
            s.d_softmax_w, s.render_mode, blur_sigma_,
            s.stream, s.num_template_types, s.num_builtins,
            /*reverse=*/false,
            sd_dz, sd_tau_z, sd_max_m);
        CUDA_KERNEL_CHECK();
    }

#ifndef __NODEBUG__
    {
        static int fwd_dbg = 0;
        if (++fwd_dbg % 15 == 0) {
            CUDA_CHECK(cudaStreamSynchronize(s.stream));
            float buf[30];
            int steps[10] = {0, s.total_pixels/10, s.total_pixels/5, s.total_pixels/3, s.total_pixels/2,
                             2*s.total_pixels/3, 3*s.total_pixels/4, s.total_pixels-100, s.total_pixels-10, s.total_pixels-1};
            for (int j = 0; j < 10; j++)
                cudaMemcpy(buf + j*3, s.d_output + steps[j]*3, 3*sizeof(float), cudaMemcpyDeviceToHost);
            int nz = 0;
            for (int j = 0; j < 30; j++) if (buf[j] > 0.1f || buf[j] < -0.1f) nz++;
            float avg_nonzero = 0; int nzc = 0;
            for (int j = 0; j < 30; j++) if (fabsf(buf[j]) > 0.1f) { avg_nonzero += buf[j]; nzc++; }
            if (nzc > 0) avg_nonzero /= nzc;

            int tile_sample[3];
            cudaMemcpy(tile_sample, s.d_tile_shape_counts, 3*sizeof(int), cudaMemcpyDeviceToHost);
            int mid_tile = (s.num_tiles/2) * s.num_tiles_x + s.num_tiles_x/2;
            int mc;
            cudaMemcpy(&mc, s.d_tile_shape_counts + mid_tile, sizeof(int), cudaMemcpyDeviceToHost);
            VIN_INFO("  [fwd2-dbg] iter %d nz=%d/30 avg=%.2f tile0=%d mid_t=%d last_t=%d mc=%d",
                     fwd_dbg, nz, avg_nonzero, tile_sample[0], mid_tile, s.num_tiles-1, mc);
        }
    }
#endif

    // MSE + grad_output (all pixels, after Phase 1)
    float mse = compute_mse_and_grad(N);

    // If alpha loss is not enabled, zero out the alpha gradient
    // to prevent it from pulling opacity toward 0 (align with diffbmp default behavior)
    if (!s.alpha_loss_enabled) {
        CUDA_CHECK(cudaMemsetAsync(s.d_grad_alpha, 0, s.total_pixels * sizeof(float), s.stream));
    }

    // Aux-Vis 自检模式：清零 MSE 像素梯度，使 backward 中只有 aux-vis 注入的梯度。
    // MSE 仍正常计算（返回值供观测），仅清零其对参数的梯度。机制纯净验证用。
    if (s.aux_vis_only) {
        CUDA_CHECK(cudaMemsetAsync(s.d_grad_output, 0, s.total_pixels * 3 * sizeof(float), s.stream));
        CUDA_CHECK(cudaMemsetAsync(s.d_grad_alpha, 0, s.total_pixels * sizeof(float), s.stream));
    }

    // EM-YMP: 启用误差调制时，每步基于当前 forward 结果重算误差图 + mean_err，
    // 传入 backward kernel 调制 vis_below_acc（高误差区放大让位，低误差区衰减）。
    // yield_lambda=0 或 yield_em_modulate=false 时传 nullptr，kernel 内 err_norm=1 退化为原 YMP。
    const float* em_error_map = nullptr;
    const float* em_mean_err = nullptr;
    if (s.yield_lambda > 0.0f && s.yield_em_modulate) {
        compute_error_map_and_mean();
        em_error_map = s.d_error_map_scratch;
        em_mean_err = s.d_mean_err;
    }

    // FA-Loss: 启用时在 forward 后、backward 前计算每形状 A_i/μ_i/σ²_i 与全局
    // fa_scale/fa_mean_t（写入 d_fa_globals），供 tile_backward_kernel 注入 dL_fa/d_ar。
    // 全部在同 stream 上，无 D2H sync（fa_scale/fa_mean_t 作为 device 指针传入 kernel）。
    const float* fa_globals_ptr = nullptr;
    const float* fa_mu_ptr = nullptr;
    if (s.fa_lambda > 0.0f) {
        compute_fa_loss_stats(N);
        fa_globals_ptr = s.d_fa_globals;
        fa_mu_ptr = s.d_fa_mu;
    }

    // === Aux-Vis gate: 启用时（aux_vis_lambda>0）在 forward 后、backward 前重算 vis + g_i ===
    // vis 用本 iter forward 的 STE params（compute_visibility 读取 d_ste_*），与 backward 一致。
    // g_i = 1 - sigmoid((vis_i-thresh)/temp) 写入 d_aux_gate，供 backward kernel 注入 ∂L_aux/∂ae。
    // 纯 GPU kernel 链（memset+visibility+gate），无 D2H sync，开销约等于一个 forward pass。
    const float* aux_gate_ptr = nullptr;
    if (s.aux_vis_lambda > 0.0f) {
        compute_visibility(N);
        compute_aux_vis_gate(N);
        aux_gate_ptr = s.d_aux_gate;
    }

    // === Phase 2: Backward ===
    for (auto& desc : batch_descs_) {
        // P3-5: 移除 mask 构建（host 端 sync 循环已消除），直接用 tile_shape_lists
        // 传入 1/T 给 backward kernel，用于补偿 softmax 温度对 grad_logits 的影响
        float inv_temperature = (temperature > 1e-6f) ? (1.0f / temperature) : 1.0f;
        cuda_tile_backward(
            s.d_grad_output, s.d_output, s.d_px, s.d_py,
            s.d_ste_cx, s.d_ste_cy, s.d_ste_rx, s.d_ste_ry, s.d_ste_angle,
            s.d_ste_colors, s.d_ste_opacity, s.d_bg_color,
            s.d_sina, s.d_cosa, s.d_blurred_templates,
            s.d_grad_cx, s.d_grad_cy, s.d_grad_rx, s.d_grad_ry,
            s.d_grad_angle, s.d_grad_colors, s.d_grad_opacity,
            s.d_tile_shape_lists, s.d_tile_shape_counts,
            s.num_tiles_x, s.num_tiles_y, s.tile_size, s.max_shapes_per_tile,
            desc.tile_start, desc.tile_count,
            canvas_w_, canvas_h_,
            s.d_grad_alpha, s.d_softmax_w, s.d_grad_logits,
            s.d_shape_types, s.render_mode, blur_sigma_,
            s.stream,
            output_quadrant, s.d_grad_rgb_quadrant, s.num_template_types, s.num_builtins,
            inv_temperature,
            /*reverse=*/false,
            sd_dz, sd_tau_z, sd_max_m, sd_grad_z,
            N,
            /*yield_lambda=*/s.yield_lambda,  // 正向 backward 启用 yield 梯度
            /*error_map=*/em_error_map,        // EM-YMP 误差图（nullptr=YMP）
            /*d_mean_err=*/em_mean_err,       // EM-YMP 全局平均误差
            /*d_fa_globals=*/fa_globals_ptr,  // FA-Loss 全局 {fa_scale, fa_mean_t}（nullptr=禁用）
            /*d_fa_mu=*/fa_mu_ptr,            // FA-Loss 每形状 μ_i
            /*d_target=*/s.d_target,           // FA-Loss 目标图像
            /*d_mask=*/s.d_mask,              // FA-Loss mask
            /*aux_vis_lambda=*/s.aux_vis_lambda,  // Aux-Vis 权重（0=禁用）
            /*d_aux_gate=*/aux_gate_ptr,          // Aux-Vis g_i（nullptr=禁用）
            /*sp_lambda=*/s.sp_lambda,            // SP-Loss 权重（0=禁用）
            /*d_k_allowed=*/(s.sp_lambda > 0.0f) ? s.d_k_allowed : nullptr,
            /*sp_geo_loss_active=*/s.sp_geo_loss_active,
            /*deterministic=*/deterministic_reduce_ != 0,
            /*grad_partials=*/s.d_grad_partials,
            /*grad_logits_partials=*/s.d_grad_logits_partials);
        CUDA_KERNEL_CHECK();
    }

    // === Phase 2b: 反向渲染（双向渲染，bidir_lambda_ > 0 时执行）===
    // 反向渲染复用正向的 tile list / batch descriptors / batch_scratch，
    // 仅在 forward/backward kernel 内部反转遍历顺序（reverse=true）。
    // 这样省去重复 culling（含 CPU sync）和 batch 重建，性能开销减半。
    if (bidir_lambda_ > 0.0f && s.d_output_rev != nullptr) {
        // 反向 Forward：清零反向输出 buffer，复用正向 batch_descs_
        CUDA_CHECK(cudaMemsetAsync(s.d_output_rev, 0, s.total_pixels * 3 * sizeof(float), s.stream));
        CUDA_CHECK(cudaMemsetAsync(s.d_rendered_alpha_rev, 0, s.total_pixels * sizeof(float), s.stream));

        for (auto& desc : batch_descs_) {
            cuda_fused_tile_forward(
                s.d_px, s.d_py, s.d_ste_cx, s.d_ste_cy, s.d_ste_rx, s.d_ste_ry,
                s.d_sina, s.d_cosa, s.d_shape_types, s.d_ste_colors, s.d_ste_opacity,
                s.d_bg_color, d_fwd_templates,
                s.d_output_rev, s.d_rendered_alpha_rev,
                /*d_sp_S=*/nullptr,  // 反向 bidir 不计算 SP-Loss S(p)
                /*d_sp_G=*/nullptr,  // 反向 bidir 不计算 SP-GEO-Loss G(p)
                s.d_batch_scratch,
                s.d_tile_shape_lists, s.d_tile_shape_counts,
                s.num_tiles_x, s.num_tiles_y, s.tile_size, s.max_shapes_per_tile,
                desc.tile_start, desc.tile_count, desc.total_pixels, desc.max_shapes,
                N, canvas_w_, canvas_h_, s.total_pixels,
                s.d_softmax_w, s.render_mode, blur_sigma_,
                s.stream, s.num_template_types, s.num_builtins,
                /*reverse=*/true,
                /*d_z=*/nullptr, /*tau_z=*/0.0f, /*soft_depth_max_m=*/0);
            CUDA_KERNEL_CHECK();
        }

        // 反向 MSE + grad（写入 d_grad_output_rev / d_grad_alpha_rev / d_mse_sum_rev）
        compute_mse_and_grad_rev(N);

        // 缩放反向梯度：d_grad_output_rev *= λ, d_grad_alpha_rev *= λ
        {
            int threads = 256;
            int blocks_rgb = (s.total_pixels * 3 + threads - 1) / threads;
            scale_grads_kernel_v2<<<blocks_rgb, threads, 0, s.stream>>>(
                s.d_grad_output_rev, s.total_pixels * 3, bidir_lambda_);
            CUDA_KERNEL_CHECK();

            int blocks_a = (s.total_pixels + threads - 1) / threads;
            scale_grads_kernel_v2<<<blocks_a, threads, 0, s.stream>>>(
                s.d_grad_alpha_rev, s.total_pixels, bidir_lambda_);
            CUDA_KERNEL_CHECK();
        }

        // 反向 Backward：复用正向 batch_descs_，梯度通过 atomicAdd 累加到现有 grad buffer
        for (auto& desc : batch_descs_) {
            float inv_temperature = (temperature > 1e-6f) ? (1.0f / temperature) : 1.0f;
            cuda_tile_backward(
                s.d_grad_output_rev, s.d_output, s.d_px, s.d_py,
                s.d_ste_cx, s.d_ste_cy, s.d_ste_rx, s.d_ste_ry, s.d_ste_angle,
                s.d_ste_colors, s.d_ste_opacity, s.d_bg_color,
                s.d_sina, s.d_cosa, s.d_blurred_templates,
                s.d_grad_cx, s.d_grad_cy, s.d_grad_rx, s.d_grad_ry,
                s.d_grad_angle, s.d_grad_colors, s.d_grad_opacity,
                s.d_tile_shape_lists, s.d_tile_shape_counts,
                s.num_tiles_x, s.num_tiles_y, s.tile_size, s.max_shapes_per_tile,
                desc.tile_start, desc.tile_count,
                canvas_w_, canvas_h_,
                s.d_grad_alpha_rev, s.d_softmax_w, s.d_grad_logits,
                s.d_shape_types, s.render_mode, blur_sigma_,
                s.stream,
                output_quadrant, s.d_grad_rgb_quadrant, s.num_template_types, s.num_builtins,
                inv_temperature,
                /*reverse=*/true,
                /*d_z=*/nullptr, /*tau_z=*/0.0f, /*soft_depth_max_m=*/0,
                /*grad_z=*/nullptr,
                N,
                /*yield_lambda=*/0.0f,  // 反向 bidir 不重复算 yield（遮挡语义不同）
                /*error_map=*/nullptr,  // 反向不使用 EM-YMP 调制
                /*d_mean_err=*/nullptr,
                /*d_fa_globals=*/nullptr,  // 反向 bidir 不注入 FA-Loss 梯度
                /*d_fa_mu=*/nullptr,
                /*d_target=*/nullptr,
                /*d_mask=*/nullptr,
                /*aux_vis_lambda=*/0.0f,      // 反向 bidir 不注入 Aux-Vis（遮挡语义不同）
                /*d_aux_gate=*/nullptr,
                /*sp_lambda=*/0.0f,            // 反向 bidir 不注入 SP-Loss（遮挡语义不同）
                /*d_k_allowed=*/nullptr,
                /*sp_geo_loss_active=*/false,
                /*deterministic=*/deterministic_reduce_ != 0,
                /*grad_partials=*/s.d_grad_partials,
                /*grad_logits_partials=*/s.d_grad_logits_partials);
            CUDA_KERNEL_CHECK();
        }
    }

    // R23: 确定性归约 — 将 per-tile partials 固定顺序归约到全局梯度缓冲。
    // 在所有 backward（正向+双向）完成后、pack 前执行。
    // d_grad_cx/cy/.../d_grad_logits 已在 backward 前清零，reduce kernel 用 += 累加。
    if (deterministic_reduce_) {
        int threads = 256;
        int blocks_red = (N + threads - 1) / threads;
        reduce_grad_partials_kernel<<<blocks_red, threads, 0, s.stream>>>(
            s.d_grad_partials,
            s.d_grad_cx, s.d_grad_cy, s.d_grad_rx, s.d_grad_ry,
            s.d_grad_angle, s.d_grad_colors, s.d_grad_opacity,
            N, s.num_tiles);
        CUDA_KERNEL_CHECK();
        int total_logits = N * s.num_template_types;
        int blocks_logits = (total_logits + threads - 1) / threads;
        reduce_logits_partials_kernel<<<blocks_logits, threads, 0, s.stream>>>(
            s.d_grad_logits_partials, s.d_grad_logits,
            N, s.num_template_types, s.num_tiles);
        CUDA_KERNEL_CHECK();
    }

#ifndef __NODEBUG__
    {
        static int pre_pack_dbg = 0;
        if (++pre_pack_dbg % 10 == 0) {
            CUDA_CHECK(cudaStreamSynchronize(s.stream));
            float ug[9], lg[16];
            cudaMemcpy(ug, s.d_grad_unpacked, 9 * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(lg, s.d_grad_logits, s.num_template_types * sizeof(float), cudaMemcpyDeviceToHost);
            int u_bad = 0, l_bad = 0;
            for (int j = 0; j < 9; j++) if (!isfinite(ug[j]) || fabsf(ug[j]) > 1e10f) u_bad++;
            for (int j = 0; j < s.num_template_types; j++) if (!isfinite(lg[j]) || fabsf(lg[j]) > 1e10f) l_bad++;
            VIN_INFO("  [pre-pk] iter %d grad_unpacked[0..8]=%.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f  bad=%d/9",
                     pre_pack_dbg, ug[0], ug[1], ug[2], ug[3], ug[4], ug[5], ug[6], ug[7], ug[8], u_bad);
            VIN_INFO("  [pre-pk] iter %d grad_logits[0..3]=%.4f %.4f %.4f %.4f  bad=%d/4",
                     pre_pack_dbg, lg[0], lg[1], lg[2], lg[3], l_bad);
        }
    }
#endif

#ifndef __NODEBUG__
    static int rfb_call_count = 0;
    rfb_call_count++;
    if (rfb_call_count % 50 == 0 && h_debug_grads_pinned_[0] != nullptr) {
        int write_buf = debug_buf_idx_;
        int read_buf = 1 - write_buf;

        CUDA_CHECK(cudaMemcpyAsync(h_debug_grads_pinned_[write_buf], s.d_grads_packed,
                                   N * (9 + s.num_template_types) * sizeof(float), cudaMemcpyDeviceToHost, s.stream));
        CUDA_CHECK(cudaEventRecord(debug_events_[write_buf], s.stream));

        if (!debug_first_[read_buf] &&
            cudaEventQuery(debug_events_[read_buf]) == cudaSuccess) {
            float gx_sum = 0, gcx_sum = 0;
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < 9; ++j)
                    gx_sum += fabsf(h_debug_grads_pinned_[read_buf][i*(9+s.num_template_types) + j]);
                for (int j = 9; j < 9 + s.num_template_types; ++j)
                    gcx_sum += fabsf(h_debug_grads_pinned_[read_buf][i*(9+s.num_template_types) + j]);
            }
            VIN_INFO("  [v7] grad_mag: shape=%.2e logits=%.2e ratio=%.1f%%",
                     gx_sum, gcx_sum, 100.0f * gcx_sum / (gx_sum + gcx_sum + 1e-10f));
        }

        debug_first_[write_buf] = false;
        debug_buf_idx_ = read_buf;
    }
#endif

    // === 熵惩罚（opacity 二值化）：L_bin = α_sparsity · Σ opi(1-opi) ===
    // 在所有 backward（正向+双向）完成后、pack 前，per-shape 注入 ∂L_bin/∂logit。
    // atomicAdd 与 MSE 梯度叠加，不破坏既有梯度流。
    if (s.alpha_sparsity_weight > 0.0f) {
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        alpha_sparsity_kernel<<<blocks, threads, 0, s.stream>>>(
            s.d_ste_opacity, s.d_grad_opacity, nullptr, N, s.alpha_sparsity_weight);
        CUDA_KERNEL_CHECK();
    }

    // === Phase 3: Pack gradients ===
    {
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        pack_and_norm_grads_kernel<<<blocks, threads, 0, s.stream>>>(
            s.d_grad_cx, s.d_grad_cy, s.d_grad_rx, s.d_grad_ry,
            s.d_grad_angle, s.d_grad_colors, s.d_grad_opacity,
            s.d_grad_logits,
            s.d_grads_packed, N, d_grad_history, write_idx, s.num_template_types);
        CUDA_KERNEL_CHECK();
    }

#ifndef __NODEBUG__
    {
        static int grad_dbg_count = 0;
        if (++grad_dbg_count % 10 == 0) {
            CUDA_CHECK(cudaStreamSynchronize(s.stream));
            int n_sample = std::min(10, N);
            std::vector<float> sample(n_sample * (9 + s.num_template_types));
            CUDA_CHECK(cudaMemcpy(sample.data(), s.d_grads_packed,
                                  sample.size() * sizeof(float), cudaMemcpyDeviceToHost));
            float gx_sum = 0, gl_sum = 0;
            for (int i = 0; i < n_sample; ++i) {
                for (int j = 0; j < 9; ++j) gx_sum += fabsf(sample[i*(9+s.num_template_types) + j]);
                for (int j = 9; j < 9 + s.num_template_types; ++j) gl_sum += fabsf(sample[i*(9+s.num_template_types) + j]);
            }
            VIN_INFO("  [pack-dbg] iter %d POST-PACK grad(%dshapes): shape=%.4f logits=%.4f",
                     grad_dbg_count, n_sample, gx_sum, gl_sum);
        }
    }
#endif

    return mse;
}

void DiffRenderer::async_copy_canvas(uint8_t* h_pinned_canvas, cudaEvent_t& event) {
    if (!initialized_) return;
    auto& s = scratch_;
    int total_elements = s.total_pixels * 3;

    // Step 1: float_to_uint8 在 compute stream（必须，因为读 d_output）
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    float_to_uint8_kernel<<<blocks, threads, 0, s.stream>>>(
        s.d_output, s.d_canvas_u8, total_elements);

    // Step 2: record event 标记 compute stream 完成 float_to_uint8
    CUDA_CHECK(cudaEventRecord(s.compute_done_event, s.stream));

    // Step 3: viz stream 等待 compute_done_event 后执行 D2H copy
    // 这样 D2H copy 不占用 compute stream，下一个 iter 的 kernel 可立即排队
    CUDA_CHECK(cudaStreamWaitEvent(s.viz_stream, s.compute_done_event, 0));
    CUDA_CHECK(cudaMemcpyAsync(h_pinned_canvas, s.d_canvas_u8,
                               total_elements * sizeof(uint8_t),
                               cudaMemcpyDeviceToHost, s.viz_stream));

    // Step 4: record event 在 viz stream（用于 query 完成状态）
    CUDA_CHECK(cudaEventRecord(event, s.viz_stream));
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
    CUDA_CHECK(cudaMemcpyAsync(h_pinned + 9 * N, s.d_type_logits, N * s.num_template_types * sizeof(float), cudaMemcpyDeviceToHost, s.stream));
    CUDA_CHECK(cudaEventRecord(event, s.stream));
}

void DiffRenderer::upload_params_gpu(const ShapeParams* shapes, int N) {
    auto& s = scratch_;
    // Unpack and upload to GPU-resident param buffers.
    // Logit parameterization: convert host linear ranges to logit space before upload.
    for (int i = 0; i < N; ++i) {
        h_cx_[i] = shapes[i].cx;
        h_cy_[i] = shapes[i].cy;
        h_rx_[i] = shapes[i].rx;
        h_ry_[i] = shapes[i].ry;
        h_angle_buf_[i] = shapes[i].angle;
        // Linear [0,255] → logit
        h_colors_buf_[i * 3 + 0] = host_logit(shapes[i].color_r / 255.0f);
        h_colors_buf_[i * 3 + 1] = host_logit(shapes[i].color_g / 255.0f);
        h_colors_buf_[i * 3 + 2] = host_logit(shapes[i].color_b / 255.0f);
        // Linear [0,1] → logit
        h_opacity_buf_[i] = host_logit(shapes[i].opacity);
    }
    CUDA_CHECK(cudaMemcpyAsync(s.d_cx, h_cx_.data(), N * sizeof(float), cudaMemcpyHostToDevice, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(s.d_cy, h_cy_.data(), N * sizeof(float), cudaMemcpyHostToDevice, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(s.d_rx, h_rx_.data(), N * sizeof(float), cudaMemcpyHostToDevice, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(s.d_ry, h_ry_.data(), N * sizeof(float), cudaMemcpyHostToDevice, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(s.d_angle, h_angle_buf_.data(), N * sizeof(float), cudaMemcpyHostToDevice, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(s.d_colors, h_colors_buf_.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(s.d_opacity, h_opacity_buf_.data(), N * sizeof(float), cudaMemcpyHostToDevice, s.stream));

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
    int W, int H,
    float lr_factor_xy, float lr_factor_r, float lr_factor_angle,
    float lr_factor_rgb, float lr_factor_a,
    float logits_lr_factor,
    int update_mask,
    float max_area_ratio,
    bool uniform_scale)
{
    auto& s = scratch_;
    int total_grad_elements = N * (9 + s.num_template_types);

    // Step 1: Global gradient norm clipping (fully GPU-fused, no CPU sync)
    if (max_grad_norm > 0) {
        CUDA_CHECK(cudaMemsetAsync(s.d_grad_norm_sq, 0, sizeof(float), s.stream));
        int threads = 256;
        int blocks = (total_grad_elements + threads - 1) / threads;
        if (deterministic_reduce_) {
            // R23: 确定性 norm_sq 归约 — 写逐元素 g*g 到缓冲，再两遍归约
            clip_grad_norm_det_pass1_kernel<<<blocks, threads, 0, s.stream>>>(
                s.d_grads_packed, s.d_gg_buffer, total_grad_elements);
            CUDA_KERNEL_CHECK();
            int reduce_blocks = (blocks + 255) / 256;
            scalar_block_reduce_kernel<<<blocks, threads, 0, s.stream>>>(
                s.d_gg_buffer, s.d_scalar_partials, total_grad_elements);
            CUDA_KERNEL_CHECK();
            scalar_final_reduce_kernel<<<1, 256, 0, s.stream>>>(
                s.d_scalar_partials, s.d_grad_norm_sq, blocks);
            CUDA_KERNEL_CHECK();
        } else {
            // Kernel 1: accumulate norm_sq via atomicAdd
            clip_grad_norm_kernel_v2<<<blocks, threads, 0, s.stream>>>(
                s.d_grads_packed, total_grad_elements, max_grad_norm, s.d_grad_norm_sq);
            CUDA_KERNEL_CHECK();
        }
        // Kernel 2: compute scale on GPU (1 thread, no CPU sync)
        compute_clip_scale_kernel<<<1, 1, 0, s.stream>>>(
            s.d_grad_norm_sq, d_clip_scale_, max_grad_norm);
        CUDA_KERNEL_CHECK();
        // Kernel 3: apply scale on GPU (read scale from GPU memory)
        scale_grads_auto_kernel<<<blocks, threads, 0, s.stream>>>(
            s.d_grads_packed, total_grad_elements, d_clip_scale_);
        CUDA_KERNEL_CHECK();
    }

    // Step 2: Fused Adam + clamp + STE
    {
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        fused_adam_kernel<<<blocks, threads, 0, s.stream>>>(
            s.d_grads_packed,
            s.d_cx, s.d_cy, s.d_rx, s.d_ry, s.d_angle, s.d_colors, s.d_opacity,
            s.d_type_logits,
            s.d_m_cx, s.d_m_cy, s.d_m_rx, s.d_m_ry, s.d_m_angle,
            s.d_m_cr, s.d_m_cg, s.d_m_cb, s.d_m_ca,
            s.d_v_cx, s.d_v_cy, s.d_v_rx, s.d_v_ry, s.d_v_angle,
            s.d_v_cr, s.d_v_cg, s.d_v_cb, s.d_v_ca,
            s.d_m_logits, s.d_v_logits,
            s.num_template_types,
            s.d_ste_cx, s.d_ste_cy, s.d_ste_rx, s.d_ste_ry,
            s.d_ste_angle, s.d_ste_colors, s.d_ste_opacity,
            s.d_ste_type_logits,
            N, adam_step, lr, beta1, beta2, eps, max_grad_norm,
            W, H, s.d_frozen_mask,
            nullptr, max_area_ratio,  // area0 已随 position loss 移除（面积硬上限仅在该实验中启用）
            lr_factor_xy, lr_factor_r, lr_factor_angle, lr_factor_rgb, lr_factor_a, logits_lr_factor, s.num_template_types,
            update_mask, uniform_scale);
        CUDA_KERNEL_CHECK();
    }

}


// === 数值校验辅助 ===
// param_offset 映射（与 packed 梯度布局一致）：0-4=几何, 9+t=logit_t。
// cx/cy/rx/ry 操作 STE 缓冲（kernel 直接读取 d_ste_*）；
// angle 操作 d_angle 主参数（kernel 读 sina/cosa）；
// logits 操作 d_type_logits 主参数（softmax_w 由 d_type_logits 重算）。
static float* ste_ptr_for_offset(RenderScratch& s, int param_offset) {
    switch (param_offset) {
        case 0: return s.d_ste_cx;
        case 1: return s.d_ste_cy;
        case 2: return s.d_ste_rx;
        case 3: return s.d_ste_ry;
        case 4: return s.d_angle;   // sincos 派生源（kernel 经 sina/cosa 消费）
        case 8: return s.d_ste_opacity;  // opacity logit (STE buffer)
        default: return nullptr;    // logits 单独处理（d_type_logits 二维索引）
    }
}

float DiffRenderer::get_ste_param(int param_offset, int idx) {
    auto& s = scratch_;
    float* base = ste_ptr_for_offset(s, param_offset);
    float h_val = 0.0f;
    if (base) {
        CUDA_CHECK(cudaMemcpy(&h_val, base + idx, sizeof(float), cudaMemcpyDeviceToHost));
    } else if (param_offset >= 9) {
        int t = param_offset - 9;
        CUDA_CHECK(cudaMemcpy(&h_val, s.d_type_logits + idx * s.num_template_types + t,
                              sizeof(float), cudaMemcpyDeviceToHost));
    }
    return h_val;
}

void DiffRenderer::set_ste_param(int param_offset, int idx, float value) {
    auto& s = scratch_;
    float* base = ste_ptr_for_offset(s, param_offset);
    if (base) {
        CUDA_CHECK(cudaMemcpy(base + idx, &value, sizeof(float), cudaMemcpyHostToDevice));
    } else if (param_offset >= 9) {
        int t = param_offset - 9;
        CUDA_CHECK(cudaMemcpy(s.d_type_logits + idx * s.num_template_types + t,
                              &value, sizeof(float), cudaMemcpyHostToDevice));
    }
}


// === Soft depth z methods ===
void DiffRenderer::upload_z(const float* h_z, int N) {
    auto& s = scratch_;
    CUDA_CHECK(cudaMemcpyAsync(s.d_z, h_z, N * sizeof(float), cudaMemcpyHostToDevice, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(s.d_ste_z, s.d_z, N * sizeof(float), cudaMemcpyDeviceToDevice, s.stream));
}

void DiffRenderer::download_z(float* h_z, int N) {
    auto& s = scratch_;
    CUDA_CHECK(cudaMemcpyAsync(h_z, s.d_z, N * sizeof(float), cudaMemcpyDeviceToHost, s.stream));
    CUDA_CHECK(cudaStreamSynchronize(s.stream));
}

void DiffRenderer::reset_z_adam_state(int N) {
    auto& s = scratch_;
    CUDA_CHECK(cudaMemsetAsync(s.d_m_z, 0, N * sizeof(float), s.stream));
    CUDA_CHECK(cudaMemsetAsync(s.d_v_z, 0, N * sizeof(float), s.stream));
}

void DiffRenderer::fused_adam_update_z(int N, int adam_step, float lr, float beta1, float beta2,
                                       float eps, float lr_factor_z) {
    auto& s = scratch_;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    fused_adam_z_kernel<<<blocks, threads, 0, s.stream>>>(
        s.d_z, s.d_grad_z, s.d_m_z, s.d_v_z, s.d_ste_z,
        s.d_frozen_mask, N, adam_step, lr, beta1, beta2, eps, lr_factor_z);
    CUDA_KERNEL_CHECK();
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

    // Logit parameterization: convert logit → host linear ranges
    for (int i = 0; i < N; ++i) {
        shapes[i].cx = h_cx_[i];
        shapes[i].cy = h_cy_[i];
        shapes[i].rx = h_rx_[i];
        shapes[i].ry = h_ry_[i];
        shapes[i].angle = h_angle_buf_[i];
        // Logit → linear [0,255]
        shapes[i].color_r = 255.0f * host_sigmoid(h_colors_buf_[i * 3 + 0]);
        shapes[i].color_g = 255.0f * host_sigmoid(h_colors_buf_[i * 3 + 1]);
        shapes[i].color_b = 255.0f * host_sigmoid(h_colors_buf_[i * 3 + 2]);
        // Logit → linear [0,1]
        shapes[i].opacity = host_sigmoid(h_opacity_buf_[i]);
    }
}

void DiffRenderer::download_params_soa(
    float* h_cx, float* h_cy, float* h_rx, float* h_ry, float* h_angle,
    float* h_colors, float* h_opacity,
    float* h_type_logits, int N)
{
    auto& s = scratch_;
    // Logit parameterization: h_colors and h_opacity are logits (direct copy from GPU).
    // Caller is responsible for any linear conversion if needed.
    // Queue all D2H copies on the same stream, then single sync
    CUDA_CHECK(cudaMemcpyAsync(h_cx, s.d_cx, N * sizeof(float), cudaMemcpyDeviceToHost, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(h_cy, s.d_cy, N * sizeof(float), cudaMemcpyDeviceToHost, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(h_rx, s.d_rx, N * sizeof(float), cudaMemcpyDeviceToHost, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(h_ry, s.d_ry, N * sizeof(float), cudaMemcpyDeviceToHost, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(h_angle, s.d_angle, N * sizeof(float), cudaMemcpyDeviceToHost, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(h_colors, s.d_colors, N * 3 * sizeof(float), cudaMemcpyDeviceToHost, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(h_opacity, s.d_opacity, N * sizeof(float), cudaMemcpyDeviceToHost, s.stream));
    if (h_type_logits) {
        CUDA_CHECK(cudaMemcpyAsync(h_type_logits, s.d_type_logits, N * s.num_template_types * sizeof(float), cudaMemcpyDeviceToHost, s.stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(s.stream));
}

void DiffRenderer::upload_params_soa(
    const float* h_cx, const float* h_cy, const float* h_rx, const float* h_ry,
    const float* h_angle, const float* h_colors, const float* h_opacity,
    const float* h_type_logits, int N)
{
    auto& s = scratch_;
    // Logit parameterization: h_colors and h_opacity are logits (direct copy to GPU).
    // Caller is responsible for any linear conversion before calling this function.
    // Upload all SoA arrays
    CUDA_CHECK(cudaMemcpyAsync(s.d_cx, h_cx, N * sizeof(float), cudaMemcpyHostToDevice, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(s.d_cy, h_cy, N * sizeof(float), cudaMemcpyHostToDevice, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(s.d_rx, h_rx, N * sizeof(float), cudaMemcpyHostToDevice, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(s.d_ry, h_ry, N * sizeof(float), cudaMemcpyHostToDevice, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(s.d_angle, h_angle, N * sizeof(float), cudaMemcpyHostToDevice, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(s.d_colors, h_colors, N * 3 * sizeof(float), cudaMemcpyHostToDevice, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(s.d_opacity, h_opacity, N * sizeof(float), cudaMemcpyHostToDevice, s.stream));

    // STE: D2D copy params to STE buffers (same as upload_params_gpu)
    CUDA_CHECK(cudaMemcpyAsync(s.d_ste_cx, s.d_cx, N * sizeof(float), cudaMemcpyDeviceToDevice, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(s.d_ste_cy, s.d_cy, N * sizeof(float), cudaMemcpyDeviceToDevice, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(s.d_ste_rx, s.d_rx, N * sizeof(float), cudaMemcpyDeviceToDevice, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(s.d_ste_ry, s.d_ry, N * sizeof(float), cudaMemcpyDeviceToDevice, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(s.d_ste_angle, s.d_angle, N * sizeof(float), cudaMemcpyDeviceToDevice, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(s.d_ste_colors, s.d_colors, N * 3 * sizeof(float), cudaMemcpyDeviceToDevice, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(s.d_ste_opacity, s.d_opacity, N * sizeof(float), cudaMemcpyDeviceToDevice, s.stream));

    if (h_type_logits) {
        CUDA_CHECK(cudaMemcpyAsync(s.d_type_logits, h_type_logits, N * s.num_template_types * sizeof(float), cudaMemcpyHostToDevice, s.stream));
        CUDA_CHECK(cudaMemcpyAsync(s.d_ste_type_logits, s.d_type_logits, N * s.num_template_types * sizeof(float), cudaMemcpyDeviceToDevice, s.stream));
    }
}

void DiffRenderer::set_frozen_mask(const int* frozen, int N) {
    auto& s = scratch_;
    CUDA_CHECK(cudaMemcpyAsync(s.d_frozen_mask, frozen, N * sizeof(int),
                               cudaMemcpyHostToDevice, s.stream));
}

void DiffRenderer::clamp_opacity(int N, float logit_min) {
    auto& s = scratch_;
    if (!initialized_ || N <= 0 || !s.d_opacity || !s.d_ste_opacity) return;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    clamp_opacity_kernel<<<blocks, threads, 0, s.stream>>>(
        s.d_opacity, s.d_ste_opacity, logit_min, N);
}

void DiffRenderer::download_grads_packed(float* h_grads, int N) {
    auto& s = scratch_;
    CUDA_CHECK(cudaMemcpyAsync(h_grads, s.d_grads_packed, N * (9 + s.num_template_types) * sizeof(float),
                               cudaMemcpyDeviceToHost, s.stream));
    CUDA_CHECK(cudaStreamSynchronize(s.stream));
}

void DiffRenderer::reset_adam_state_gpu(const std::vector<int>& indices) {
    auto& s = scratch_;
    int num = static_cast<int>(indices.size());
    if (num == 0) return;

    // Upload indices to GPU and launch batch reset kernel
    CUDA_CHECK(cudaMemcpyAsync(s.d_reset_indices, indices.data(),
                               num * sizeof(int), cudaMemcpyHostToDevice, s.stream));

    int threads = 256;
    int blocks = (num + threads - 1) / threads;
    reset_adam_state_indices_kernel<<<blocks, threads, 0, s.stream>>>(
        s.d_reset_indices, num,
        s.d_m_cx, s.d_m_cy, s.d_m_rx, s.d_m_ry, s.d_m_angle,
        s.d_m_cr, s.d_m_cg, s.d_m_cb, s.d_m_ca,
        s.d_v_cx, s.d_v_cy, s.d_v_rx, s.d_v_ry, s.d_v_angle,
        s.d_v_cr, s.d_v_cg, s.d_v_cb, s.d_v_ca,
        s.d_m_logits, s.d_v_logits,
        s.num_template_types, s.num_template_types);
}

void DiffRenderer::reset_adam_state_all_gpu(int N) {
    auto& s = scratch_;
    // 合并 buffer: 一次 memset 清零所有参数的 m/v state
    // 注意：偏移指针布局为 [(9+num_template_types) * max_shapes]，每段 max_shapes 个 float，
    // 所以必须 memset 整个 max_shapes * (9+num_template_types) 范围，否则只有前几个参数（cx/cy/rx）的前 N 个
    // 元素被清零，ry/angle/colors/opacity/logits 的 Adam state 会保留旧值。
    CUDA_CHECK(cudaMemsetAsync(s.d_adam_m_buffer, 0, (size_t)max_shapes_ * (9 + s.num_template_types) * sizeof(float), s.stream));
    CUDA_CHECK(cudaMemsetAsync(s.d_adam_v_buffer, 0, (size_t)max_shapes_ * (9 + s.num_template_types) * sizeof(float), s.stream));
}

// Compact GPU Adam state: surviving shapes keep momentum, new shapes zeroed.
// Adam state layout: [max_shapes * (9 + num_types)] per buffer (m and v),
//   param p occupies [p * max_shapes, (p+1) * max_shapes).
// We download to CPU, gather per new_to_old, and upload back — reloc runs
// infrequently (every ~100 iters) so the D2H/H2D cost (~300 KB) is negligible.
void DiffRenderer::compact_adam_state_gpu(const std::vector<int>& new_to_old, int N) {
    auto& s = scratch_;
    int total_params = 9 + s.num_template_types;
    size_t buf_f32 = (size_t)max_shapes_ * total_params;
    size_t buf_bytes = buf_f32 * sizeof(float);

    std::vector<float> h_m(buf_f32), h_v(buf_f32);
    CUDA_CHECK(cudaMemcpyAsync(h_m.data(), s.d_adam_m_buffer, buf_bytes, cudaMemcpyDeviceToHost, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(h_v.data(), s.d_adam_v_buffer, buf_bytes, cudaMemcpyDeviceToHost, s.stream));
    CUDA_CHECK(cudaStreamSynchronize(s.stream));

    std::vector<float> h_m_new(buf_f32, 0.0f), h_v_new(buf_f32, 0.0f);
    for (int new_i = 0; new_i < N; ++new_i) {
        int old_i = new_to_old[new_i];
        if (old_i < 0) continue;  // new/reloc shape: leave zeroed
        for (int p = 0; p < total_params; ++p) {
            h_m_new[(size_t)p * max_shapes_ + new_i] = h_m[(size_t)p * max_shapes_ + old_i];
            h_v_new[(size_t)p * max_shapes_ + new_i] = h_v[(size_t)p * max_shapes_ + old_i];
        }
    }

    CUDA_CHECK(cudaMemcpyAsync(s.d_adam_m_buffer, h_m_new.data(), buf_bytes, cudaMemcpyHostToDevice, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(s.d_adam_v_buffer, h_v_new.data(), buf_bytes, cudaMemcpyHostToDevice, s.stream));
}

void DiffRenderer::upload_type_logits(const float* logits, int N) {
    auto& s = scratch_;
    CUDA_CHECK(cudaMemcpyAsync(s.d_type_logits, logits, N * s.num_template_types * sizeof(float),
                               cudaMemcpyHostToDevice, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(s.d_ste_type_logits, s.d_type_logits, N * s.num_template_types * sizeof(float),
                               cudaMemcpyDeviceToDevice, s.stream));
}

void DiffRenderer::download_type_logits(float* logits, int N) {
    auto& s = scratch_;
    CUDA_CHECK(cudaMemcpyAsync(logits, s.d_type_logits, N * s.num_template_types * sizeof(float),
                               cudaMemcpyDeviceToHost, s.stream));
    CUDA_CHECK(cudaStreamSynchronize(s.stream));
}

void DiffRenderer::download_type_logits_async(int N) {
#ifndef __NODEBUG__
    auto& s = scratch_;
    int write_buf = debug_buf_idx_;
    CUDA_CHECK(cudaMemcpyAsync(h_debug_logits_pinned_[write_buf], s.d_type_logits,
                               N * s.num_template_types * sizeof(float), cudaMemcpyDeviceToHost, s.stream));
    CUDA_CHECK(cudaEventRecord(debug_events_[write_buf], s.stream));
    debug_first_[write_buf] = false;
    debug_buf_idx_ = 1 - write_buf;
#else
    (void)N;  // __NODEBUG__: no-op
#endif
}

bool DiffRenderer::try_process_type_logits(int N, float* h_logits_out) {
#ifndef __NODEBUG__
    int read_buf = debug_buf_idx_;
    if (debug_first_[read_buf]) return false;
    if (cudaEventQuery(debug_events_[read_buf]) != cudaSuccess) return false;
    memcpy(h_logits_out, h_debug_logits_pinned_[read_buf], N * scratch_.num_template_types * sizeof(float));
    debug_first_[read_buf] = true;
    return true;
#else
    (void)N; (void)h_logits_out;  // __NODEBUG__: always returns false
    return false;
#endif
}

void DiffRenderer::set_render_mode(int mode) {
    scratch_.render_mode = mode;
}

void DiffRenderer::lock_types(const int* h_types, int N) {
    auto& s = scratch_;
    CUDA_CHECK(cudaMemcpyAsync(s.d_shape_types, h_types, N * sizeof(int),
                               cudaMemcpyHostToDevice, s.stream));
}

void DiffRenderer::upload_types(const int* h_types, int N) {
    auto& s = scratch_;
    CUDA_CHECK(cudaMemcpyAsync(s.d_shape_types, h_types, N * sizeof(int),
                               cudaMemcpyHostToDevice, s.stream));
}

void DiffRenderer::download_types(int* h_types, int N) {
    auto& s = scratch_;
    CUDA_CHECK(cudaMemcpyAsync(h_types, s.d_shape_types, N * sizeof(int),
                               cudaMemcpyDeviceToHost, s.stream));
    CUDA_CHECK(cudaStreamSynchronize(s.stream));
}

void DiffRenderer::sync_stream() {
    if (!initialized_) return;
    CUDA_CHECK(cudaStreamSynchronize(scratch_.stream));
}

void DiffRenderer::compute_error_map_gpu() {
    if (!initialized_) return;
    auto& s = scratch_;
    int threads = 256;
    int blocks = (s.total_pixels + threads - 1) / threads;
    compute_error_map_kernel<<<blocks, threads, 0, s.stream>>>(
        s.d_output, s.d_target, s.d_mask, s.d_error_map_scratch, s.total_pixels);
    CUDA_KERNEL_CHECK();
}

void DiffRenderer::download_error_map(float* h_error_map) {
    if (!initialized_) return;
    auto& s = scratch_;
    CUDA_CHECK(cudaMemcpyAsync(h_error_map, s.d_error_map_scratch,
                               s.total_pixels * sizeof(float), cudaMemcpyDeviceToHost, s.stream));
    CUDA_CHECK(cudaStreamSynchronize(s.stream));
}

void DiffRenderer::compute_error_map_and_mean() {
    // EM-YMP: 在 backward 前调用。基于当前 forward 渲染结果 d_output 重算逐像素误差图，
    // 并归约出全局平均误差 mean_err（device 标量，避免每步 D2H sync）。
    // 全部在同 stream 上，与后续 backward kernel 保持流顺序（无需显式 sync）。
    if (!initialized_) return;
    auto& s = scratch_;
    int threads = 256;
    int blocks = (s.total_pixels + threads - 1) / threads;

    // 1. 逐像素平方误差（masked 像素置 0）
    compute_error_map_kernel<<<blocks, threads, 0, s.stream>>>(
        s.d_output, s.d_target, s.d_mask, s.d_error_map_scratch, s.total_pixels);
    CUDA_KERNEL_CHECK();

    // 2. 归约 sum(error_map)
    CUDA_CHECK(cudaMemsetAsync(s.d_error_sum, 0, sizeof(float), s.stream));
    sum_error_map_kernel<<<blocks, threads, 0, s.stream>>>(
        s.d_error_map_scratch, s.d_error_sum, s.total_pixels);
    CUDA_KERNEL_CHECK();

    // 3. mean_err = sum / valid_count
    compute_mean_err_kernel<<<1, 1, 0, s.stream>>>(
        s.d_error_sum, s.d_mean_err, valid_pixel_count_);
    CUDA_KERNEL_CHECK();

#ifndef __NODEBUG__
    // 一次性数值校验：前 3 次调用打印 mean_err / error_sum / valid_count，
    // 并下载 error_map 用 CPU 独立重算 mean 交叉验证归约正确性。
    {
        static int em_dbg = 0;
        if (em_dbg < 3) {
            float h_sum = 0.0f, h_mean = 0.0f;
            CUDA_CHECK(cudaMemcpyAsync(&h_sum, s.d_error_sum, sizeof(float), cudaMemcpyDeviceToHost, s.stream));
            CUDA_CHECK(cudaMemcpyAsync(&h_mean, s.d_mean_err, sizeof(float), cudaMemcpyDeviceToHost, s.stream));
            CUDA_CHECK(cudaStreamSynchronize(s.stream));
            std::vector<float> h_em(s.total_pixels);
            CUDA_CHECK(cudaMemcpy(h_em.data(), s.d_error_map_scratch, s.total_pixels * sizeof(float), cudaMemcpyDeviceToHost));
            double cpu_sum = 0.0; int nz = 0;
            for (int p = 0; p < s.total_pixels; ++p) { cpu_sum += h_em[p]; if (h_em[p] > 0.0f) nz++; }
            double cpu_mean = (valid_pixel_count_ > 0) ? cpu_sum / valid_pixel_count_ : 0.0;
            VIN_INFO("[EM-YMP dbg %d] valid_count=%d  gpu_sum=%.1f gpu_mean=%.4f  cpu_sum=%.1f cpu_mean=%.4f (nz=%d)",
                     em_dbg, valid_pixel_count_, h_sum, h_mean, cpu_sum, cpu_mean, nz);
            em_dbg++;
        }
    }
#endif
}

void DiffRenderer::compute_fa_loss_stats(int N) {
    // FA-Loss: 在 forward 后、backward 前调用。
    // 三 pass 流水线（全在同 stream 上，与后续 backward 保持流顺序）：
    //   Pass 1 (cuda_fa_stats):  每像素遍历形状累加 per-shape sum_w/sum_wI[3]/sum_wI2
    //   Pass 2 (cuda_fa_reduce): 每形状算 A_i/μ_i/σ²_i，归约全局 s=ΣA, T=ΣA·σ²
    //   Pass 3 (cuda_fa_finalize): 计算 fa_scale=λ/(v·s)、fa_mean_t=T/s，写回 d_fa_globals
    // backward kernel 通过 fa_scale/fa_mean_t/d_fa_mu 注入 dL_fa/d_ar 梯度。
    if (!initialized_ || N <= 0 || scratch_.fa_lambda <= 0.0f) return;
    auto& s = scratch_;

    // 清零 stats 和 globals（reduce 用 atomicAdd 累加）
    CUDA_CHECK(cudaMemsetAsync(s.d_fa_stats, 0, (size_t)N * 5 * sizeof(float), s.stream));
    CUDA_CHECK(cudaMemsetAsync(s.d_fa_globals, 0, 3 * sizeof(float), s.stream));

    // Pass 1: 使用 STE 参数（与渲染/visibility kernel 一致）。VW-FA 需 opacity 计算透过率。
    cuda_fa_stats(
        s.d_px, s.d_py,
        s.d_ste_cx, s.d_ste_cy, s.d_ste_rx, s.d_ste_ry,
        s.d_sina, s.d_cosa,
        s.d_shape_types,
        s.d_ste_opacity,
        s.d_blurred_templates,
        s.d_target, s.d_mask,
        s.d_fa_stats,
        s.d_all_active_indices, N, N, s.total_pixels,
        s.d_softmax_w,
        s.render_mode, blur_sigma_,
        s.stream,
        s.num_template_types,
        s.num_builtins);
    CUDA_KERNEL_CHECK();

    // Pass 2: 每形状算 μ_i 和 σ²_i，归约全局 s/T
    cuda_fa_reduce(
        s.d_fa_stats, s.d_fa_mu, s.d_fa_globals,
        N, s.stream);
    CUDA_KERNEL_CHECK();

    // Pass 3: finalize → fa_scale, fa_mean_t, fa_vis_weight
    cuda_fa_finalize(
        s.d_fa_globals,
        s.fa_lambda, s.fa_sigma2_global, s.fa_vis_weight,
        s.stream);
    CUDA_KERNEL_CHECK();
}

void DiffRenderer::compute_coverage_count(int N) {
    if (!initialized_ || N == 0) return;
    auto& s = scratch_;
    CUDA_CHECK(cudaMemsetAsync(s.d_coverage_count, 0, s.total_pixels * sizeof(int), s.stream));
    cuda_fused_coverage_count(
        s.d_px, s.d_py, s.d_ste_cx, s.d_ste_cy, s.d_ste_rx, s.d_ste_ry,
        s.d_sina, s.d_cosa,
        s.d_shape_types, s.d_ste_opacity,
        s.d_blurred_templates,
        s.d_coverage_count,
        s.d_all_active_indices, N, N, s.total_pixels,
        s.d_softmax_w, s.render_mode, blur_sigma_,
        s.stream, s.num_template_types, s.num_builtins);
}

void DiffRenderer::download_coverage_count(int* h_count, int total_pixels) {
    if (!initialized_) return;
    auto& s = scratch_;
    CUDA_CHECK(cudaMemcpyAsync(h_count, s.d_coverage_count,
                               total_pixels * sizeof(int), cudaMemcpyDeviceToHost, s.stream));
    CUDA_CHECK(cudaStreamSynchronize(s.stream));
}

void DiffRenderer::download_rendered_alpha(float* h_alpha) {
    if (!initialized_) return;
    auto& s = scratch_;
    CUDA_CHECK(cudaMemcpyAsync(h_alpha, s.d_rendered_alpha,
                               s.total_pixels * sizeof(float), cudaMemcpyDeviceToHost, s.stream));
    CUDA_CHECK(cudaStreamSynchronize(s.stream));
}

void DiffRenderer::upload_potential_field(const float* h_grad_x, const float* h_grad_y) {
    if (!initialized_) return;
    auto& s = scratch_;
    CUDA_CHECK(cudaMemcpyAsync(s.d_pf_grad_x, h_grad_x,
                               s.total_pixels * sizeof(float), cudaMemcpyHostToDevice, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(s.d_pf_grad_y, h_grad_y,
                               s.total_pixels * sizeof(float), cudaMemcpyHostToDevice, s.stream));
}

void DiffRenderer::apply_potential_force(int N, float lambda, float gate_k, float gate_temp) {
    if (!initialized_ || N <= 0) return;
    auto& s = scratch_;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    potential_force_kernel<<<blocks, threads, 0, s.stream>>>(
        s.d_cx, s.d_cy, s.d_ste_cx, s.d_ste_cy,
        s.d_pf_grad_x, s.d_pf_grad_y, s.d_rendered_alpha,
        N, canvas_w_, canvas_h_,
        lambda, gate_k, gate_temp);
    CUDA_KERNEL_CHECK();
}

// === Frequency-Hole Potential Field (FH-PF) ===
void DiffRenderer::compute_fh_vis_gate(int N, float vis_thresh, float vis_temp) {
    if (!initialized_ || N <= 0) return;
    auto& s = scratch_;
    if (!s.d_visibility || !s.d_fh_vis_gate) return;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    fh_vis_gate_kernel<<<blocks, threads, 0, s.stream>>>(
        s.d_visibility, s.d_fh_vis_gate, N, vis_thresh, vis_temp);
    CUDA_KERNEL_CHECK();
}

void DiffRenderer::compute_aux_vis_gate(int N) {
    // g_i = 1 - sigmoid((vis_i - thresh)/temp)，写入 d_aux_gate [N]。
    // 复用 fh_vis_gate_kernel（同逻辑：低 vis→gate 大，高 vis→gate≈0），
    // 区别仅在输出 buffer（d_aux_gate）和门控参数（aux_vis_thresh/temp）。
    // g_i 视为常数权重（stop-gradient）：backward kernel 只读取，不对 vis_i 求导。
    if (!initialized_ || N <= 0) return;
    auto& s = scratch_;
    if (!s.d_visibility || !s.d_aux_gate) return;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    fh_vis_gate_kernel<<<blocks, threads, 0, s.stream>>>(
        s.d_visibility, s.d_aux_gate, N, s.aux_vis_thresh, s.aux_vis_temp);
    CUDA_KERNEL_CHECK();
}

void DiffRenderer::apply_fh_potential_force(int N, float lambda) {
    if (!initialized_ || N <= 0) return;
    auto& s = scratch_;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    fh_potential_force_kernel<<<blocks, threads, 0, s.stream>>>(
        s.d_cx, s.d_cy, s.d_ste_cx, s.d_ste_cy,
        s.d_pf_grad_x, s.d_pf_grad_y, s.d_fh_vis_gate,
        N, canvas_w_, canvas_h_, lambda);
    CUDA_KERNEL_CHECK();
}

// ============================================================
// SP-GEO (Stacking Penalty Geometric) — 拥挤度几何外力
// ============================================================

// SP-GEO 外力 kernel: 直接改位置/半径（及 STE），绕过 Adam。
//   gate = min(1, d(center) / d_ref)         // d_ref = 正 d 的 90 分位（host 算后上传）
//   pos -= force · gate · ∇d_norm(center)    // 沿负梯度 = 朝不拥挤
//   rx  -= shrink · d(center)                // 拥挤负担 → 收缩（降几何覆盖 ar>0）
//   ry  -= shrink · d(center)
// 不拥挤（d(center)≤0）时 gate=0、收缩=0，形状不受扰（保护已收敛形状）。
// d(center)>0 但 ∇d≈0 时仅收缩不移动（局部极值，缩小自身以降拥挤）。
__global__ void sp_geo_force_kernel(
    float* __restrict__ d_cx, float* __restrict__ d_cy,
    float* __restrict__ d_rx, float* __restrict__ d_ry,
    float* __restrict__ d_ste_cx, float* __restrict__ d_ste_cy,
    float* __restrict__ d_ste_rx, float* __restrict__ d_ste_ry,
    const float* __restrict__ d_field,   // [total_pixels] d(p)
    const float* __restrict__ grad_x,    // [total_pixels] 归一化 ∇d x
    const float* __restrict__ grad_y,    // [total_pixels] 归一化 ∇d y
    int N, int W, int H,
    float force, float shrink, float d_ref)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float cx = d_cx[i];
    float cy = d_cy[i];

    float d_center = pf_bilinear(d_field, cx, cy, W, H);
    if (d_center <= 0.0f) return;       // 不拥挤 → 不动（保护正常拟合形状）

    float inv_ref = (d_ref > 1e-8f) ? (1.0f / d_ref) : 0.0f;
    float gate = d_center * inv_ref;
    if (gate > 1.0f) gate = 1.0f;       // min(1, d/d_ref)

    // 位置：沿 -∇d 移动（朝不拥挤）
    float gx = pf_bilinear(grad_x, cx, cy, W, H);
    float gy = pf_bilinear(grad_y, cx, cy, W, H);
    cx = fminf(fmaxf(cx - force * gate * gx, 0.0f), (float)(W - 1));
    cy = fminf(fmaxf(cy - force * gate * gy, 0.0f), (float)(H - 1));

    // 半径：按 d(center) 收缩（拥挤负担越大缩越多）
    float rx = fmaxf(d_rx[i] - shrink * d_center, 0.5f);
    float ry = fmaxf(d_ry[i] - shrink * d_center, 0.5f);

    d_cx[i] = cx;
    d_cy[i] = cy;
    d_rx[i] = rx;
    d_ry[i] = ry;
    d_ste_cx[i] = cx;
    d_ste_cy[i] = cy;
    d_ste_rx[i] = rx;
    d_ste_ry[i] = ry;
}

void DiffRenderer::download_k_allowed(float* h_k, int total_pixels) {
    if (!initialized_ || !scratch_.d_k_allowed || !h_k) return;
    auto& s = scratch_;
    cudaStreamSynchronize(s.stream);
    CUDA_CHECK(cudaMemcpy(h_k, s.d_k_allowed, total_pixels * sizeof(float),
                          cudaMemcpyDeviceToHost));
}

void DiffRenderer::download_sp_S(float* h_S, int total_pixels) {
    if (!initialized_ || !scratch_.d_sp_S || !h_S) return;
    auto& s = scratch_;
    cudaStreamSynchronize(s.stream);
    CUDA_CHECK(cudaMemcpy(h_S, s.d_sp_S, total_pixels * sizeof(float),
                          cudaMemcpyDeviceToHost));
}

void DiffRenderer::download_sp_G(float* h_G, int total_pixels) {
    if (!initialized_ || !scratch_.d_sp_G || !h_G) return;
    auto& s = scratch_;
    cudaStreamSynchronize(s.stream);
    CUDA_CHECK(cudaMemcpy(h_G, s.d_sp_G, total_pixels * sizeof(float),
                          cudaMemcpyDeviceToHost));
}

void DiffRenderer::upload_sp_geo_field(const float* h_field, const float* h_gx,
                                       const float* h_gy, int total_pixels) {
    if (!initialized_) return;
    auto& s = scratch_;
    CUDA_CHECK(cudaMemcpyAsync(s.d_sp_geo_field, h_field, total_pixels * sizeof(float),
                               cudaMemcpyHostToDevice, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(s.d_sp_geo_grad_x, h_gx, total_pixels * sizeof(float),
                               cudaMemcpyHostToDevice, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(s.d_sp_geo_grad_y, h_gy, total_pixels * sizeof(float),
                               cudaMemcpyHostToDevice, s.stream));
}

void DiffRenderer::apply_sp_geo_force(int N, float force, float shrink) {
    if (!initialized_ || N <= 0) return;
    auto& s = scratch_;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    sp_geo_force_kernel<<<blocks, threads, 0, s.stream>>>(
        s.d_cx, s.d_cy, s.d_rx, s.d_ry,
        s.d_ste_cx, s.d_ste_cy, s.d_ste_rx, s.d_ste_ry,
        s.d_sp_geo_field, s.d_sp_geo_grad_x, s.d_sp_geo_grad_y,
        N, canvas_w_, canvas_h_, force, shrink, s.sp_geo_d_ref);
    CUDA_KERNEL_CHECK();
}

} // namespace vinylizer
