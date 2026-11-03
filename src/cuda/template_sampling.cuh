#pragma once

// Template-based alpha sampling for Vinylizer v7
// Replaces per-shape alpha blur with: pre-rendered templates + affine sampling
// Pipeline: generate_templates (once) -> blur_templates (per iter) -> sample_alpha (forward) -> template_backward (backward)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <algorithm>

#include "common/alpha_228.h"
#include "common/types.h"
#include "cuda/color_utils.cuh"

// ============================================================
// Constants
// ============================================================
// TEMPLATE_SIZE and MAX_TEMPLATE_TYPES are defined in common/types.h
// Template type indices:
//   0 = TYPE_228 (soft ellipse), 1 = TYPE_HARD_ELLIPSE,
//   2 = TYPE_RECT, 3 = TYPE_TRIANGLE
//   >=4 = external stamp types (runtime num_types)

#define PI_F 3.14159265358979323846f

// ============================================================
// Sigmoid helpers for logit parameterization of color and opacity.
// GPU-resident d_colors / d_opacity are stored in logit space
// (aligned with diffbmp's sigmoid(tile_v) / sigmoid(tile_c) parameterization).
// Render kernels convert logit -> linear at use; backward kernels multiply
// gradients by sigmoid'(s) = s*(1-s) for chain rule.
// ============================================================
__device__ __forceinline__ float sigmoid_logit(float x) {
    return 1.0f / (1.0f + expf(-x));
}
// s = sigmoid(x) already computed; returns ds/dx = s*(1-s)
__device__ __forceinline__ float sigmoid_deriv_from_s(float s) {
    return s * (1.0f - s);
}

// ============================================================
// Device helper: map old type values to template indices
// ============================================================

// shape_types 已统一为模板索引：
//   [0, num_builtins) 为内置图元，>= num_builtins 为外部图章。
// DARTS 哨兵 -1 在调用前已过滤，此处不处理。
__device__ __forceinline__ int to_template_idx(int type_val, int num_builtins) {
    if (type_val < 0) return 0;  // 异常回退
    return type_val;             // 已是模板索引，原样返回
}

// ============================================================
// Device helpers: bilinear sampling
// ============================================================

// Bilinear sample from a TEMPLATE_SIZE x TEMPLATE_SIZE template.
// u, v: normalized coords [-1, 1] with align_corners=True mapping.
// Returns sampled value; out-of-range returns 0.0.
__device__ __forceinline__ float bilinear_sample_template(
    const float* tmpl, float u, float v)
{
    const float S1 = (float)(TEMPLATE_SIZE - 1);  // 255.0f
    float fx = (u + 1.0f) * 0.5f * S1;  // [0, 255]
    float fy = (v + 1.0f) * 0.5f * S1;

    if (fx < 0.0f || fx > S1 || fy < 0.0f || fy > S1) return 0.0f;

    int x0 = (int)floorf(fx);
    int y0 = (int)floorf(fy);
    int x1 = min(x0 + 1, TEMPLATE_SIZE - 1);
    int y1 = min(y0 + 1, TEMPLATE_SIZE - 1);
    x0 = max(0, x0);
    y0 = max(0, y0);

    float wx = fx - (float)x0;
    float wy = fy - (float)y0;

    float v00 = tmpl[y0 * TEMPLATE_SIZE + x0];
    float v10 = tmpl[y0 * TEMPLATE_SIZE + x1];
    float v01 = tmpl[y1 * TEMPLATE_SIZE + x0];
    float v11 = tmpl[y1 * TEMPLATE_SIZE + x1];

    return (1.0f - wx) * (1.0f - wy) * v00
         + wx         * (1.0f - wy) * v10
         + (1.0f - wx) * wy         * v01
         + wx         * wy         * v11;
}

// ============================================================
// Device helper: template scale factors per type
// ============================================================

// Different shape types interpret rx/ry differently:
//   Ellipse/228: rx, ry = semi-axes          -> su=rx, sv=ry
//   Rectangle:   rx, ry = full width/height  -> su=rx*0.5, sv=ry*0.5
//   Triangle:    rx = half-width, ry = full height -> su=rx, sv=ry*0.5
// Template coords u,v in [-1,1] represent the shape boundary.
// num_builtins: 4 if built-in templates are present, 0 if pure external stamps.
//   Types < num_builtins are built-in; types >= num_builtins are external stamps.
__device__ __forceinline__ void get_template_scale(
    int type, float rx, float ry, float* su, float* sv, int num_builtins)
{
    // External stamps (including all types when num_builtins=0): uniform scale
    if (type >= num_builtins) {
        *su = rx; *sv = ry; return;
    }
    // Built-in types
    switch (type) {
        case 0: // 228 soft ellipse - rx,ry = semi-axes
        case 1: // hard ellipse - rx,ry = semi-axes
            *su = rx; *sv = ry; break;
        case 2: // rectangle - rx,ry = full width/height
            *su = rx * 0.5f; *sv = ry * 0.5f; break;
        case 3: // triangle - rx = half-width, ry = full height
            *su = rx; *sv = ry * 0.5f; break;
        default:
            *su = rx; *sv = ry; break;
    }
}

// ============================================================
// Device helper: bilinear sampling gradient
// ============================================================

// Returns d(alpha)/d(u) and d(alpha)/d(v) for backpropagation.
__device__ __forceinline__ void bilinear_sample_gradient(
    const float* tmpl, float u, float v,
    float* dadu, float* dadv)
{
    const float S1 = (float)(TEMPLATE_SIZE - 1);
    float fx = (u + 1.0f) * 0.5f * S1;
    float fy = (v + 1.0f) * 0.5f * S1;

    if (fx < 0.0f || fx > S1 || fy < 0.0f || fy > S1) {
        *dadu = 0.0f; *dadv = 0.0f; return;
    }

    int x0 = (int)floorf(fx);
    int y0 = (int)floorf(fy);
    int x1 = min(x0 + 1, TEMPLATE_SIZE - 1);
    int y1 = min(y0 + 1, TEMPLATE_SIZE - 1);
    x0 = max(0, x0);
    y0 = max(0, y0);

    float wx = fx - (float)x0;
    float wy = fy - (float)y0;

    float v00 = tmpl[y0 * TEMPLATE_SIZE + x0];
    float v10 = tmpl[y0 * TEMPLATE_SIZE + x1];
    float v01 = tmpl[y1 * TEMPLATE_SIZE + x0];
    float v11 = tmpl[y1 * TEMPLATE_SIZE + x1];

    // d(alpha)/d(fx) = (1-wy)*(v10-v00) + wy*(v11-v01)
    float dadfx = (1.0f - wy) * (v10 - v00) + wy * (v11 - v01);
    // d(alpha)/d(fy) = (1-wx)*(v01-v00) + wx*(v11-v10)
    float dadfy = (1.0f - wx) * (v01 - v00) + wx * (v11 - v10);

    // d(fx)/d(u) = 0.5 * S1, d(fy)/d(v) = 0.5 * S1
    *dadu = dadfx * 0.5f * S1;
    *dadv = dadfy * 0.5f * S1;
}

// ============================================================
// Template generation (CPU)
// ============================================================

// Generate 256x256 template data and upload to GPU.
// Always generates exactly 4 built-in templates.
// External stamps are loaded separately by DiffRenderer::load_external_stamps().
inline void generate_and_upload_templates(float* d_templates) {
    const int S = TEMPLATE_SIZE;
    const int S2 = S * S;
    const int num_builtins = 4;
    float* h_templates = new float[num_builtins * S2];

    for (int t = 0; t < num_builtins; t++) {
        float* tmpl = h_templates + t * S2;
        for (int y = 0; y < S; y++) {
            for (int x = 0; x < S; x++) {
                // Normalized coords [-1, 1] with align_corners=True
                float nx = ((float)x - (S - 1) * 0.5f) / ((S - 1) * 0.5f);
                float ny = ((float)y - (S - 1) * 0.5f) / ((S - 1) * 0.5f);
                float alpha = 0.0f;

                switch (t) {
                    case 0: {  // 228 soft ellipse
                        float rn = sqrtf(nx * nx + ny * ny);
                        // Use Newton-inverted alpha_228 for exact match with alpha_228_tex()
                        alpha = vinylizer::alpha_228_newton(rn);
                        break;
                    }
                    case 1: {  // Hard ellipse
                        float rn = sqrtf(nx * nx + ny * ny);
                        alpha = (rn < 1.0f) ? 1.0f : 0.0f;
                        break;
                    }
                    case 2: {  // Hard rectangle (square template)
                        alpha = (fabsf(nx) <= 1.0f && fabsf(ny) <= 1.0f) ? 1.0f : 0.0f;
                        break;
                    }
                    case 3: {  // Hard triangle (isosceles, pointing up)
                        // Vertices: (0,-1)=apex, (-1,1)=bottom-left, (1,1)=bottom-right
                        // Inside: |nx| <= (1+ny)/2 && ny >= -1
                        alpha = (fabsf(nx) <= (1.0f + ny) * 0.5f && ny >= -1.0f) ? 1.0f : 0.0f;
                        break;
                    }
                }
                tmpl[y * S + x] = alpha;
            }
        }
    }

    cudaMemcpy(d_templates, h_templates,
               num_builtins * S2 * sizeof(float), cudaMemcpyHostToDevice);
    delete[] h_templates;
}

// ============================================================
// Kernel declarations (definitions in template_sampling.cu)
// ============================================================

__global__ void template_blur_horizontal_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int S,
    const float* __restrict__ kernel, int radius, int ksize,
    int total_rows);

__global__ void template_blur_vertical_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int S,
    const float* __restrict__ kernel, int radius, int ksize,
    int total_images);

// ============================================================
// Visibility kernel (used by relocation)
// ============================================================

__global__ void fused_template_visibility_kernel(
    const float* __restrict__ px,
    const float* __restrict__ py,
    const float* __restrict__ cx, const float* __restrict__ cy,
    const float* __restrict__ rx, const float* __restrict__ ry,
    const float* __restrict__ sina, const float* __restrict__ cosa,
    const int* __restrict__ shape_types,
    const float* __restrict__ opacity,
    const float* __restrict__ d_blurred_templates,
    float* __restrict__ d_visibility,       // output: [N] — Σ(ae·T), 实际可见贡献
    float* __restrict__ d_visibility_full,   // output: [N] — Σ(ae), 无遮挡理论最大贡献
    const int* __restrict__ active_indices,
    int num_active, int N, int total_pixels,
    const float* __restrict__ softmax_w,
    int render_mode,
    float blur_sigma,
    int num_types,
    int num_builtins);

// ============================================================
// Coverage count kernel (diagnosis only)
// 统计每个像素被多少个形状覆盖（ar > 0），独立于渲染/visibility，不影响优化。
// ============================================================

__global__ void fused_coverage_count_kernel(
    const float* __restrict__ px,
    const float* __restrict__ py,
    const float* __restrict__ cx, const float* __restrict__ cy,
    const float* __restrict__ rx, const float* __restrict__ ry,
    const float* __restrict__ sina, const float* __restrict__ cosa,
    const int* __restrict__ shape_types,
    const float* __restrict__ opacity,
    const float* __restrict__ d_blurred_templates,
    int* __restrict__ d_coverage_count,   // output: [total_pixels] per-pixel 覆盖形状数
    const int* __restrict__ active_indices,
    int num_active, int N, int total_pixels,
    const float* __restrict__ softmax_w,
    int render_mode,
    float blur_sigma,
    int num_types,
    int num_builtins);

// ============================================================
// Tile-based forward kernel (batch-aware, compact scratch)
// ============================================================

__global__ void fused_tile_forward_kernel(
    const float* __restrict__ px,
    const float* __restrict__ py,
    const float* __restrict__ cx, const float* __restrict__ cy,
    const float* __restrict__ rx, const float* __restrict__ ry,
    const float* __restrict__ sina, const float* __restrict__ cosa,
    const int* __restrict__ shape_types,
    const float* __restrict__ colors, const float* __restrict__ opacity,
    const float* __restrict__ bg_color,
    const float* __restrict__ d_blurred_templates,
    float* __restrict__ d_output,
    float* __restrict__ d_rendered_alpha,
    float* __restrict__ d_sp_S,            // [total_pixels] SP-Loss: S(p)=Σae_i (nullptr=禁用)
    float* __restrict__ d_sp_G,            // [total_pixels] SP-GEO-Loss: G(p)=Σar_i (nullptr=禁用)
    __half* __restrict__ d_batch_scratch,
    const int* __restrict__ d_tile_shape_lists,
    const int* __restrict__ d_tile_shape_counts,
    int num_tiles_x, int num_tiles_y, int tile_size, int max_shapes_per_tile,
    int batch_tile_start, int batch_tile_count,
    int batch_total_pixels, int batch_max_shapes,
    int N, int W, int H, int total_pixels,
    const float* __restrict__ softmax_w,
    int render_mode, float blur_sigma,
    int num_types,
    int num_builtins,
    bool reverse,
    const float* __restrict__ d_z,        // soft depth per shape (nullptr=禁用)
    float tau_z,                          // soft depth temperature (<=0=禁用)
    int soft_depth_max_m);                // tile 内参与 pairwise 软遮挡的最大形状数

// ============================================================
// Tile-based backward kernel (batch-aware, local lTN computation)
// ============================================================

__global__ void tile_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ px, const float* __restrict__ py,
    const float* __restrict__ cx, const float* __restrict__ cy,
    const float* __restrict__ rx, const float* __restrict__ ry,
    const float* __restrict__ angle,
    const float* __restrict__ colors, const float* __restrict__ opacity,
    const float* __restrict__ bg_color,
    const float* __restrict__ sina,
    const float* __restrict__ cosa,
    const float* __restrict__ d_blurred_templates,
    float* __restrict__ grad_cx,
    float* __restrict__ grad_cy,
    float* __restrict__ grad_rx,
    float* __restrict__ grad_ry,
    float* __restrict__ grad_angle,
    float* __restrict__ grad_colors,
    float* __restrict__ grad_opacity,
    const int* __restrict__ d_tile_shape_lists,
    const int* __restrict__ d_tile_shape_counts,
    int num_tiles_x, int num_tiles_y, int tile_size, int max_shapes_per_tile,
    int batch_tile_start, int batch_tile_count,
    int W, int H,
    const float* __restrict__ grad_alpha,
    const float* __restrict__ softmax_w,
    float* __restrict__ grad_logits,
    const int* __restrict__ shape_types,
    int render_mode,
    float blur_sigma,
    bool output_quadrant,                       // 是否输出象限 RGB 梯度
    float* __restrict__ grad_rgb_quadrant,      // [N*4*3] 输出，可为 nullptr
    int num_types,
    int num_builtins,
    float inv_temperature,                      // 1/T，DARTS softmax 温度的倒数
    bool reverse,
    const float* __restrict__ d_z,              // soft depth (nullptr=禁用)
    float tau_z,                                // soft depth temperature
    int soft_depth_max_m,
    float* __restrict__ grad_z,                 // [N] z 梯度输出 (nullptr=禁用)
    float yield_lambda,                         // 让位梯度权重 (0=禁用)
    const float* __restrict__ error_map,        // [total_pixels] EM-YMP 逐像素平方误差 (nullptr=YMP 不调制)
    const float* __restrict__ d_mean_err,       // [1] 全局平均误差 device 标量 (nullptr=YMP 不调制)
    // === FA-Loss 注入参数（d_fa_globals!=nullptr 时启用）===
    const float* __restrict__ d_fa_globals,    // [3] {fa_scale=λ/v, fa_mean_t=T/s, fa_vis_weight} (nullptr=禁用)
    const float* __restrict__ d_fa_mu,          // [N*3] 每形状 μ_i 颜色 (nullptr=禁用)
    const float* __restrict__ d_target,         // [total_pixels*3] 目标图像
    const float* __restrict__ d_mask,           // [total_pixels] mask (1=opaque)
    // === Auxiliary Visibility Gradient 注入参数（aux_vis_lambda>0 且 d_aux_gate!=nullptr 时启用）===
    // 在 dlae（∂L/∂ae）上累加 ∂L_aux/∂ae_i = -λ·g_i·T_i，通过现有 ae→参数链传回。
    // g_i 视为常数权重（stop-gradient）：预计算上传，kernel 只读取 d_aux_gate[i]。
    float aux_vis_lambda,                       // 辅助可见性梯度权重 (0=禁用)
    const float* __restrict__ d_aux_gate,       // [N] g_i 唤醒权重 (nullptr=禁用)
    // === Stacking Penalty Loss 注入参数（sp_lambda>0 且 d_k_allowed!=nullptr 时启用）===
    // L_sp = (λ/N_pixels)·Σ_p max(0, W(p)-K_allowed(p))², W=S-C, S=Σae, C=1-Π(1-ae)
    // ∂L_sp/∂ae_i = (2λ/N_pixels)·max(0,W-K)·(1-Π_{j≠i}(1-ae_j))
    // S/C 在 Pass1 局部累加；K_allowed 复杂度自适应（预计算上传）。注入到 dlae。
    float sp_lambda,                            // SP-Loss 权重 (0=禁用，反向 bidir 传 0)
    const float* __restrict__ d_k_allowed,     // [total_pixels] K_allowed(p) (nullptr=禁用)
    // === SP-GEO-Loss: 几何覆盖层数损失（sp_geo_loss_active 且 sp_lambda>0 且 d_k_allowed!=nullptr 时启用）===
    // L_stack = (λ/N_pixels)·Σ_p max(0, G(p)-K_allowed(p))², G=Σar（不含 opacity）
    // ∂L/∂ar_k(p) = (2λ/N_pixels)·max(0,G-K) 直接注入 ar 链（不进 dlae，不被 opacity 衰减）
    // 参照 fa_grad_ar 模式：与 dlae*opi 并列累加到 lcx/lcy/lrx/lry/lan。
    bool sp_geo_loss_active,                    // true=几何覆盖损失模式（禁用旧 ae-based SP-Loss）
    // === R23: 确定性归约参数 ===
    bool deterministic,                         // true=写 per-block partials 替代 atomicAdd
    float* __restrict__ grad_partials,          // [num_blocks * N * 9] 确定性几何梯度 partials
    float* __restrict__ grad_logits_partials);  // [num_blocks * N * num_types] 确定性 logits 梯度 partials

// ============================================================
// Host wrapper declarations (extern "C", defined in template_sampling.cu)
// ============================================================

#ifdef __cplusplus
extern "C" {
#endif

// 方向3：设置 opacity 温度 τ（sigmoid 锐化）。τ=1.0 默认（baseline 一致）。
void cuda_set_opacity_tau(float tau);

// 设置 alpha_upper_bound（对齐 diffbmp，默认 1.0=无 cap，diffbmp 默认 0.5）
void cuda_set_alpha_upper_bound(float aub);

void cuda_template_init(float* d_templates);

void cuda_template_blur(
    const float* d_templates,
    float* d_blurred_templates,
    float* d_tmp,
    float sigma,
    cudaStream_t stream,
    float* d_kernel = nullptr,
    int num_images = 4);

// 通用交错图像高斯模糊（双模糊退火：目标图按当前 sigma 模糊）
// 布局 [H*W, C] 通道交错；sigma<=0 时退化为 src->dst 拷贝。
// d_kernel 需至少容纳 256 floats（sigma<=42）；传 nullptr 则内部临时分配。
void cuda_blur_image(
    const float* d_src,
    float* d_dst,
    float* d_tmp,
    int W, int H, int C,
    float sigma,
    cudaStream_t stream,
    float* d_kernel = nullptr);

void cuda_fused_template_visibility(
    const float* px, const float* py,
    const float* cx, const float* cy,
    const float* rx, const float* ry,
    const float* sina, const float* cosa,
    const int* shape_types,
    const float* opacity,
    const float* d_blurred_templates,
    float* d_visibility,
    float* d_visibility_full,
    const int* active_indices,
    int num_active, int N, int total_pixels,
    const float* softmax_w,
    int render_mode,
    float blur_sigma,
    cudaStream_t stream,
    int num_types,
    int num_builtins);

// Coverage count host wrapper (diagnosis only)
void cuda_fused_coverage_count(
    const float* px, const float* py,
    const float* cx, const float* cy,
    const float* rx, const float* ry,
    const float* sina, const float* cosa,
    const int* shape_types,
    const float* opacity,
    const float* d_blurred_templates,
    int* d_coverage_count,
    const int* active_indices,
    int num_active, int N, int total_pixels,
    const float* softmax_w,
    int render_mode,
    float blur_sigma,
    cudaStream_t stream,
    int num_types,
    int num_builtins);

// === Tile forward host wrapper ===
void cuda_fused_tile_forward(
    const float* px, const float* py,
    const float* cx, const float* cy, const float* rx, const float* ry,
    const float* sina, const float* cosa,
    const int* shape_types, const float* colors, const float* opacity,
    const float* bg_color, const float* d_blurred_templates,
    float* d_output, float* d_rendered_alpha,
    float* d_sp_S,
    float* d_sp_G,
    __half* d_batch_scratch,
    const int* d_tile_shape_lists, const int* d_tile_shape_counts,
    int num_tiles_x, int num_tiles_y, int tile_size, int max_shapes_per_tile,
    int batch_tile_start, int batch_tile_count,
    int batch_total_pixels, int batch_max_shapes,
    int N, int W, int H, int total_pixels,
    const float* softmax_w, int render_mode, float blur_sigma,
    cudaStream_t stream,
    int num_types,
    int num_builtins,
    bool reverse,
    const float* d_z, float tau_z, int soft_depth_max_m);

// === Tile backward host wrapper ===
void cuda_tile_backward(
    const float* grad_output, const float* srgb_out, const float* px, const float* py,
    const float* cx, const float* cy, const float* rx, const float* ry,
    const float* angle, const float* colors, const float* opacity,
    const float* bg_color, const float* sina, const float* cosa,
    const float* d_blurred_templates,
    float* grad_cx, float* grad_cy, float* grad_rx, float* grad_ry,
    float* grad_angle, float* grad_colors, float* grad_opacity,
    const int* d_tile_shape_lists, const int* d_tile_shape_counts,
    int num_tiles_x, int num_tiles_y, int tile_size, int max_shapes_per_tile,
    int batch_tile_start, int batch_tile_count,
    int W, int H,
    const float* grad_alpha, const float* softmax_w,
    float* grad_logits, const int* shape_types,
    int render_mode, float blur_sigma, cudaStream_t stream,
    bool output_quadrant, float* grad_rgb_quadrant,
    int num_types,
    int num_builtins,
    float inv_temperature,                     // 1/T，DARTS softmax 温度的倒数
    bool reverse,
    const float* d_z, float tau_z, int soft_depth_max_m,
    float* grad_z,
    int N,                                      // 形状总数（用于 yield 归一化 1/N）
    float yield_lambda,                        // 让位梯度权重 (0=禁用，正向 backward 传入，反向 bidir 传 0)
    const float* error_map,                    // [total_pixels] EM-YMP 误差图 (nullptr=YMP/反向 bidir)
    const float* d_mean_err,                   // [1] 全局平均误差 device 标量 (nullptr=YMP/反向 bidir)
    // === FA-Loss 注入参数 ===
    const float* d_fa_globals,                 // [3] {fa_scale, fa_mean_t} (nullptr=禁用)
    const float* d_fa_mu,                      // [N*3] 每形状 μ_i (nullptr=禁用)
    const float* d_target,                     // [total_pixels*3] 目标图像
    const float* d_mask,                       // [total_pixels] mask
    // === Auxiliary Visibility Gradient 注入参数 ===
    float aux_vis_lambda,                      // 辅助可见性梯度权重 (0=禁用，反向 bidir 传 0)
    const float* d_aux_gate,                   // [N] g_i 唤醒权重 (nullptr=禁用)
    // === Stacking Penalty Loss 注入参数 ===
    float sp_lambda,                           // SP-Loss 权重 (0=禁用，反向 bidir 传 0)
    const float* d_k_allowed,                 // [total_pixels] K_allowed(p) (nullptr=禁用)
    bool sp_geo_loss_active,                   // true=几何覆盖损失模式
    // === R23: 确定性归约参数 ===
    bool deterministic,                        // true=写 per-tile partials 替代 atomicAdd
    float* grad_partials,                      // [num_tiles * N * 9] 确定性梯度 partials
    float* grad_logits_partials);              // [num_tiles * N * num_types] 确定性 logits partials

// === Tile culling host wrapper ===
void cuda_cull_shapes_to_tiles(
    const float* cx, const float* cy,
    const float* rx, const float* ry,
    const float* sina, const float* cosa,
    const int* shape_types,
    int N, int W, int H,
    int tile_size, int num_tiles_x, int num_tiles_y, int max_shapes_per_tile,
    int* d_tile_shape_lists, int* d_tile_shape_counts, int* d_tile_overflow,
    int* h_tile_shape_counts,
    float blur_sigma, cudaStream_t stream,
    int num_types,
    int num_builtins,
    bool reverse);

// === FA-Loss stats kernels ===
// Pass 1: 每像素【逆序】遍历形状（顶→底），维护透过率链 T，VW 权重 w=ar·mask·T。
//         d_fa_stats layout: [N*5] = {sum_w, sum_wI_r, sum_wI_g, sum_wI_b, sum_wI2}
void cuda_fa_stats(
    const float* px, const float* py,
    const float* cx, const float* cy, const float* rx, const float* ry,
    const float* sina, const float* cosa,
    const int* shape_types,
    const float* opacity,      // [N] opacity logits (STE)
    const float* d_blurred_templates,
    const float* d_target,     // [total_pixels*3]
    const float* d_mask,       // [total_pixels]
    float* d_fa_stats,         // [N*5] (caller 已 memset 为 0)
    const int* active_indices,
    int num_active, int N, int total_pixels,
    const float* softmax_w,
    int render_mode, float blur_sigma,
    cudaStream_t stream,
    int num_types,
    int num_builtins);

// Pass 2: 每形状计算 A_i/μ_i/σ²_i，atomicAdd 全局 s/T，写 μ_i 到 d_fa_mu。
//         d_fa_globals layout: [2] = {s=ΣA, T=ΣA·σ²} (caller 已 memset 为 0)
void cuda_fa_reduce(
    const float* d_fa_stats,    // [N*5]
    float* d_fa_mu,            // [N*3] output
    float* d_fa_globals,       // [2] output {s, T}
    int N,
    cudaStream_t stream);

// Pass 3: 单线程，读 s/T，计算 fa_scale=λ/v（量级修正，去 s 归一化）、fa_mean_t=T/s、fa_vis_weight，写回 d_fa_globals。
void cuda_fa_finalize(
    float* d_fa_globals,       // [3] in-out: {s, T} → {fa_scale, fa_mean_t, fa_vis_weight}
    float lambda,
    float sigma2_global,
    float vis_weight,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif
