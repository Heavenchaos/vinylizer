// Template sampling kernel definitions + host wrappers
// Kernel declarations are in template_sampling.cuh

#include "cuda/template_sampling.cuh"

// ============================================================
// Template blur kernels (separable Gaussian)
// ============================================================

__global__ void template_blur_horizontal_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int S,
    const float* __restrict__ kernel, int radius, int ksize,
    int total_rows)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_rows * S) return;

    int col = idx % S;
    int row = idx / S;

    float sum = 0.0f;
    for (int k = 0; k < ksize; k++) {
        int src_col = min(max(col - radius + k, 0), S - 1);
        sum += kernel[k] * input[row * S + src_col];
    }
    output[row * S + col] = sum;
}

__global__ void template_blur_vertical_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int S,
    const float* __restrict__ kernel, int radius, int ksize,
    int total_images)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_images * S * S) return;

    int col = idx % S;
    int row = (idx / S) % S;
    int tidx = idx / (S * S);

    float sum = 0.0f;
    for (int k = 0; k < ksize; k++) {
        int src_row = min(max(row - radius + k, 0), S - 1);
        sum += kernel[k] * input[(tidx * S + src_row) * S + col];
    }
    output[(tidx * S + row) * S + col] = sum;
}

// ============================================================
// Opacity temperature (方向3：温度退火)
// 前向：opi = sigmoid_logit(opacity[i] / g_opacity_tau)
// 反向：∂L/∂lo = (1/τ) · opi·(1-opi) · ∂L/∂opi
// τ=1.0 时与 baseline 完全一致（回归安全）。τ<1 锐化 sigmoid 趋向二值化。
// 用 __constant__ memory：所有 kernel 高速读取，host 用 cudaMemcpyToSymbol 更新。
// ============================================================
__constant__ float g_opacity_tau = 1.0f;

// ============================================================
// alpha_upper_bound（对齐 diffbmp alpha_upper_bound，默认 1.0=无 cap）
// 前向：ae = ar * opi * g_alpha_upper_bound
// 反向：链式法则自动乘上 g_alpha_upper_bound（梯度对 opacity 乘 cap）
// 默认 1.0 时与 baseline 完全一致。
// ============================================================
__constant__ float g_alpha_upper_bound = 1.0f;

// ============================================================
// Host wrapper implementations (extern "C")
// ============================================================

extern "C" {

void cuda_set_opacity_tau(float tau) {
    if (tau < 0.01f) tau = 0.01f;  // 数值保护：避免除零/极端放大
    cudaMemcpyToSymbol(g_opacity_tau, &tau, sizeof(float));
}

void cuda_set_alpha_upper_bound(float aub) {
    if (aub < 0.001f) aub = 0.001f;  // 数值保护
    if (aub > 1.0f) aub = 1.0f;
    cudaMemcpyToSymbol(g_alpha_upper_bound, &aub, sizeof(float));
}

void cuda_template_init(float* d_templates) {
    generate_and_upload_templates(d_templates);
}

void cuda_template_blur(
    const float* d_templates,
    float* d_blurred_templates,
    float* d_tmp,
    float sigma,
    cudaStream_t stream,
    float* d_kernel,
    int num_images)
{
    const int S = TEMPLATE_SIZE;
    const int S2 = S * S;
    const int total = num_images * S2;

    if (sigma <= 0.0f) {
        cudaMemcpyAsync(d_blurred_templates, d_templates,
                        total * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        return;
    }

    // ksize 公式与 vinylizer-cpp-v7 对齐（能跑出 32 dB 的原版配置）
    int radius = std::max(1, (int)std::ceil(3.0f * sigma));
    int ksize = 2 * radius + 1;
    // 双模糊退火支持 sigma 最大 ~42（ksize<=255）；超出则截断 radius 保护
    if (ksize > 255) {
        radius = 127;
        ksize = 255;
    }
    float h_kernel[256];
    float sum = 0.0f;
    for (int i = -radius; i <= radius; i++) {
        float val = expf(-0.5f * (float)i * (float)i / (sigma * sigma));
        h_kernel[i + radius] = val;
        sum += val;
    }
    for (int i = 0; i < ksize; i++) h_kernel[i] /= sum;

    // Use pre-allocated kernel buffer if available, otherwise allocate (fallback)
    float* d_k = d_kernel;
    bool own_kernel = false;
    if (d_k == nullptr) {
        cudaMalloc(&d_k, ksize * sizeof(float));
        own_kernel = true;
    }
    cudaMemcpyAsync(d_k, h_kernel, ksize * sizeof(float), cudaMemcpyHostToDevice, stream);

    int threads = 256;

    int h_total = num_images * S;
    int h_elems = h_total * S;
    int h_blocks = (h_elems + threads - 1) / threads;
    template_blur_horizontal_kernel<<<h_blocks, threads, 0, stream>>>(
        d_templates, d_tmp, S, d_k, radius, ksize, h_total);

    int v_elems = total;
    int v_blocks = (v_elems + threads - 1) / threads;
    template_blur_vertical_kernel<<<v_blocks, threads, 0, stream>>>(
        d_tmp, d_blurred_templates, S, d_k, radius, ksize, num_images);

    if (own_kernel) {
        cudaFree(d_k);
    }
}

// ============================================================
// Generic interleaved image Gaussian blur (separable, clamp boundary)
// 用于双模糊退火：对目标图按当前 sigma 做高斯模糊。
// 数据布局：[H*W, C] 行优先、通道交错（与 d_target / d_output 一致）。
// ============================================================

__global__ void image_blur_horizontal_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int W, int H, int C,
    const float* __restrict__ kernel, int radius, int ksize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= H * W) return;

    int x = idx % W;
    int y = idx / W;

    for (int c = 0; c < C; c++) {
        float sum = 0.0f;
        for (int k = 0; k < ksize; k++) {
            int src_x = min(max(x - radius + k, 0), W - 1);
            sum += kernel[k] * input[(y * W + src_x) * C + c];
        }
        output[(y * W + x) * C + c] = sum;
    }
}

__global__ void image_blur_vertical_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int W, int H, int C,
    const float* __restrict__ kernel, int radius, int ksize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= H * W) return;

    int x = idx % W;
    int y = idx / W;

    for (int c = 0; c < C; c++) {
        float sum = 0.0f;
        for (int k = 0; k < ksize; k++) {
            int src_y = min(max(y - radius + k, 0), H - 1);
            sum += kernel[k] * input[(src_y * W + x) * C + c];
        }
        output[(y * W + x) * C + c] = sum;
    }
}

void cuda_blur_image(
    const float* d_src,
    float* d_dst,
    float* d_tmp,
    int W, int H, int C,
    float sigma,
    cudaStream_t stream,
    float* d_kernel)
{
    const int total = W * H * C;

    if (sigma <= 0.0f) {
        cudaMemcpyAsync(d_dst, d_src, total * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        return;
    }

    int radius = std::max(1, (int)std::ceil(3.0f * sigma));
    int ksize = 2 * radius + 1;
    if (ksize > 255) {  // 与模板模糊同一上限（d_blur_kernel 预分配 256 floats）
        radius = 127;
        ksize = 255;
    }
    float h_kernel[256];
    float sum = 0.0f;
    for (int i = -radius; i <= radius; i++) {
        float val = expf(-0.5f * (float)i * (float)i / (sigma * sigma));
        h_kernel[i + radius] = val;
        sum += val;
    }
    for (int i = 0; i < ksize; i++) h_kernel[i] /= sum;

    float* d_k = d_kernel;
    bool own_kernel = false;
    if (d_k == nullptr) {
        cudaMalloc(&d_k, ksize * sizeof(float));
        own_kernel = true;
    }
    cudaMemcpyAsync(d_k, h_kernel, ksize * sizeof(float), cudaMemcpyHostToDevice, stream);

    int threads = 256;
    int blocks = (W * H + threads - 1) / threads;
    image_blur_horizontal_kernel<<<blocks, threads, 0, stream>>>(
        d_src, d_tmp, W, H, C, d_k, radius, ksize);
    image_blur_vertical_kernel<<<blocks, threads, 0, stream>>>(
        d_tmp, d_dst, W, H, C, d_k, radius, ksize);

    if (own_kernel) {
        cudaFree(d_k);
    }
}

} // extern "C"

// AABB check for a rotated shape. Returns true if pixel (px,py) might be inside
// the shape (including blur margin).
__device__ __forceinline__ bool shape_aabb_check(
    float pxi, float pyi,
    float cxi, float cyi, float rxc, float ryc,
    float coi, float sii, int stype, float blur_sigma,
    int num_builtins)
{
    float su, sv;
    get_template_scale(stype, rxc, ryc, &su, &sv, num_builtins);
    float half_w = fabsf(su * coi) + fabsf(sv * sii);
    float half_h = fabsf(su * sii) + fabsf(sv * coi);
    // Expand by blur margin (blur is in template space, map to pixel space)
    float margin = 3.0f * blur_sigma * fmaxf(su, sv) / (TEMPLATE_SIZE * 0.5f);
    half_w += margin;
    half_h += margin;
    return (fabsf(pxi - cxi) <= half_w && fabsf(pyi - cyi) <= half_h);
}

__global__ void fused_template_visibility_kernel(
    const float* __restrict__ px,
    const float* __restrict__ py,
    const float* __restrict__ cx, const float* __restrict__ cy,
    const float* __restrict__ rx, const float* __restrict__ ry,
    const float* __restrict__ sina, const float* __restrict__ cosa,
    const int* __restrict__ shape_types,
    const float* __restrict__ opacity,
    const float* __restrict__ d_blurred_templates,
    float* __restrict__ d_visibility,
    float* __restrict__ d_visibility_full,
    const int* __restrict__ active_indices,
    int num_active, int N, int total_pixels,
    const float* __restrict__ softmax_w,
    int render_mode,
    float blur_sigma,
    int num_types,
    int num_builtins)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pixels) return;
    float pxi = px[idx], pyi = py[idx];

    // 遍历顺序必须与渲染顺序一致：fused_tile_forward_kernel 按 0→N-1 处理，
    // over operator 中后处理（大索引）= 顶层（覆盖前序），前序（小索引）= 底层（被覆盖）。
    // 因此 T 衰减必须从顶层开始：vis[i] = ae · Π_{j>i}(1 - ae_j)
    //   顶层（i=N-1）：T=1，utilization 高（不被遮挡）
    //   底层（i=0）  ：T 衰减，utilization 低（被上层遮挡）
    float T = 1.0f;
    for (int j = num_active - 1; j >= 0; j--) {
        int i = active_indices[j];
        float coi = cosa[i], sii = sina[i];
        float rxc = fmaxf(rx[i], 0.5f), ryc = fmaxf(ry[i], 0.5f);

        int stype_for_aabb = (shape_types[i] < 0) ? 0 : to_template_idx(shape_types[i], num_builtins);
        if (!shape_aabb_check(pxi, pyi, cx[i], cy[i], rxc, ryc, coi, sii, stype_for_aabb, blur_sigma, num_builtins))
            continue;

        float dx = pxi - cx[i], dy = pyi - cy[i];
        float u_rot = dx * coi + dy * sii;
        float v_rot = -dx * sii + dy * coi;

        float ar = 0.0f;
        if (shape_types[i] < 0) {
            for (int t = 0; t < num_types; t++) {
                float wt = softmax_w[i * num_types + t];
                if (wt < 0.02f) continue;  // DARTS 提前剪枝
                float su, sv;
                get_template_scale(t, rxc, ryc, &su, &sv, num_builtins);
                float u = u_rot / su, v = v_rot / sv;
                ar += wt * bilinear_sample_template(d_blurred_templates + t * TEMPLATE_SIZE * TEMPLATE_SIZE, u, v);
            }
        } else {
            int stype = to_template_idx(shape_types[i], num_builtins);
            float su, sv;
            get_template_scale(stype, rxc, ryc, &su, &sv, num_builtins);
            float u = u_rot / su, v = v_rot / sv;
            ar = bilinear_sample_template(
                d_blurred_templates + stype * TEMPLATE_SIZE * TEMPLATE_SIZE, u, v);
        }

        if (ar > 0.0f) {
            // Logit parameterization: opacity stored as logit
            float opi = sigmoid_logit(opacity[i] / g_opacity_tau);
            float ae = ar * opi * g_alpha_upper_bound;
            atomicAdd(&d_visibility[i], ae * T);
            atomicAdd(&d_visibility_full[i], ae);  // 不计 T 衰减，反映形状自身最大可贡献
            T *= (1.0f - ae);
        }
    }
}

// ============================================================
// Coverage count kernel (diagnosis only)
// 与 visibility kernel 相同的遍历逻辑，但只统计 per-pixel 覆盖形状数（ar > 0）。
// 完全独立，不输出 vis/vis_full，不影响渲染/优化。
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
    int* __restrict__ d_coverage_count,
    const int* __restrict__ active_indices,
    int num_active, int N, int total_pixels,
    const float* __restrict__ softmax_w,
    int render_mode,
    float blur_sigma,
    int num_types,
    int num_builtins)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pixels) return;
    float pxi = px[idx], pyi = py[idx];

    int count = 0;
    for (int j = num_active - 1; j >= 0; j--) {
        int i = active_indices[j];
        float coi = cosa[i], sii = sina[i];
        float rxc = fmaxf(rx[i], 0.5f), ryc = fmaxf(ry[i], 0.5f);

        int stype_for_aabb = (shape_types[i] < 0) ? 0 : to_template_idx(shape_types[i], num_builtins);
        if (!shape_aabb_check(pxi, pyi, cx[i], cy[i], rxc, ryc, coi, sii, stype_for_aabb, blur_sigma, num_builtins))
            continue;

        float dx = pxi - cx[i], dy = pyi - cy[i];
        float u_rot = dx * coi + dy * sii;
        float v_rot = -dx * sii + dy * coi;

        float ar = 0.0f;
        if (shape_types[i] < 0) {
            for (int t = 0; t < num_types; t++) {
                float wt = softmax_w[i * num_types + t];
                if (wt < 0.02f) continue;
                float su, sv;
                get_template_scale(t, rxc, ryc, &su, &sv, num_builtins);
                float u = u_rot / su, v = v_rot / sv;
                ar += wt * bilinear_sample_template(d_blurred_templates + t * TEMPLATE_SIZE * TEMPLATE_SIZE, u, v);
            }
        } else {
            int stype = to_template_idx(shape_types[i], num_builtins);
            float su, sv;
            get_template_scale(stype, rxc, ryc, &su, &sv, num_builtins);
            float u = u_rot / su, v = v_rot / sv;
            ar = bilinear_sample_template(
                d_blurred_templates + stype * TEMPLATE_SIZE * TEMPLATE_SIZE, u, v);
        }

        if (ar > 0.0f) {
            count++;
        }
    }
    d_coverage_count[idx] = count;
}

// ============================================================
// Fused host wrapper implementations (extern "C")
// ============================================================

extern "C" {

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
    int num_builtins)
{
    if (num_active == 0 || total_pixels == 0) return;
    int threads = 256;
    int blocks = (total_pixels + threads - 1) / threads;
    fused_template_visibility_kernel<<<blocks, threads, 0, stream>>>(
        px, py, cx, cy, rx, ry, sina, cosa,
        shape_types, opacity, d_blurred_templates, d_visibility, d_visibility_full,
        active_indices, num_active, N, total_pixels,
        softmax_w, render_mode, blur_sigma, num_types, num_builtins);
}

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
    int num_builtins)
{
    if (num_active == 0 || total_pixels == 0) return;
    int threads = 256;
    int blocks = (total_pixels + threads - 1) / threads;
    fused_coverage_count_kernel<<<blocks, threads, 0, stream>>>(
        px, py, cx, cy, rx, ry, sina, cosa,
        shape_types, opacity, d_blurred_templates, d_coverage_count,
        active_indices, num_active, N, total_pixels,
        softmax_w, render_mode, blur_sigma, num_types, num_builtins);
}

} // extern "C" (fused)

// ============================================================
// Tile culling kernel (v2: per-tile, index-ascending write)
// ============================================================

__global__ void cull_shapes_to_tiles_kernel(
    const float* __restrict__ cx, const float* __restrict__ cy,
    const float* __restrict__ rx, const float* __restrict__ ry,
    const float* __restrict__ sina, const float* __restrict__ cosa,
    const int* __restrict__ shape_types,
    int N, int W, int H,
    int tile_size, int num_tiles_x, int num_tiles_y, int max_shapes_per_tile,
    int* __restrict__ d_tile_shape_lists,
    int* __restrict__ d_tile_shape_counts,
    int* __restrict__ d_tile_overflow,
    float blur_sigma,
    int num_types,
    int num_builtins,
    bool reverse)
{
    int tile_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tile_idx >= num_tiles_x * num_tiles_y) return;

    int tx = tile_idx % num_tiles_x;
    int ty = tile_idx / num_tiles_x;
    float tile_min_x = (float)(tx * tile_size);
    float tile_max_x = (float)min((tx + 1) * tile_size, W);
    float tile_min_y = (float)(ty * tile_size);
    float tile_max_y = (float)min((ty + 1) * tile_size, H);

    int count = 0;
    int* list = d_tile_shape_lists + tile_idx * max_shapes_per_tile;

    if (reverse) {
        for (int i = N - 1; i >= 0; i--) {
            float rxc = fmaxf(rx[i], 0.5f), ryc = fmaxf(ry[i], 0.5f);
            float coi = cosa[i], sii = sina[i];
            int stype = (shape_types[i] < 0) ? 0 : to_template_idx(shape_types[i], num_builtins);

            float su, sv;
            get_template_scale(stype, rxc, ryc, &su, &sv, num_builtins);
            float half_w = fabsf(su * coi) + fabsf(sv * sii);
            float half_h = fabsf(su * sii) + fabsf(sv * coi);
            float margin = 3.0f * blur_sigma * fmaxf(su, sv) / (TEMPLATE_SIZE * 0.5f);
            half_w += margin;
            half_h += margin;

            if (cx[i] + half_w < tile_min_x || cx[i] - half_w > tile_max_x) continue;
            if (cy[i] + half_h < tile_min_y || cy[i] - half_h > tile_max_y) continue;

            if (count < max_shapes_per_tile) {
                list[count++] = i;
            } else {
                atomicExch(d_tile_overflow, 1);
                break;
            }
        }
    } else {
        for (int i = 0; i < N; i++) {
            float rxc = fmaxf(rx[i], 0.5f), ryc = fmaxf(ry[i], 0.5f);
            float coi = cosa[i], sii = sina[i];
            int stype = (shape_types[i] < 0) ? 0 : to_template_idx(shape_types[i], num_builtins);

            float su, sv;
            get_template_scale(stype, rxc, ryc, &su, &sv, num_builtins);
            float half_w = fabsf(su * coi) + fabsf(sv * sii);
            float half_h = fabsf(su * sii) + fabsf(sv * coi);
            float margin = 3.0f * blur_sigma * fmaxf(su, sv) / (TEMPLATE_SIZE * 0.5f);
            half_w += margin;
            half_h += margin;

            if (cx[i] + half_w < tile_min_x || cx[i] - half_w > tile_max_x) continue;
            if (cy[i] + half_h < tile_min_y || cy[i] - half_h > tile_max_y) continue;

            if (count < max_shapes_per_tile) {
                list[count++] = i;
            } else {
                atomicExch(d_tile_overflow, 1);
                break;
            }
        }
    }
    d_tile_shape_counts[tile_idx] = count;
}

extern "C" {

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
    bool reverse)
{
    if (N == 0 || num_tiles_x == 0 || num_tiles_y == 0) return;

    cudaMemsetAsync(d_tile_shape_counts, 0, num_tiles_x * num_tiles_y * sizeof(int), stream);
    cudaMemsetAsync(d_tile_overflow, 0, sizeof(int), stream);

    int threads = 256;
    int blocks = (num_tiles_x * num_tiles_y + threads - 1) / threads;
    cull_shapes_to_tiles_kernel<<<blocks, threads, 0, stream>>>(
        cx, cy, rx, ry, sina, cosa, shape_types,
        N, W, H, tile_size, num_tiles_x, num_tiles_y, max_shapes_per_tile,
        d_tile_shape_lists, d_tile_shape_counts, d_tile_overflow, blur_sigma, num_types, num_builtins, reverse);

    cudaMemcpyAsync(h_tile_shape_counts, d_tile_shape_counts,
                    num_tiles_x * num_tiles_y * sizeof(int),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
}

} // extern "C" (culling)

// ============================================================
// FA-Loss stats kernel (Pass 1) —— VW-FA (Visibility-Weighted)
// 每像素【逆序】遍历形状（顶→底），维护透过率链 T：
//   active_indices 为 [0..N-1]（底→顶，与 culling/forward 合成顺序一致），
//   故逆序 j=num_active-1→0 即从顶层往底层遍历。
//   对形状 i（位置 j），T = ∏_{k>j}(1-ae_k) = 遮挡 i 的所有上层形状的残余透过率。
// VW 权重：w_i(p) = ar_i(p) · mask(p) · T_i(p)  （只统计实际可见贡献）
//   sum_w   = Σ_p ar_i(p) · mask(p) · T_i(p)
//   sum_wI_c = Σ_p ar_i(p) · mask(p) · T_i(p) · I_c(p)
//   sum_wI2 = Σ_p ar_i(p) · mask(p) · T_i(p) · ||I(p)||²
// ae = ar · opi（opi=sigmoid(opacity_logit)），与 forward over compositing 一致；
// 被完全遮挡的底层形状 T≈0 → 权重≈0 → 不被考核（修复"虚假退化"bug）。
// ============================================================
__global__ void fa_stats_kernel(
    const float* __restrict__ px,
    const float* __restrict__ py,
    const float* __restrict__ cx, const float* __restrict__ cy,
    const float* __restrict__ rx, const float* __restrict__ ry,
    const float* __restrict__ sina, const float* __restrict__ cosa,
    const int* __restrict__ shape_types,
    const float* __restrict__ opacity,            // [N] opacity logits (STE)
    const float* __restrict__ d_blurred_templates,
    const float* __restrict__ d_target,     // [total_pixels*3]
    const float* __restrict__ d_mask,       // [total_pixels]
    float* __restrict__ d_fa_stats,         // [N*5]
    const int* __restrict__ active_indices,
    int num_active, int N, int total_pixels,
    const float* __restrict__ softmax_w,
    int render_mode,
    float blur_sigma,
    int num_types,
    int num_builtins)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pixels) return;
    float pxi = px[idx], pyi = py[idx];
    float mask_val = d_mask[idx];
    if (mask_val < 0.5f) return;  // 透明像素不参与统计

    float Ir = d_target[idx * 3 + 0];
    float Ig = d_target[idx * 3 + 1];
    float Ib = d_target[idx * 3 + 2];
    float I_norm_sq = Ir * Ir + Ig * Ig + Ib * Ib;

    // 逆序遍历（顶→底），维护透过率 T：初始 1（顶层之上无遮挡）
    float T = 1.0f;
    for (int j = num_active - 1; j >= 0; j--) {
        int i = active_indices[j];
        float coi = cosa[i], sii = sina[i];
        float rxc = fmaxf(rx[i], 0.5f), ryc = fmaxf(ry[i], 0.5f);

        int stype_for_aabb = (shape_types[i] < 0) ? 0 : to_template_idx(shape_types[i], num_builtins);
        if (!shape_aabb_check(pxi, pyi, cx[i], cy[i], rxc, ryc, coi, sii, stype_for_aabb, blur_sigma, num_builtins))
            continue;

        float dx = pxi - cx[i], dy = pyi - cy[i];
        float u_rot = dx * coi + dy * sii;
        float v_rot = -dx * sii + dy * coi;

        float ar = 0.0f;
        if (shape_types[i] < 0) {
            for (int t = 0; t < num_types; t++) {
                float wt = softmax_w[i * num_types + t];
                if (wt < 0.02f) continue;
                float su, sv;
                get_template_scale(t, rxc, ryc, &su, &sv, num_builtins);
                float u = u_rot / su, v = v_rot / sv;
                ar += wt * bilinear_sample_template(d_blurred_templates + t * TEMPLATE_SIZE * TEMPLATE_SIZE, u, v);
            }
        } else {
            int stype = to_template_idx(shape_types[i], num_builtins);
            float su, sv;
            get_template_scale(stype, rxc, ryc, &su, &sv, num_builtins);
            float u = u_rot / su, v = v_rot / sv;
            ar = bilinear_sample_template(
                d_blurred_templates + stype * TEMPLATE_SIZE * TEMPLATE_SIZE, u, v);
        }

        if (ar > 0.0f) {
            float opi = sigmoid_logit(opacity[i] / g_opacity_tau);
            float ae = ar * opi * g_alpha_upper_bound;
            // VW 权重：名义覆盖 ar·mask × 上层残余透过率 T
            float w = ar * mask_val * T;
            atomicAdd(&d_fa_stats[i * 5 + 0], w);
            atomicAdd(&d_fa_stats[i * 5 + 1], w * Ir);
            atomicAdd(&d_fa_stats[i * 5 + 2], w * Ig);
            atomicAdd(&d_fa_stats[i * 5 + 3], w * Ib);
            atomicAdd(&d_fa_stats[i * 5 + 4], w * I_norm_sq);
            // 更新透过率供下一层（更底层）形状：T ← T·(1-ae)
            T *= (1.0f - ae);
            if (T < 1e-12f) T = 0.0f;  // 完全遮挡后底层 T=0，不再考核
        }
    }
}

extern "C" {

void cuda_fa_stats(
    const float* px, const float* py,
    const float* cx, const float* cy, const float* rx, const float* ry,
    const float* sina, const float* cosa,
    const int* shape_types,
    const float* opacity,
    const float* d_blurred_templates,
    const float* d_target,
    const float* d_mask,
    float* d_fa_stats,
    const int* active_indices,
    int num_active, int N, int total_pixels,
    const float* softmax_w,
    int render_mode, float blur_sigma,
    cudaStream_t stream,
    int num_types, int num_builtins)
{
    if (num_active == 0 || total_pixels == 0) return;
    int threads = 256;
    int blocks = (total_pixels + threads - 1) / threads;
    fa_stats_kernel<<<blocks, threads, 0, stream>>>(
        px, py, cx, cy, rx, ry, sina, cosa,
        shape_types, opacity, d_blurred_templates, d_target, d_mask, d_fa_stats,
        active_indices, num_active, N, total_pixels,
        softmax_w, render_mode, blur_sigma, num_types, num_builtins);
}

} // extern "C" (fa_stats)

// ============================================================
// FA-Loss reduce kernel (Pass 2)
// 每形状一个线程：读 stats → A_i, μ_i, σ²_i → 写 μ_i，atomicAdd 全局 s/T。
// ============================================================
__global__ void fa_reduce_kernel(
    const float* __restrict__ d_fa_stats,    // [N*5]
    float* __restrict__ d_fa_mu,             // [N*3]
    float* __restrict__ d_fa_globals,       // [3] {s, T} (reduce 写入 [0]=s, [1]=T)
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float sum_w   = d_fa_stats[i * 5 + 0];
    float sum_wIr = d_fa_stats[i * 5 + 1];
    float sum_wIg = d_fa_stats[i * 5 + 2];
    float sum_wIb = d_fa_stats[i * 5 + 3];
    float sum_wI2 = d_fa_stats[i * 5 + 4];

    // 数值稳定：A_i 可能很小（小形状或几乎不可见）
    float A_i = fmaxf(sum_w, 1e-6f);
    float mu_r = sum_wIr / A_i;
    float mu_g = sum_wIg / A_i;
    float mu_b = sum_wIb / A_i;

    // σ²_i = E[||I||²] - ||μ||²  (numerically stabilized; clamp to 0)
    float E2 = sum_wI2 / A_i;
    float mu_norm_sq = mu_r * mu_r + mu_g * mu_g + mu_b * mu_b;
    float sigma2_i = E2 - mu_norm_sq;
    if (sigma2_i < 0.0f) sigma2_i = 0.0f;  // 数值噪声可能导致负值

    d_fa_mu[i * 3 + 0] = mu_r;
    d_fa_mu[i * 3 + 1] = mu_g;
    d_fa_mu[i * 3 + 2] = mu_b;

    atomicAdd(&d_fa_globals[0], A_i);           // s += A_i
    atomicAdd(&d_fa_globals[1], A_i * sigma2_i); // T += A_i · σ²_i
}

extern "C" {

void cuda_fa_reduce(
    const float* d_fa_stats,
    float* d_fa_mu,
    float* d_fa_globals,
    int N,
    cudaStream_t stream)
{
    if (N == 0) return;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    fa_reduce_kernel<<<blocks, threads, 0, stream>>>(
        d_fa_stats, d_fa_mu, d_fa_globals, N);
}

} // extern "C" (fa_reduce)

// ============================================================
// FA-Loss finalize kernel (Pass 3)
// 单线程：读 s/T，计算 fa_scale=λ/v（量级修正：去掉 s 归一化，梯度放大 s 倍）、
// fa_mean_t=T/s，写回 d_fa_globals。
// 原 fa_scale=λ/(v·s) 因 s≈5e5 使梯度小 ~1e9 倍，λ 需 1e5 才见效且崩溃；
// 去 s 后 fa_scale=λ/v，梯度量级 O(λ)，与 MSE 可比，λ 工作区落到 1e-6~1e-2。
// ============================================================
__global__ void fa_finalize_kernel(
    float* __restrict__ d_fa_globals,    // [3] in-out: {s, T} → {fa_scale, fa_mean_t, fa_vis_weight}
    float lambda,
    float sigma2_global,
    float vis_weight)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float s = fmaxf(d_fa_globals[0], 1e-6f);
        float T = d_fa_globals[1];
        float v = fmaxf(sigma2_global, 1e-6f);
        // 量级修正：fa_scale = λ/v（不再除以 s）
        // dL_fa/d_ar_k(p) = (λ/v) · (||I(p)-μ_k||² - T/s) · mask(p) · lerp(1, T_k(p), vis_weight)
        d_fa_globals[0] = lambda / v;          // fa_scale（去掉 s 归一化）
        d_fa_globals[1] = T / s;               // fa_mean_t（全局加权平均 σ²）
        d_fa_globals[2] = vis_weight;          // fa_vis_weight（1=VW, 0=原FA）
    }
}

extern "C" {

void cuda_fa_finalize(
    float* d_fa_globals,
    float lambda,
    float sigma2_global,
    float vis_weight,
    cudaStream_t stream)
{
    fa_finalize_kernel<<<1, 1, 0, stream>>>(
        d_fa_globals, lambda, sigma2_global, vis_weight);
}

} // extern "C" (fa_finalize)

// ============================================================
// Tile forward kernel (batch-aware, compact scratch, linear composite)
// ============================================================

__device__ __forceinline__ int compute_tile_local_pixel_idx(
    int tile_idx, int batch_tile_start,
    int num_tiles_x, int num_tiles_y, int tile_size,
    int W, int H, int intra_tile_x, int intra_tile_y)
{
    int local_pixel_idx = 0;
    for (int t = batch_tile_start; t < tile_idx; t++) {
        int ttx = t % num_tiles_x, tty = t / num_tiles_x;
        int tw = min(tile_size, W - ttx * tile_size);
        int th = min(tile_size, H - tty * tile_size);
        local_pixel_idx += tw * th;
    }
    int tx = tile_idx % num_tiles_x;
    int tile_w = min(tile_size, W - tx * tile_size);
    local_pixel_idx += intra_tile_y * tile_w + intra_tile_x;
    return local_pixel_idx;
}

__global__ void fused_tile_forward_kernel(
    const float* __restrict__ px,
    const float* __restrict__ py,
    const float* __restrict__ cx, const float* __restrict__ cy,
    const float* __restrict__ rx, const float* __restrict__ ry,
    const float* __restrict__ sina, const float* __restrict__ cosa,
    const int* __restrict__ shape_types,
    const float* __restrict__ colors,
    const float* __restrict__ opacity,
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
    int render_mode,
    float blur_sigma,
    int num_types,
    int num_builtins,
    bool reverse,
    const float* __restrict__ d_z,
    float tau_z,
    int soft_depth_max_m)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pixels) return;
    float pxi = px[idx], pyi = py[idx];

    int tx = min((int)(pxi / tile_size), num_tiles_x - 1);
    int ty = min((int)(pyi / tile_size), num_tiles_y - 1);
    int tile_idx = ty * num_tiles_x + tx;

    if (tile_idx < batch_tile_start || tile_idx >= batch_tile_start + batch_tile_count) return;

    int intra_tile_x = (int)pxi - tx * tile_size;
    int intra_tile_y = (int)pyi - ty * tile_size;
    int local_pixel_idx = compute_tile_local_pixel_idx(
        tile_idx, batch_tile_start, num_tiles_x, num_tiles_y, tile_size, W, H,
        intra_tile_x, intra_tile_y);

    int count = min(d_tile_shape_counts[tile_idx], max_shapes_per_tile);
    const int* shape_list = d_tile_shape_lists + tile_idx * max_shapes_per_tile;

    // 线性空间混合：背景色以 sRGB 存储，先转线性（白 255 在两空间数值相同）
    float r = srgb_to_linear(bg_color[0]), g = srgb_to_linear(bg_color[1]), b = srgb_to_linear(bg_color[2]);
    float total_alpha = 0.0f;
    float sum_ae = 0.0f;   // SP-Loss: S(p) = Σ_i ae_i(p)（名义投入，不含遮挡衰减）
    float sum_ar = 0.0f;   // SP-GEO-Loss: G(p) = Σ_i ar_i(p)（几何覆盖层数，不含 opacity）

    // === Forward: 标准 over compositing (NeuralSort 风格) ===
    // 前向不再使用软混合，直接按 z 硬排序做 over compositing，与最终评估目标一致。
    // tau_z 仅用于 backward 的 z 梯度计算（STE 直通估计），不影响前向渲染。
    // 这样优化器直接学 over compositing 下的最优 z 排序，消除优化-评估偏差。
    if (reverse) {
        for (int k = count - 1; k >= 0; k--) {
            int i = shape_list[k];
            float coi = cosa[i], sii = sina[i];
            float rxc = fmaxf(rx[i], 0.5f), ryc = fmaxf(ry[i], 0.5f);

            int stype_for_aabb = (shape_types[i] < 0) ? 0 : to_template_idx(shape_types[i], num_builtins);

            float ar = 0.0f;
            if (shape_aabb_check(pxi, pyi, cx[i], cy[i], rxc, ryc, coi, sii, stype_for_aabb, blur_sigma, num_builtins)) {
                float dx = pxi - cx[i], dy = pyi - cy[i];
                float u_rot = dx * coi + dy * sii;
                float v_rot = -dx * sii + dy * coi;

                if (shape_types[i] < 0) {
                    ar = 0.0f;
                    for (int t = 0; t < num_types; t++) {
                        float wt = softmax_w[i * num_types + t];
                        if (wt < 0.02f) continue;
                        float su, sv;
                        get_template_scale(t, rxc, ryc, &su, &sv, num_builtins);
                        float u = u_rot / su, v = v_rot / sv;
                        ar += wt * bilinear_sample_template(d_blurred_templates + t * TEMPLATE_SIZE * TEMPLATE_SIZE, u, v);
                    }
                } else {
                    int stype = to_template_idx(shape_types[i], num_builtins);
                    float su, sv;
                    get_template_scale(stype, rxc, ryc, &su, &sv, num_builtins);
                    float u = u_rot / su, v = v_rot / sv;
                    ar = bilinear_sample_template(
                        d_blurred_templates + stype * TEMPLATE_SIZE * TEMPLATE_SIZE, u, v);
                }
            }

            d_batch_scratch[k * batch_total_pixels + local_pixel_idx] = __float2half(ar);

            if (ar <= 0.0f) continue;

            // Logit parameterization: opacity/colors stored as logits; convert to linear
            float opi = sigmoid_logit(opacity[i] / g_opacity_tau);
            float ae = ar * opi * g_alpha_upper_bound;
            if (d_sp_S) sum_ae += ae;   // SP-Loss: 累加有效 alpha（仅 SP 启用时，避免 baseline 副作用）
            if (d_sp_G) sum_ar += ar;   // SP-GEO-Loss: 累加几何覆盖 ar（不含 opacity）
            float oma = 1.0f - ae;
            // sRGB→linear：形状颜色以 sRGB 存储，混合须在线性空间进行
            float lr = srgb_to_linear(255.0f * sigmoid_logit(colors[i * 3 + 0]));
            float lg = srgb_to_linear(255.0f * sigmoid_logit(colors[i * 3 + 1]));
            float lb = srgb_to_linear(255.0f * sigmoid_logit(colors[i * 3 + 2]));
            r = ae * lr + oma * r;
            g = ae * lg + oma * g;
            b = ae * lb + oma * b;
            total_alpha = ae + oma * total_alpha;
        }
    } else {
        // === z-sorted over compositing ===
        // d_z 存在时按 z 降序排序（z 大=底层先画，z 小=顶层后画），与 preview 一致。
        // 使用插入排序在小数组上（count ≤ max_shapes_per_tile，通常 ≤ 64）。
        const int SM_BW = 96;
        int sorted_list[SM_BW];
        bool use_z_sort = (d_z != nullptr && count > 1 && count <= SM_BW);
        if (use_z_sort) {
            for (int k = 0; k < count; k++) sorted_list[k] = shape_list[k];
            // 插入排序：按 z 降序（d_z 大的在前）
            for (int k = 1; k < count; k++) {
                int key_idx = sorted_list[k];
                float key_z = d_z[key_idx];
                int j = k - 1;
                while (j >= 0 && d_z[sorted_list[j]] < key_z) {
                    sorted_list[j + 1] = sorted_list[j];
                    j--;
                }
                sorted_list[j + 1] = key_idx;
            }
        }
        const int* active_list = use_z_sort ? sorted_list : shape_list;

        for (int k = 0; k < count; k++) {
            int i = active_list[k];
            float coi = cosa[i], sii = sina[i];
            float rxc = fmaxf(rx[i], 0.5f), ryc = fmaxf(ry[i], 0.5f);

            int stype_for_aabb = (shape_types[i] < 0) ? 0 : to_template_idx(shape_types[i], num_builtins);

            float ar = 0.0f;
            if (shape_aabb_check(pxi, pyi, cx[i], cy[i], rxc, ryc, coi, sii, stype_for_aabb, blur_sigma, num_builtins)) {
                float dx = pxi - cx[i], dy = pyi - cy[i];
                float u_rot = dx * coi + dy * sii;
                float v_rot = -dx * sii + dy * coi;

                if (shape_types[i] < 0) {
                    ar = 0.0f;
                    for (int t = 0; t < num_types; t++) {
                        float wt = softmax_w[i * num_types + t];
                        if (wt < 0.02f) continue;
                        float su, sv;
                        get_template_scale(t, rxc, ryc, &su, &sv, num_builtins);
                        float u = u_rot / su, v = v_rot / sv;
                        ar += wt * bilinear_sample_template(d_blurred_templates + t * TEMPLATE_SIZE * TEMPLATE_SIZE, u, v);
                    }
                } else {
                    int stype = to_template_idx(shape_types[i], num_builtins);
                    float su, sv;
                    get_template_scale(stype, rxc, ryc, &su, &sv, num_builtins);
                    float u = u_rot / su, v = v_rot / sv;
                    ar = bilinear_sample_template(
                        d_blurred_templates + stype * TEMPLATE_SIZE * TEMPLATE_SIZE, u, v);
                }
            }

            d_batch_scratch[k * batch_total_pixels + local_pixel_idx] = __float2half(ar);

            if (ar <= 0.0f) continue;

            // Logit parameterization: opacity/colors stored as logits; convert to linear
            float opi = sigmoid_logit(opacity[i] / g_opacity_tau);
            float ae = ar * opi * g_alpha_upper_bound;
            if (d_sp_S) sum_ae += ae;   // SP-Loss: 累加有效 alpha（仅 SP 启用时，避免 baseline 副作用）
            if (d_sp_G) sum_ar += ar;   // SP-GEO-Loss: 累加几何覆盖 ar（不含 opacity）
            float oma = 1.0f - ae;
            // sRGB→linear：形状颜色以 sRGB 存储，混合须在线性空间进行
            float lr = srgb_to_linear(255.0f * sigmoid_logit(colors[i * 3 + 0]));
            float lg = srgb_to_linear(255.0f * sigmoid_logit(colors[i * 3 + 1]));
            float lb = srgb_to_linear(255.0f * sigmoid_logit(colors[i * 3 + 2]));
            r = ae * lr + oma * r;
            g = ae * lg + oma * g;
            b = ae * lb + oma * b;
            total_alpha = ae + oma * total_alpha;
        }
    }

    // linear→sRGB：输出回到 sRGB 空间（与目标图 / MSE 口径一致）
    d_output[idx * 3 + 0] = linear_to_srgb(r);
    d_output[idx * 3 + 1] = linear_to_srgb(g);
    d_output[idx * 3 + 2] = linear_to_srgb(b);
    d_rendered_alpha[idx] = total_alpha * 255.0f;
    if (d_sp_S) d_sp_S[idx] = sum_ae;   // SP-Loss: S(p)（[0,1] 概率口径，未归一化）
    if (d_sp_G) d_sp_G[idx] = sum_ar;   // SP-GEO-Loss: G(p)=Σar
}

// ============================================================
// Tile forward host wrappers
// ============================================================

extern "C" {

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
    const float* d_z, float tau_z, int soft_depth_max_m)
{
    if (N == 0 || total_pixels == 0 || batch_tile_count == 0) return;
    int threads = 256;
    int blocks = (total_pixels + threads - 1) / threads;
    fused_tile_forward_kernel<<<blocks, threads, 0, stream>>>(
        px, py, cx, cy, rx, ry, sina, cosa,
        shape_types, colors, opacity, bg_color,
        d_blurred_templates, d_output, d_rendered_alpha, d_sp_S, d_sp_G,
        d_batch_scratch,
        d_tile_shape_lists, d_tile_shape_counts,
        num_tiles_x, num_tiles_y, tile_size, max_shapes_per_tile,
        batch_tile_start, batch_tile_count,
        batch_total_pixels, batch_max_shapes,
        N, W, H, total_pixels,
        softmax_w, render_mode, blur_sigma, num_types, num_builtins, reverse,
        d_z, tau_z, soft_depth_max_m);
}

} // extern "C" (tile forward)

// ============================================================
// Tile backward kernel (batch-aware, local lTN computation in Pass 1)
// ============================================================

__global__ void tile_backward_kernel(
    const float* __restrict__ grad_output,
    // 渲染输出（sRGB 编码后的成品）：反向要按输出像素值取 sRGB 编码的导数，
    // 才能正确地把 dL/d(输出 sRGB) 换算成 dL/d(线性空间合成)。可为 nullptr（=跳过修正）。
    const float* __restrict__ srgb_out,
    const float* __restrict__ px, const float* __restrict__ py,
    const float* __restrict__ cx, const float* __restrict__ cy,
    const float* __restrict__ rx, const float* __restrict__ ry,
    const float* __restrict__ angle,
    const float* __restrict__ colors,
    const float* __restrict__ opacity,
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
    bool output_quadrant,
    float* __restrict__ grad_rgb_quadrant,
    int num_types,
    int num_builtins,
    float inv_temperature,
    bool reverse,
    const float* __restrict__ d_z,
    float tau_z,
    int soft_depth_max_m,
    float* __restrict__ grad_z,
    int N,
    float yield_lambda,
    const float* __restrict__ error_map,        // [total_pixels] EM-YMP 逐像素平方误差 (nullptr=YMP)
    const float* __restrict__ d_mean_err,       // [1] 全局平均误差 device 标量 (nullptr=YMP)
    // === FA-Loss 注入参数 ===
    const float* __restrict__ d_fa_globals,    // [3] {fa_scale=λ/v, fa_mean_t=T/s, fa_vis_weight} (nullptr=禁用)
    const float* __restrict__ d_fa_mu,          // [N*3] 每形状 μ_i (nullptr=禁用)
    const float* __restrict__ d_target,          // [total_pixels*3] 目标图像 (nullptr=禁用)
    const float* __restrict__ d_mask,            // [total_pixels] mask (nullptr=禁用)
    // === Auxiliary Visibility Gradient 注入（aux_vis_lambda>0 且 d_aux_gate!=nullptr 时启用）===
    // ∂L_aux/∂ae_i = -λ·g_i·T_i 累加到 dlae，g_i 为常数权重（stop-gradient）。
    float aux_vis_lambda,                       // 0=禁用（反向 bidir 传 0）
    const float* __restrict__ d_aux_gate,       // [N] g_i 唤醒权重 (nullptr=禁用)
    // === Stacking Penalty Loss (SP-Loss) 注入（sp_lambda>0 且 d_k_allowed!=nullptr 时启用）===
    // L_sp = (λ/N)·Σ_p max(0, W(p)-K_allowed(p))², W=S-C, S=Σae, C=1-Π(1-ae)
    // ∂L_sp/∂ae_i = (2λ/N)·max(0,W-K)·(1-Π_{j≠i}(1-ae_j)) = (2λ/N)·max(0,W-K)·(1-(1-C)/(1-ae_i))
    // S/C 在 Pass1 局部累加（与 forward 同序，float 一致）；K_allowed 复杂度自适应（预计算上传）。
    // 注入到 dlae，自动流回 cx/cy/rx/ry/angle/opacity/logits（不改渲染语义，不影响 colors）。
    float sp_lambda,                            // SP-Loss 权重 (0=禁用，反向 bidir 传 0)
    const float* __restrict__ d_k_allowed,      // [total_pixels] K_allowed(p) (nullptr=禁用)
    bool sp_geo_loss_active,                    // true=几何覆盖损失模式（禁用旧 ae-based SP-Loss）
    // === R23: 确定性归约参数 ===
    bool deterministic,
    float* __restrict__ grad_partials,
    float* __restrict__ grad_logits_partials)
{
    int tile_idx = batch_tile_start + blockIdx.x;
    if (tile_idx >= batch_tile_start + batch_tile_count) return;

    int tile_x = tile_idx % num_tiles_x;
    int tile_y = tile_idx / num_tiles_x;

    // block 内 thread 映射到 tile 内像素（tile_size=16, block=256=16*16）
    int intra_x = threadIdx.x % tile_size;
    int intra_y = threadIdx.x / tile_size;
    int global_x = tile_x * tile_size + intra_x;
    int global_y = tile_y * tile_size + intra_y;

    // 边界 check：不 early return（保证 reduction 的 __syncthreads 安全）
    bool valid_pixel = (global_x < W && global_y < H);
    int idx = valid_pixel ? (global_y * W + global_x) : 0;

    // 读取输入（仅 valid pixel）
    float pxi = 0.0f, pyi = 0.0f;
    float gdr = 0.0f, gdg = 0.0f, gdb = 0.0f, gda = 0.0f;
    // === FA-Loss 预读取：从 d_fa_globals 读标量 + 像素级 mask 与目标颜色 ===
    // dL_fa/d_ar_k(p) = fa_scale · (||I(p)-μ_k||² - fa_mean_t) · mask(p)
    // μ_k 在形状循环内按 i 读取；fa_scale/fa_mean_t 是 kernel 级常量（所有 thread 同值）。
    float fa_scale = 0.0f;
    float fa_mean_t = 0.0f;
    float fa_vis_weight = 1.0f;   // 默认 VW 开（向后兼容）
    if (d_fa_globals != nullptr) {
        fa_scale = d_fa_globals[0];
        fa_mean_t = d_fa_globals[1];
        fa_vis_weight = d_fa_globals[2];
    }
    float fa_mask = 0.0f;
    float fa_Ir = 0.0f, fa_Ig = 0.0f, fa_Ib = 0.0f;
    if (valid_pixel) {
        pxi = px[idx]; pyi = py[idx];
        gdr = grad_output[idx * 3 + 0];
        gdg = grad_output[idx * 3 + 1];
        gdb = grad_output[idx * 3 + 2];
        gda = (grad_alpha) ? grad_alpha[idx] : 0.0f;
        // ── sRGB 链式法则修正（只在反向） ─────────────────────────────────
        // 前向：颜色经 srgb_to_linear 后**在线性空间**合成，最后 linear_to_srgb 编码回 sRGB
        //       （本文件 1026-1039）。而 grad_output 是 dL/d(输出 sRGB)，下面的合成循环却在
        //       线性空间做微分 ⇒ 必须补上 d(linear)/d(srgb) = 1 / (d(srgb)/d(linear))。
        // 修正前反向微分的是"直接在线性值上合成"的另一个函数：方向不至反，但每个像素的
        // 缩放系数是错的（暗部应放大却被压缩），实测暗带误差 75 vs 中带 12。
        // alpha 不在此列：前向 d_rendered_alpha = total_alpha * 255 是线性的，无需雅可比。
        if (srgb_out != nullptr) {
            gdr /= srgb_encode_deriv(srgb_out[idx * 3 + 0]);
            gdg /= srgb_encode_deriv(srgb_out[idx * 3 + 1]);
            gdb /= srgb_encode_deriv(srgb_out[idx * 3 + 2]);
        }
        if (fa_scale > 0.0f && d_mask != nullptr && d_target != nullptr) {
            fa_mask = d_mask[idx];
            fa_Ir = d_target[idx * 3 + 0];
            fa_Ig = d_target[idx * 3 + 1];
            fa_Ib = d_target[idx * 3 + 2];
        }
    }

    int wid = threadIdx.x >> 5;
    int lid = threadIdx.x & 31;
    int num_warps = blockDim.x >> 5;
    __shared__ float s_block[9 * 32];
    // R23: 确定性 grad_logits 归约用的 shared memory（跨 warp reduction）
    // 布局: [warp_id * MAX_TEMPLATE_TYPES + t]，每个 warp 写 num_types 个值
    __shared__ float s_gl[MAX_TEMPLATE_TYPES * 32];

    // P3-1: tile 的 shape list（O(k)），替代旧 O(N) 遍历
    int count = min(d_tile_shape_counts[tile_idx], max_shapes_per_tile);
    const int* shape_list = d_tile_shape_lists + (size_t)tile_idx * max_shapes_per_tile;

    // === Soft depth: 按 z 降序排序 shape list（z大=底层=先渲染） ===
    // 与 forward 的 use_z_sort 逻辑保持一致：只要 d_z 存在且 count <= SM_BW 就排序。
    // tau_z>0: 软遮挡 backward 需要 z 排序以匹配 forward 软遮挡逻辑。
    // tau_z=0 (hard steps): forward 走 z-sorted over compositing，backward 也必须用相同顺序。
    const int SM_BW = 96;  // 与 forward 一致
    int sorted_list[SM_BW];
    const int* active_list = shape_list;
    bool use_z_sort = (d_z != nullptr && count > 1 && count <= SM_BW);
    if (use_z_sort) {
        // 复制 shape indices 并按 z 降序排序（插入排序）
        for (int k = 0; k < count; k++) sorted_list[k] = shape_list[k];
        for (int k = 1; k < count; k++) {
            int key_idx = sorted_list[k];
            float key_z = d_z[key_idx];
            int j = k - 1;
            while (j >= 0 && d_z[sorted_list[j]] < key_z) {
                sorted_list[j + 1] = sorted_list[j];
                j--;
            }
            sorted_list[j + 1] = key_idx;
        }
        active_list = sorted_list;
    }

    // ========== Soft depth 预计算：局部化（方案 B）—— 只收集覆盖当前像素的形状 ==========
    // 旧版受 count <= soft_depth_max_m 限制：tile 内 AABB 相交形状数 > 64 时 z 梯度被
    // 完全跳过（即使该像素只被少数形状覆盖），导致 STE 在密集区域失效（-0.027 dB）。
    // 现改为遍历 tile 内所有 count 个形状，只收集 ae>0（覆盖当前像素）的到 cover 数组。
    // cover_count 通常 << 64（一个像素很少被 >64 形状覆盖）。T_i 与 z 梯度都在 cover 数组上
    // 计算——z 梯度只在空间真正重叠的形状对间流动，不再受 tile 总形状数限制。
    // 避免 tau_z=0 时 inv_tau=inf 产生 NaN（当 z_i==z_j 时 diff=0*inf=NaN）。
    const int SM_SD = 64;
    float ae_arr_sd[SM_SD], zi_arr_sd[SM_SD], Ti_arr_sd[SM_SD];
    float clr_arr_sd[SM_SD], clg_arr_sd[SM_SD], clb_arr_sd[SM_SD];
    int cover_shape_id[SM_SD];   // cover 形状的 global shape id（用于 grad_z 索引）
    int cover_count = 0;

    // count <= soft_depth_max_m：实验证明 count > 64 的密集 tile 上 z 梯度有害
    //（STE 软遮挡在密集区域梯度方向不一致）。avg tile count=82.7 > 64，全局版本
    // 在大多数 tile 上跳过 z 梯度——这是 STE 在此场景下几乎无效的根本原因。
    // cover 数组实现（只收集 ae>0 的形状）与全局版本数学等价，但代码更清晰。
    bool use_sd_backward = (d_z != nullptr && tau_z > 0.0f && count > 1 && count <= soft_depth_max_m);
    const int MAX_COVER = 32;  // 仅用于 pass3_iters 上界（count<=64 时 pass3_iters=min(count,32)）
    if (use_sd_backward) {
        // SD-a: 遍历 tile 内所有形状，只收集覆盖当前像素 (ae > 0) 的
        for (int k = 0; k < count && cover_count < MAX_COVER; k++) {
            int i = active_list[k];
            float coi = cosa[i], sii = sina[i];
            float rxc = fmaxf(rx[i], 0.5f), ryc = fmaxf(ry[i], 0.5f);
            int stype_aabb = (shape_types[i] < 0) ? 0 : to_template_idx(shape_types[i], num_builtins);
            float ar = 0.0f;
            if (valid_pixel && shape_aabb_check(pxi, pyi, cx[i], cy[i], rxc, ryc,
                                                coi, sii, stype_aabb, blur_sigma, num_builtins)) {
                float dx = pxi - cx[i], dy = pyi - cy[i];
                float u_rot = dx * coi + dy * sii;
                float v_rot = -dx * sii + dy * coi;
                if (shape_types[i] < 0) {
                    for (int t = 0; t < num_types; t++) {
                        float wt = softmax_w[i * num_types + t];
                        if (wt < 0.02f) continue;
                        float su, sv;
                        get_template_scale(t, rxc, ryc, &su, &sv, num_builtins);
                        float u = u_rot / su, v = v_rot / sv;
                        ar += wt * bilinear_sample_template(
                            d_blurred_templates + t * TEMPLATE_SIZE * TEMPLATE_SIZE, u, v);
                    }
                } else {
                    int stype = to_template_idx(shape_types[i], num_builtins);
                    float su, sv;
                    get_template_scale(stype, rxc, ryc, &su, &sv, num_builtins);
                    float u = u_rot / su, v = v_rot / sv;
                    ar = bilinear_sample_template(
                        d_blurred_templates + stype * TEMPLATE_SIZE * TEMPLATE_SIZE, u, v);
                }
            }
            float opi = sigmoid_logit(opacity[i] / g_opacity_tau);
            float ae = ar * opi * g_alpha_upper_bound;
            if (ae <= 0.0f) continue;  // 不覆盖当前像素，跳过（局部化核心）
            ae_arr_sd[cover_count] = ae;
            zi_arr_sd[cover_count] = d_z[i];
            clr_arr_sd[cover_count] = 255.0f * sigmoid_logit(colors[i * 3 + 0]);
            clg_arr_sd[cover_count] = 255.0f * sigmoid_logit(colors[i * 3 + 1]);
            clb_arr_sd[cover_count] = 255.0f * sigmoid_logit(colors[i * 3 + 2]);
            cover_shape_id[cover_count] = i;
            cover_count++;
        }
        // SD-b: 计算软遮挡 T_i = ∏_{j≠i} (1 - ae_j * p_ji), log-space（只在 cover 形状间）
        // p_ji = sigmoid((z_i - z_j)/tau) = P(j 是 i 的顶层, j 遮挡 i)
        if (cover_count > 1) {
            float inv_tau_sd = 1.0f / tau_z;
            for (int k = 0; k < cover_count; k++) {
                double log_T = 0.0;
                for (int j = 0; j < cover_count; j++) {
                    if (j == k) continue;
                    float diff = (zi_arr_sd[k] - zi_arr_sd[j]) * inv_tau_sd;
                    diff = fmaxf(-20.0f, fminf(20.0f, diff));
                    float p = 1.0f / (1.0f + expf(-diff));
                    float om = 1.0f - ae_arr_sd[j] * p;
                    if (om < 1e-15f) om = 1e-15f;
                    log_T += log((double)om);
                }
                float Tk = expf((float)log_T);
                if (Tk > 1.0f) Tk = 1.0f;
                Ti_arr_sd[k] = Tk;
            }
        }
    }

    // ========== Pass 1: 局部计算 lTN（遍历 tile 内 k 个 shape） ==========
    // v2 修正：lTN 不再读取全局 d_lTN，改为局部计算，与 lTB 基于同一组 shape
    double lTN = 0.0;
    // SP-Loss: Pass 1 同步累加 S=Σae 与 C=总覆盖（与 forward 同序同型，float 一致）
    // sp_lambda=0 时累加器不参与任何输出，回归 baseline 行为逐位一致。
    float sp_sum_ae = 0.0f;
    float sp_sum_ar = 0.0f;
    float sp_total_alpha = 0.0f;
    bool sp_active = (sp_lambda > 0.0f && d_k_allowed != nullptr && !sp_geo_loss_active);
    bool sp_geo_active = (sp_lambda > 0.0f && d_k_allowed != nullptr && sp_geo_loss_active);
    if (reverse) {
        for (int k = count - 1; k >= 0; k--) {
            int i = active_list[k];
            if (!valid_pixel) continue;

            float coi = cosa[i], sii = sina[i];
            float rxc = fmaxf(rx[i], 0.5f), ryc = fmaxf(ry[i], 0.5f);
            int stype_for_aabb = (shape_types[i] < 0) ? 0 : to_template_idx(shape_types[i], num_builtins);

            if (!shape_aabb_check(pxi, pyi, cx[i], cy[i], rxc, ryc, coi, sii, stype_for_aabb, blur_sigma, num_builtins))
                continue;

            float dx = pxi - cx[i], dy = pyi - cy[i];
            float u_rot = dx * coi + dy * sii;
            float v_rot = -dx * sii + dy * coi;
            float ar = 0.0f;

            if (shape_types[i] < 0) {
                for (int t = 0; t < num_types; t++) {
                    float wt = softmax_w[i * num_types + t];
                    if (wt < 0.02f) continue;
                    float su, sv;
                    get_template_scale(t, rxc, ryc, &su, &sv, num_builtins);
                    float u = u_rot / su, v = v_rot / sv;
                    const float* tmpl = d_blurred_templates + t * TEMPLATE_SIZE * TEMPLATE_SIZE;
                    ar += wt * bilinear_sample_template(tmpl, u, v);
                }
            } else {
                int stype = to_template_idx(shape_types[i], num_builtins);
                float su, sv;
                get_template_scale(stype, rxc, ryc, &su, &sv, num_builtins);
                float u = u_rot / su, v = v_rot / sv;
                const float* tmpl = d_blurred_templates + stype * TEMPLATE_SIZE * TEMPLATE_SIZE;
                ar = bilinear_sample_template(tmpl, u, v);
            }

            if (ar > 0.0f) {
                // Logit parameterization: opacity stored as logit, convert to linear
                float opi = sigmoid_logit(opacity[i] / g_opacity_tau);
                float ae = ar * opi * g_alpha_upper_bound;
                double oma = 1.0 - (double)ae;
                if (oma < 1e-15) oma = 1e-15;
                lTN += log(oma);
                if (sp_active) {
                    sp_sum_ae += ae;
                    sp_total_alpha = ae + (1.0f - ae) * sp_total_alpha;
                }
                if (sp_geo_active) sp_sum_ar += ar;
            }
        }
    } else {
        for (int k = 0; k < count; k++) {
            int i = active_list[k];
            if (!valid_pixel) continue;

            float coi = cosa[i], sii = sina[i];
            float rxc = fmaxf(rx[i], 0.5f), ryc = fmaxf(ry[i], 0.5f);
            int stype_for_aabb = (shape_types[i] < 0) ? 0 : to_template_idx(shape_types[i], num_builtins);

            if (!shape_aabb_check(pxi, pyi, cx[i], cy[i], rxc, ryc, coi, sii, stype_for_aabb, blur_sigma, num_builtins))
                continue;

            float dx = pxi - cx[i], dy = pyi - cy[i];
            float u_rot = dx * coi + dy * sii;
            float v_rot = -dx * sii + dy * coi;
            float ar = 0.0f;

            if (shape_types[i] < 0) {
                for (int t = 0; t < num_types; t++) {
                    float wt = softmax_w[i * num_types + t];
                    if (wt < 0.02f) continue;
                    float su, sv;
                    get_template_scale(t, rxc, ryc, &su, &sv, num_builtins);
                    float u = u_rot / su, v = v_rot / sv;
                    const float* tmpl = d_blurred_templates + t * TEMPLATE_SIZE * TEMPLATE_SIZE;
                    ar += wt * bilinear_sample_template(tmpl, u, v);
                }
            } else {
                int stype = to_template_idx(shape_types[i], num_builtins);
                float su, sv;
                get_template_scale(stype, rxc, ryc, &su, &sv, num_builtins);
                float u = u_rot / su, v = v_rot / sv;
                const float* tmpl = d_blurred_templates + stype * TEMPLATE_SIZE * TEMPLATE_SIZE;
                ar = bilinear_sample_template(tmpl, u, v);
            }

            if (ar > 0.0f) {
                // Logit parameterization: opacity stored as logit, convert to linear
                float opi = sigmoid_logit(opacity[i] / g_opacity_tau);
                float ae = ar * opi * g_alpha_upper_bound;
                double oma = 1.0 - (double)ae;
                if (oma < 1e-15) oma = 1e-15;
                lTN += log(oma);
                if (sp_active) {
                    sp_sum_ae += ae;
                    sp_total_alpha = ae + (1.0f - ae) * sp_total_alpha;
                }
                if (sp_geo_active) sp_sum_ar += ar;
            }
        }
    }

    // ========== Pass 2: Compute gradients per shape ==========
    double lTB = 0.0;
    float car = bg_color[0], cag = bg_color[1], cab = bg_color[2];

    // YMP (Yield Message Passing): vis_below_acc 累加已处理形状（当前 k 之下）的潜在能力
    //   vis_full_i = opi_i · π · rx_i · ry_i （解析量，不因遮挡衰减）。
    // 非反向分支（k=0→count-1，从底到顶）处理形状 k 时，vis_below_acc = Σ_{i<k, i 覆盖本像素} vis_full_i
    //   即 k 遮挡的所有下层形状的能力总和（让位压力）。反向分支 yield_lambda=0，不使用此累加器。
    // 与旧 yield loss 的唯一区别：累加量从 ae·T（实际贡献，被遮挡→衰减为0）改为 vis_full（潜在能力，不衰减）。
    double vis_below_acc = 0.0;
    // yield 归一化因子：L_yield = λ · (1/N) · Σ_p Σ_i ae_i·(1-T_i)，∂L/∂ae_k 每像素贡献 λ/N
    // 用 1/N 归一化（而非 1/(W·H)）使 λ 语义为"每形状让位成本权重"，量级与 MSE 可比
    float yield_norm = (yield_lambda > 0.0f && N > 0)
                       ? (yield_lambda / (float)N)
                       : 0.0f;

    // SP-Loss 归一化系数：L_sp = (λ/N_pixels)·Σ_p max(0,W-K)²，与 MSE（逐像素均值）同口径
    // 使 λ 跨分辨率可解释；W*H=total_pixels。
    float sp_norm = ((sp_active || sp_geo_active) && W > 0 && H > 0)
                    ? (2.0f * sp_lambda / (float)(W * H))
                    : 0.0f;

    // EM-YMP (Error-Modulated YMP): 把 vis_below_acc 的累加量乘以局部误差归一化
    //   err_norm = err(p) / mean_err。高误差区(>1)放大下层让位压力，低误差区(<1)衰减，
    //   使上层只为"真正没拟合好"的位置让位，而非为无用的下层形状牺牲 PSNR。
    // error_map/d_mean_err 为 nullptr（YMP 对照或反向 bidir）时 err_norm=1，退化为原 YMP。
    float err_norm = 1.0f;
    if (yield_norm > 0.0f && error_map != nullptr && d_mean_err != nullptr) {
        float err_p = valid_pixel ? error_map[idx] : 0.0f;
        float mean_err = *d_mean_err;
        err_norm = err_p / fmaxf(mean_err, 1e-6f);
        if (err_norm > 10.0f) err_norm = 10.0f;   // clip 防个别高误差像素放大过头
    }

    if (reverse) {
        for (int k = count - 1; k >= 0; k--) {
            int i = active_list[k];

            float lcx = 0, lcy = 0, lrx = 0, lry = 0, lan = 0, lcr = 0, lcg = 0, lcb = 0, lop = 0;
            float lgl[MAX_TEMPLATE_TYPES] = {0};  // R23: 确定性 grad_logits 累加器

            if (valid_pixel) {
                float coi = cosa[i], sii = sina[i];
                float rxc = fmaxf(rx[i], 0.5f), ryc = fmaxf(ry[i], 0.5f);

                int stype_for_aabb = (shape_types[i] < 0) ? 0 : to_template_idx(shape_types[i], num_builtins);
                bool in_aabb = shape_aabb_check(pxi, pyi, cx[i], cy[i], rxc, ryc, coi, sii, stype_for_aabb, blur_sigma, num_builtins);

                if (in_aabb) {
                    float dx = pxi - cx[i], dy = pyi - cy[i];
                    float u_rot = dx * coi + dy * sii;
                    float v_rot = -dx * sii + dy * coi;
                    float ar = 0.0f;

                    if (shape_types[i] < 0) {
                        float sampled[MAX_TEMPLATE_TYPES];
                        for (int t = 0; t < num_types; t++) {
                            float wt = softmax_w[i * num_types + t];
                            if (wt < 0.02f) { sampled[t] = 0.0f; continue; }
                            float su, sv;
                            get_template_scale(t, rxc, ryc, &su, &sv, num_builtins);
                            float u = u_rot / su, v = v_rot / sv;
                            const float* tmpl = d_blurred_templates + t * TEMPLATE_SIZE * TEMPLATE_SIZE;
                            sampled[t] = bilinear_sample_template(tmpl, u, v);
                            ar += wt * sampled[t];
                        }

                        if (ar > 0.0f) {
                            // Logit parameterization: opacity/colors stored as logits
                            float opi = sigmoid_logit(opacity[i] / g_opacity_tau);
                            float clr = 255.0f * sigmoid_logit(colors[i * 3 + 0]);
                            float clg = 255.0f * sigmoid_logit(colors[i * 3 + 1]);
                            float clb = 255.0f * sigmoid_logit(colors[i * 3 + 2]);
                            float ae = ar * opi * g_alpha_upper_bound;
                            double oma = 1.0 - (double)ae;
                            if (oma < 1e-15) oma = 1e-15;
                            float omaf = (float)oma;
                            double lT_i = lTN - lTB - log(oma);
                            float Ti = exp((float)lT_i);
                            if (Ti > 1.0f) Ti = 1.0f;

                            lcr = gdr * ae * Ti; lcg = gdg * ae * Ti; lcb = gdb * ae * Ti;

                            // Quadrant RGB gradient accumulation (split analysis)
                            if (output_quadrant && grad_rgb_quadrant != nullptr) {
                                int q = (u_rot >= 0.0f ? 1 : 0) | (v_rot >= 0.0f ? 2 : 0);
                                atomicAdd(&grad_rgb_quadrant[((long long)i * 4 + q) * 3 + 0], lcr);
                                atomicAdd(&grad_rgb_quadrant[((long long)i * 4 + q) * 3 + 1], lcg);
                                atomicAdd(&grad_rgb_quadrant[((long long)i * 4 + q) * 3 + 2], lcb);
                            }

                            if (Ti > 1e-6f) {
                                float dcdar = Ti * (clr - car), dcdag = Ti * (clg - cag), dcdab = Ti * (clb - cab);
                                float dlae = gdr * dcdar + gdg * dcdag + gdb * dcdab + gda * 255.0f * Ti;
                                // Yield gradient: ∂L_yield/∂ae_k = yield_norm · [(1-T_k) + vis_below_acc/(1-ae_k)]
                                // 第一项 (1-T_k): j 扩大但 vis 没相应增加的惩罚（被遮挡时大）
                                // 第二项 vis_below_acc/(1-ae_k): j 遮挡下层形状 vis 的惩罚（顶层大）
                                if (yield_norm > 0.0f) dlae += yield_norm * (-Ti + (float)vis_below_acc / fmaxf(omaf, 1e-6f));
                                // Aux-Vis: ∂L_aux/∂ae_i = -λ·g_i·T_i（g_i 常数权重，stop-gradient）。
                                // 低 vis 形状 g_i 大→朝"上层透过率 T_i 高"方向移动（dlae<0 使 ae 增大→vis 上升）。
                                // 反向 bidir 调用时 aux_vis_lambda=0，不注入（与 yield 反向传 0 同理）。
                                if (aux_vis_lambda > 0.0f && d_aux_gate != nullptr) dlae += -aux_vis_lambda * d_aux_gate[i] * Ti;
                                // FA-Loss 梯度（通过 ar 链流回 cx/cy/rx/ry/angle/logits，不流回 opacity/color）
                                // VW-FA: dL_fa/d_ar_k(p) = fa_scale · (||I(p)-μ_k||² - fa_mean_t) · mask(p) · lerp(1, T_k(p), vis_w)
                                // T_k(p)=Ti 已在上方算出（上层残余透过率）；vis_w=1 时被遮挡 Ti≈0→不误伤，vis_w=0 退化为原 FA。
                                float fa_grad_ar = 0.0f;
                                if (fa_scale > 0.0f && fa_mask > 0.5f && d_fa_mu != nullptr) {
                                    float mu_kr = d_fa_mu[i * 3 + 0];
                                    float mu_kg = d_fa_mu[i * 3 + 1];
                                    float mu_kb = d_fa_mu[i * 3 + 2];
                                    float fdr = fa_Ir - mu_kr, fdg = fa_Ig - mu_kg, fdb = fa_Ib - mu_kb;
                                    float dist_sq = fdr * fdr + fdg * fdg + fdb * fdb;
                                    float vw = 1.0f + (Ti - 1.0f) * fa_vis_weight;  // lerp(1, Ti, vis_w)
                                    fa_grad_ar = fa_scale * (dist_sq - fa_mean_t) * fa_mask * vw;
                                }
                                // SP-Loss 梯度注入：∂L_sp/∂ae_i = sp_norm · max(0, W-K) · (1 - (1-C)/(1-ae_i))
                                // W=sp_sum_ae-sp_total_alpha（Pass1 累加），K=d_k_allowed[idx]
                                // (1-(1-C)/(1-ae_i)) = 1-Π_{j≠i}(1-ae_j) = 其余形状总覆盖（clip 到 [0,1] 防数值漂移）
                                if (sp_active) {
                                    float W = sp_sum_ae - sp_total_alpha;
                                    float K = d_k_allowed[idx];
                                    float W_minus_K = W - K;
                                    if (W_minus_K > 0.0f) {
                                        float one_minus_ae = fmaxf(1.0f - ae, 1e-6f);
                                        // ∂W/∂ae_i = 1 - Π_{j≠i}(1-ae_j) = 1 - (1-C)/(1-ae_i)
                                        // 注意：必须用 1 - (...)/(1-ae_i)，旧版漏了 "1 -" 导致梯度错误
                                        float T_without_i = 1.0f - (1.0f - sp_total_alpha) / one_minus_ae;
                                        T_without_i = fmaxf(0.0f, fminf(1.0f, T_without_i));
                                        dlae += sp_norm * W_minus_K * T_without_i;
                                    }
                                }
                                // SP-GEO-Loss: 几何覆盖层数梯度，直接注入 ar 链（不进 dlae，不被 opacity 衰减）
                                float geo_grad_ar = 0.0f;
                                if (sp_geo_active) {
                                    float G = sp_sum_ar;
                                    float K = d_k_allowed[idx];
                                    float G_minus_K = G - K;
                                    if (G_minus_K > 0.0f) {
                                        geo_grad_ar = sp_norm * G_minus_K;   // (2λ/N)·max(0, G-K)
                                    }
                                }

                                for (int t = 0; t < num_types; t++) {
                                    float wt = softmax_w[i * num_types + t];
                                    if (wt < 0.02f) continue;
                                    float su, sv;
                                    get_template_scale(t, rxc, ryc, &su, &sv, num_builtins);
                                    float u = u_rot / su, v = v_rot / sv;
                                    const float* tmpl = d_blurred_templates + t * TEMPLATE_SIZE * TEMPLATE_SIZE;

                                    float dadu, dadv;
                                    bilinear_sample_gradient(tmpl, u, v, &dadu, &dadv);

                                    float dadcx = dadu * (-coi / su) + dadv * (sii / sv);
                                    float dadcy = dadu * (-sii / su) + dadv * (-coi / sv);
                                    float dadrx = dadu * (-u / rxc);
                                    float dadry = dadv * (-v / ryc);
                                    float dadan = dadu * (v_rot / su * PI_F / 180.0f)
                                                + dadv * (-u_rot / sv * PI_F / 180.0f);

                                    lcx += (dlae * opi + fa_grad_ar + geo_grad_ar) * wt * dadcx;
                                    lcy += (dlae * opi + fa_grad_ar + geo_grad_ar) * wt * dadcy;
                                    lrx += (dlae * opi + fa_grad_ar + geo_grad_ar) * wt * dadrx;
                                    lry += (dlae * opi + fa_grad_ar + geo_grad_ar) * wt * dadry;
                                    lan += (dlae * opi + fa_grad_ar + geo_grad_ar) * wt * dadan;
                                }
                                lop = dlae * ar;

                                // DARTS logit gradient (generalized loop)
                                // 注意：不乘 1/T 温度因子。实验验证（E52 vs E46 on 105053）：
                                //   有 1/T: 28.57 dB
                                //   无 1/T: 29.22 dB (+0.65 dB)
                                // Adam 会自动归一化梯度幅度，温度只影响 softmax 分布尖锐度，
                                // 不需要梯度补偿。1/T 反而破坏了探索-利用平衡。
                                for (int t = 0; t < num_types; t++) {
                                    float gl_t = (dlae * opi + fa_grad_ar + geo_grad_ar) * softmax_w[i * num_types + t] * (sampled[t] - ar);
                                    if (deterministic) {
                                        lgl[t] += gl_t;
                                    } else {
                                        atomicAdd(&grad_logits[i * num_types + t], gl_t);
                                    }
                                }
                            }

                            car = ae * clr + omaf * car;
                            cag = ae * clg + omaf * cag;
                            cab = ae * clb + omaf * cab;
                            lTB += log(oma);
                            // YMP: 广播当前形状的潜在能力 vis_full=opi·π·rx·ry 到 vis_below_acc（不衰减，供上层 k 让位）
                            if (yield_norm > 0.0f) vis_below_acc += (double)opi * (double)PI_F * (double)rxc * (double)ryc * (double)err_norm;
                        }
                    } else {
                        int stype = to_template_idx(shape_types[i], num_builtins);
                        float su, sv;
                        get_template_scale(stype, rxc, ryc, &su, &sv, num_builtins);
                        float u = u_rot / su, v = v_rot / sv;
                        const float* tmpl = d_blurred_templates + stype * TEMPLATE_SIZE * TEMPLATE_SIZE;
                        ar = bilinear_sample_template(tmpl, u, v);

                        if (ar > 0.0f) {
                            // Logit parameterization: opacity/colors stored as logits
                            float opi = sigmoid_logit(opacity[i] / g_opacity_tau);
                            float clr = 255.0f * sigmoid_logit(colors[i * 3 + 0]);
                            float clg = 255.0f * sigmoid_logit(colors[i * 3 + 1]);
                            float clb = 255.0f * sigmoid_logit(colors[i * 3 + 2]);
                            float ae = ar * opi * g_alpha_upper_bound;
                            double oma = 1.0 - (double)ae;
                            if (oma < 1e-15) oma = 1e-15;
                            float omaf = (float)oma;
                            double lT_i = lTN - lTB - log(oma);
                            float Ti = exp((float)lT_i);
                            if (Ti > 1.0f) Ti = 1.0f;

                            lcr = gdr * ae * Ti; lcg = gdg * ae * Ti; lcb = gdb * ae * Ti;

                            // Quadrant RGB gradient accumulation (split analysis)
                            if (output_quadrant && grad_rgb_quadrant != nullptr) {
                                int q = (u_rot >= 0.0f ? 1 : 0) | (v_rot >= 0.0f ? 2 : 0);
                                atomicAdd(&grad_rgb_quadrant[((long long)i * 4 + q) * 3 + 0], lcr);
                                atomicAdd(&grad_rgb_quadrant[((long long)i * 4 + q) * 3 + 1], lcg);
                                atomicAdd(&grad_rgb_quadrant[((long long)i * 4 + q) * 3 + 2], lcb);
                            }

                            if (Ti > 1e-6f) {
                                float dcdar = Ti * (clr - car), dcdag = Ti * (clg - cag), dcdab = Ti * (clb - cab);
                                float dlae = gdr * dcdar + gdg * dcdag + gdb * dcdab + gda * 255.0f * Ti;
                                // Yield gradient: ∂L_yield/∂ae_k = yield_norm · [(1-T_k) + vis_below_acc/(1-ae_k)]
                                // 第一项 (1-T_k): j 扩大但 vis 没相应增加的惩罚（被遮挡时大）
                                // 第二项 vis_below_acc/(1-ae_k): j 遮挡下层形状 vis 的惩罚（顶层大）
                                if (yield_norm > 0.0f) dlae += yield_norm * (-Ti + (float)vis_below_acc / fmaxf(omaf, 1e-6f));
                                // Aux-Vis: ∂L_aux/∂ae_i = -λ·g_i·T_i（g_i 常数权重，stop-gradient）。
                                // 低 vis 形状 g_i 大→朝"上层透过率 T_i 高"方向移动（dlae<0 使 ae 增大→vis 上升）。
                                // 反向 bidir 调用时 aux_vis_lambda=0，不注入（与 yield 反向传 0 同理）。
                                if (aux_vis_lambda > 0.0f && d_aux_gate != nullptr) dlae += -aux_vis_lambda * d_aux_gate[i] * Ti;
                                // FA-Loss 梯度（通过 ar 链流回 cx/cy/rx/ry/angle/logits，不流回 opacity/color）
                                // VW-FA: dL_fa/d_ar_k(p) = fa_scale · (||I(p)-μ_k||² - fa_mean_t) · mask(p) · lerp(1, T_k(p), vis_w)
                                // T_k(p)=Ti 已在上方算出（上层残余透过率）；vis_w=1 时被遮挡 Ti≈0→不误伤，vis_w=0 退化为原 FA。
                                float fa_grad_ar = 0.0f;
                                if (fa_scale > 0.0f && fa_mask > 0.5f && d_fa_mu != nullptr) {
                                    float mu_kr = d_fa_mu[i * 3 + 0];
                                    float mu_kg = d_fa_mu[i * 3 + 1];
                                    float mu_kb = d_fa_mu[i * 3 + 2];
                                    float fdr = fa_Ir - mu_kr, fdg = fa_Ig - mu_kg, fdb = fa_Ib - mu_kb;
                                    float dist_sq = fdr * fdr + fdg * fdg + fdb * fdb;
                                    float vw = 1.0f + (Ti - 1.0f) * fa_vis_weight;  // lerp(1, Ti, vis_w)
                                    fa_grad_ar = fa_scale * (dist_sq - fa_mean_t) * fa_mask * vw;
                                }
                                // SP-Loss 梯度注入（locked 分支）：同 DARTS 分支公式
                                if (sp_active) {
                                    float W = sp_sum_ae - sp_total_alpha;
                                    float K = d_k_allowed[idx];
                                    float W_minus_K = W - K;
                                    if (W_minus_K > 0.0f) {
                                        float one_minus_ae = fmaxf(1.0f - ae, 1e-6f);
                                        // ∂W/∂ae_i = 1 - Π_{j≠i}(1-ae_j) = 1 - (1-C)/(1-ae_i)
                                        // 注意：必须用 1 - (...)/(1-ae_i)，旧版漏了 "1 -" 导致梯度错误
                                        float T_without_i = 1.0f - (1.0f - sp_total_alpha) / one_minus_ae;
                                        T_without_i = fmaxf(0.0f, fminf(1.0f, T_without_i));
                                        dlae += sp_norm * W_minus_K * T_without_i;
                                    }
                                }
                                // SP-GEO-Loss: 几何覆盖层数梯度，直接注入 ar 链（不进 dlae，不被 opacity 衰减）
                                float geo_grad_ar = 0.0f;
                                if (sp_geo_active) {
                                    float G = sp_sum_ar;
                                    float K = d_k_allowed[idx];
                                    float G_minus_K = G - K;
                                    if (G_minus_K > 0.0f) {
                                        geo_grad_ar = sp_norm * G_minus_K;   // (2λ/N)·max(0, G-K)
                                    }
                                }

                                float dadu, dadv;
                                bilinear_sample_gradient(tmpl, u, v, &dadu, &dadv);

                                float dadcx = dadu * (-coi / su) + dadv * (sii / sv);
                                float dadcy = dadu * (-sii / su) + dadv * (-coi / sv);
                                float dadrx = dadu * (-u / rxc);
                                float dadry = dadv * (-v / ryc);
                                float dadan = dadu * (v_rot / su * PI_F / 180.0f)
                                            + dadv * (-u_rot / sv * PI_F / 180.0f);

                                lcx = (dlae * opi + fa_grad_ar + geo_grad_ar) * dadcx;
                                lcy = (dlae * opi + fa_grad_ar + geo_grad_ar) * dadcy;
                                lrx = (dlae * opi + fa_grad_ar + geo_grad_ar) * dadrx;
                                lry = (dlae * opi + fa_grad_ar + geo_grad_ar) * dadry;
                                lan = (dlae * opi + fa_grad_ar + geo_grad_ar) * dadan;
                                lop = dlae * ar;

                                float sampled_all[MAX_TEMPLATE_TYPES];
                                #pragma unroll
                                for (int t = 0; t < num_types; t++) {
                                    float su_t, sv_t;
                                    get_template_scale(t, rxc, ryc, &su_t, &sv_t, num_builtins);
                                    float u_t = u_rot / su_t, v_t = v_rot / sv_t;
                                    const float* tmpl_t = d_blurred_templates + t * TEMPLATE_SIZE * TEMPLATE_SIZE;
                                    sampled_all[t] = bilinear_sample_template(tmpl_t, u_t, v_t);
                                }
                                // 锁定分支仍计算 grad_logits（退出机制），不乘 1/T（与 DARTS 分支一致）
                                for (int t = 0; t < num_types; t++) {
                                    float gl_t = (dlae * opi + fa_grad_ar + geo_grad_ar) * softmax_w[i * num_types + t] * (sampled_all[t] - ar);
                                    if (deterministic) {
                                        lgl[t] += gl_t;
                                    } else {
                                        atomicAdd(&grad_logits[i * num_types + t], gl_t);
                                    }
                                }
                            }

                            car = ae * clr + omaf * car;
                            cag = ae * clg + omaf * cag;
                            cab = ae * clb + omaf * cab;
                            lTB += log(oma);
                            // YMP: 广播当前形状的潜在能力 vis_full=opi·π·rx·ry 到 vis_below_acc（不衰减，供上层 k 让位）
                            if (yield_norm > 0.0f) vis_below_acc += (double)opi * (double)PI_F * (double)rxc * (double)ryc * (double)err_norm;
                        }
                    }
                }
            }

            // warp reduction（所有 thread 参与，valid_pixel=false 的贡献 0）
            unsigned warp_active = __activemask();
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                lcx += __shfl_down_sync(warp_active, lcx, off);
                lcy += __shfl_down_sync(warp_active, lcy, off);
                lrx += __shfl_down_sync(warp_active, lrx, off);
                lry += __shfl_down_sync(warp_active, lry, off);
                lan += __shfl_down_sync(warp_active, lan, off);
                lcr += __shfl_down_sync(warp_active, lcr, off);
                lcg += __shfl_down_sync(warp_active, lcg, off);
                lcb += __shfl_down_sync(warp_active, lcb, off);
                lop += __shfl_down_sync(warp_active, lop, off);
            }
            // R23: intra-warp reduction for grad_logits (deterministic only)
            if (deterministic) {
                for (int t = 0; t < num_types; t++)
                    for (int off = 16; off > 0; off >>= 1)
                        lgl[t] += __shfl_down_sync(warp_active, lgl[t], off);
            }
            if (lid == 0) {
                s_block[wid * 9 + 0] = lcx; s_block[wid * 9 + 1] = lcy;
                s_block[wid * 9 + 2] = lrx; s_block[wid * 9 + 3] = lry;
                s_block[wid * 9 + 4] = lan;
                s_block[wid * 9 + 5] = lcr; s_block[wid * 9 + 6] = lcg;
                s_block[wid * 9 + 7] = lcb; s_block[wid * 9 + 8] = lop;
                // R23: write lgl to shared memory for cross-warp reduction
                if (deterministic) {
                    for (int t = 0; t < num_types; t++)
                        s_gl[wid * MAX_TEMPLATE_TYPES + t] = lgl[t];
                }
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
                    // Chain rule: grad_logit = grad_linear * d_linear/d_logit
                    // For color: clr = 255*sigmoid(lc), so d_clr/d_lc = 255 * s_c * (1-s_c)
                    // For opacity: opi = sigmoid(lo), so d_opi/d_lo = s_o * (1-s_o)
                    float sc0 = sigmoid_logit(colors[i * 3 + 0]);
                    float sc1 = sigmoid_logit(colors[i * 3 + 1]);
                    float sc2 = sigmoid_logit(colors[i * 3 + 2]);
                    float so  = sigmoid_logit(opacity[i] / g_opacity_tau);
                    float g_cr = blcr * 255.0f * sigmoid_deriv_from_s(sc0);
                    float g_cg = blcg * 255.0f * sigmoid_deriv_from_s(sc1);
                    float g_cb = blcb * 255.0f * sigmoid_deriv_from_s(sc2);
                    float g_op = blop * sigmoid_deriv_from_s(so) / g_opacity_tau;
                    if (deterministic) {
                        // R23: 写 per-tile-per-shape partial，后续 reduce kernel 固定顺序归约
                        // 用 tile_idx（全局）而非 blockIdx.x（batch 内），避免 batch 间覆盖
                        int base = (tile_idx * N + i) * 9;
                        grad_partials[base + 0] = blcx; grad_partials[base + 1] = blcy;
                        grad_partials[base + 2] = blrx; grad_partials[base + 3] = blry;
                        grad_partials[base + 4] = blan;
                        grad_partials[base + 5] = g_cr;  grad_partials[base + 6] = g_cg;
                        grad_partials[base + 7] = g_cb;  grad_partials[base + 8] = g_op;
                    } else {
                        atomicAdd(&grad_cx[i], blcx); atomicAdd(&grad_cy[i], blcy);
                        atomicAdd(&grad_rx[i], blrx); atomicAdd(&grad_ry[i], blry);
                        atomicAdd(&grad_angle[i], blan);
                        atomicAdd(&grad_colors[i * 3 + 0], g_cr);
                        atomicAdd(&grad_colors[i * 3 + 1], g_cg);
                        atomicAdd(&grad_colors[i * 3 + 2], g_cb);
                        atomicAdd(&grad_opacity[i], g_op);
                    }
                }
                // R23: cross-warp reduction for grad_logits (deterministic only)
                if (deterministic) {
                    for (int t = 0; t < num_types; t++) {
                        float val = (lid < num_warps) ? s_gl[lid * MAX_TEMPLATE_TYPES + t] : 0.0f;
                        for (int off = 16; off > 0; off >>= 1)
                            val += __shfl_down_sync(bmask, val, off);
                        if (lid == 0)
                            grad_logits_partials[(tile_idx * N + i) * num_types + t] = val;
                    }
                }
            }
            __syncthreads();
        }
    } else {
        for (int k = 0; k < count; k++) {
            int i = active_list[k];

            float lcx = 0, lcy = 0, lrx = 0, lry = 0, lan = 0, lcr = 0, lcg = 0, lcb = 0, lop = 0;
            float lgl[MAX_TEMPLATE_TYPES] = {0};  // R23: 确定性 grad_logits 累加器

            if (valid_pixel) {
                float coi = cosa[i], sii = sina[i];
                float rxc = fmaxf(rx[i], 0.5f), ryc = fmaxf(ry[i], 0.5f);

                int stype_for_aabb = (shape_types[i] < 0) ? 0 : to_template_idx(shape_types[i], num_builtins);
                bool in_aabb = shape_aabb_check(pxi, pyi, cx[i], cy[i], rxc, ryc, coi, sii, stype_for_aabb, blur_sigma, num_builtins);

                if (in_aabb) {
                    float dx = pxi - cx[i], dy = pyi - cy[i];
                    float u_rot = dx * coi + dy * sii;
                    float v_rot = -dx * sii + dy * coi;
                    float ar = 0.0f;

                    if (shape_types[i] < 0) {
                        float sampled[MAX_TEMPLATE_TYPES];
                        for (int t = 0; t < num_types; t++) {
                            float wt = softmax_w[i * num_types + t];
                            if (wt < 0.02f) { sampled[t] = 0.0f; continue; }
                            float su, sv;
                            get_template_scale(t, rxc, ryc, &su, &sv, num_builtins);
                            float u = u_rot / su, v = v_rot / sv;
                            const float* tmpl = d_blurred_templates + t * TEMPLATE_SIZE * TEMPLATE_SIZE;
                            sampled[t] = bilinear_sample_template(tmpl, u, v);
                            ar += wt * sampled[t];
                        }

                        if (ar > 0.0f) {
                            // Logit parameterization: opacity/colors stored as logits
                            float opi = sigmoid_logit(opacity[i] / g_opacity_tau);
                            float clr = 255.0f * sigmoid_logit(colors[i * 3 + 0]);
                            float clg = 255.0f * sigmoid_logit(colors[i * 3 + 1]);
                            float clb = 255.0f * sigmoid_logit(colors[i * 3 + 2]);
                            float ae = ar * opi * g_alpha_upper_bound;
                            double oma = 1.0 - (double)ae;
                            if (oma < 1e-15) oma = 1e-15;
                            float omaf = (float)oma;
                            double lT_i = lTN - lTB - log(oma);
                            float Ti = exp((float)lT_i);
                            if (Ti > 1.0f) Ti = 1.0f;

                            lcr = gdr * ae * Ti; lcg = gdg * ae * Ti; lcb = gdb * ae * Ti;

                            // Quadrant RGB gradient accumulation (split analysis)
                            if (output_quadrant && grad_rgb_quadrant != nullptr) {
                                int q = (u_rot >= 0.0f ? 1 : 0) | (v_rot >= 0.0f ? 2 : 0);
                                atomicAdd(&grad_rgb_quadrant[((long long)i * 4 + q) * 3 + 0], lcr);
                                atomicAdd(&grad_rgb_quadrant[((long long)i * 4 + q) * 3 + 1], lcg);
                                atomicAdd(&grad_rgb_quadrant[((long long)i * 4 + q) * 3 + 2], lcb);
                            }

                            if (Ti > 1e-6f) {
                                float dcdar = Ti * (clr - car), dcdag = Ti * (clg - cag), dcdab = Ti * (clb - cab);
                                float dlae = gdr * dcdar + gdg * dcdag + gdb * dcdab + gda * 255.0f * Ti;
                                // Yield gradient: ∂L_yield/∂ae_k = yield_norm · [(1-T_k) + vis_below_acc/(1-ae_k)]
                                // 第一项 (1-T_k): j 扩大但 vis 没相应增加的惩罚（被遮挡时大）
                                // 第二项 vis_below_acc/(1-ae_k): j 遮挡下层形状 vis 的惩罚（顶层大）
                                if (yield_norm > 0.0f) dlae += yield_norm * (-Ti + (float)vis_below_acc / fmaxf(omaf, 1e-6f));
                                // Aux-Vis: ∂L_aux/∂ae_i = -λ·g_i·T_i（g_i 常数权重，stop-gradient）。
                                // 低 vis 形状 g_i 大→朝"上层透过率 T_i 高"方向移动（dlae<0 使 ae 增大→vis 上升）。
                                // 反向 bidir 调用时 aux_vis_lambda=0，不注入（与 yield 反向传 0 同理）。
                                if (aux_vis_lambda > 0.0f && d_aux_gate != nullptr) dlae += -aux_vis_lambda * d_aux_gate[i] * Ti;
                                // FA-Loss 梯度（通过 ar 链流回 cx/cy/rx/ry/angle/logits，不流回 opacity/color）
                                // VW-FA: dL_fa/d_ar_k(p) = fa_scale · (||I(p)-μ_k||² - fa_mean_t) · mask(p) · lerp(1, T_k(p), vis_w)
                                // T_k(p)=Ti 已在上方算出（上层残余透过率）；vis_w=1 时被遮挡 Ti≈0→不误伤，vis_w=0 退化为原 FA。
                                float fa_grad_ar = 0.0f;
                                if (fa_scale > 0.0f && fa_mask > 0.5f && d_fa_mu != nullptr) {
                                    float mu_kr = d_fa_mu[i * 3 + 0];
                                    float mu_kg = d_fa_mu[i * 3 + 1];
                                    float mu_kb = d_fa_mu[i * 3 + 2];
                                    float fdr = fa_Ir - mu_kr, fdg = fa_Ig - mu_kg, fdb = fa_Ib - mu_kb;
                                    float dist_sq = fdr * fdr + fdg * fdg + fdb * fdb;
                                    float vw = 1.0f + (Ti - 1.0f) * fa_vis_weight;  // lerp(1, Ti, vis_w)
                                    fa_grad_ar = fa_scale * (dist_sq - fa_mean_t) * fa_mask * vw;
                                }
                                // SP-Loss 梯度注入：∂L_sp/∂ae_i = sp_norm · max(0, W-K) · (1 - (1-C)/(1-ae_i))
                                // W=sp_sum_ae-sp_total_alpha（Pass1 累加），K=d_k_allowed[idx]
                                // (1-(1-C)/(1-ae_i)) = 1-Π_{j≠i}(1-ae_j) = 其余形状总覆盖（clip 到 [0,1] 防数值漂移）
                                if (sp_active) {
                                    float W = sp_sum_ae - sp_total_alpha;
                                    float K = d_k_allowed[idx];
                                    float W_minus_K = W - K;
                                    if (W_minus_K > 0.0f) {
                                        float one_minus_ae = fmaxf(1.0f - ae, 1e-6f);
                                        // ∂W/∂ae_i = 1 - Π_{j≠i}(1-ae_j) = 1 - (1-C)/(1-ae_i)
                                        // 注意：必须用 1 - (...)/(1-ae_i)，旧版漏了 "1 -" 导致梯度错误
                                        float T_without_i = 1.0f - (1.0f - sp_total_alpha) / one_minus_ae;
                                        T_without_i = fmaxf(0.0f, fminf(1.0f, T_without_i));
                                        dlae += sp_norm * W_minus_K * T_without_i;
                                    }
                                }
                                // SP-GEO-Loss: 几何覆盖层数梯度，直接注入 ar 链（不进 dlae，不被 opacity 衰减）
                                float geo_grad_ar = 0.0f;
                                if (sp_geo_active) {
                                    float G = sp_sum_ar;
                                    float K = d_k_allowed[idx];
                                    float G_minus_K = G - K;
                                    if (G_minus_K > 0.0f) {
                                        geo_grad_ar = sp_norm * G_minus_K;   // (2λ/N)·max(0, G-K)
                                    }
                                }

                                for (int t = 0; t < num_types; t++) {
                                    float wt = softmax_w[i * num_types + t];
                                    if (wt < 0.02f) continue;
                                    float su, sv;
                                    get_template_scale(t, rxc, ryc, &su, &sv, num_builtins);
                                    float u = u_rot / su, v = v_rot / sv;
                                    const float* tmpl = d_blurred_templates + t * TEMPLATE_SIZE * TEMPLATE_SIZE;

                                    float dadu, dadv;
                                    bilinear_sample_gradient(tmpl, u, v, &dadu, &dadv);

                                    float dadcx = dadu * (-coi / su) + dadv * (sii / sv);
                                    float dadcy = dadu * (-sii / su) + dadv * (-coi / sv);
                                    float dadrx = dadu * (-u / rxc);
                                    float dadry = dadv * (-v / ryc);
                                    float dadan = dadu * (v_rot / su * PI_F / 180.0f)
                                                + dadv * (-u_rot / sv * PI_F / 180.0f);

                                    lcx += (dlae * opi + fa_grad_ar + geo_grad_ar) * wt * dadcx;
                                    lcy += (dlae * opi + fa_grad_ar + geo_grad_ar) * wt * dadcy;
                                    lrx += (dlae * opi + fa_grad_ar + geo_grad_ar) * wt * dadrx;
                                    lry += (dlae * opi + fa_grad_ar + geo_grad_ar) * wt * dadry;
                                    lan += (dlae * opi + fa_grad_ar + geo_grad_ar) * wt * dadan;
                                }
                                lop = dlae * ar;

                                // DARTS logit gradient (generalized loop)
                                // 注意：不乘 1/T 温度因子。实验验证（E52 vs E46 on 105053）：
                                //   有 1/T: 28.57 dB
                                //   无 1/T: 29.22 dB (+0.65 dB)
                                // Adam 会自动归一化梯度幅度，温度只影响 softmax 分布尖锐度，
                                // 不需要梯度补偿。1/T 反而破坏了探索-利用平衡。
                                for (int t = 0; t < num_types; t++) {
                                    float gl_t = (dlae * opi + fa_grad_ar + geo_grad_ar) * softmax_w[i * num_types + t] * (sampled[t] - ar);
                                    if (deterministic) {
                                        lgl[t] += gl_t;
                                    } else {
                                        atomicAdd(&grad_logits[i * num_types + t], gl_t);
                                    }
                                }
                            }

                            car = ae * clr + omaf * car;
                            cag = ae * clg + omaf * cag;
                            cab = ae * clb + omaf * cab;
                            lTB += log(oma);
                            // YMP: 广播当前形状的潜在能力 vis_full=opi·π·rx·ry 到 vis_below_acc（不衰减，供上层 k 让位）
                            if (yield_norm > 0.0f) vis_below_acc += (double)opi * (double)PI_F * (double)rxc * (double)ryc * (double)err_norm;
                        }
                    } else {
                        int stype = to_template_idx(shape_types[i], num_builtins);
                        float su, sv;
                        get_template_scale(stype, rxc, ryc, &su, &sv, num_builtins);
                        float u = u_rot / su, v = v_rot / sv;
                        const float* tmpl = d_blurred_templates + stype * TEMPLATE_SIZE * TEMPLATE_SIZE;
                        ar = bilinear_sample_template(tmpl, u, v);

                        if (ar > 0.0f) {
                            // Logit parameterization: opacity/colors stored as logits
                            float opi = sigmoid_logit(opacity[i] / g_opacity_tau);
                            float clr = 255.0f * sigmoid_logit(colors[i * 3 + 0]);
                            float clg = 255.0f * sigmoid_logit(colors[i * 3 + 1]);
                            float clb = 255.0f * sigmoid_logit(colors[i * 3 + 2]);
                            float ae = ar * opi * g_alpha_upper_bound;
                            double oma = 1.0 - (double)ae;
                            if (oma < 1e-15) oma = 1e-15;
                            float omaf = (float)oma;
                            double lT_i = lTN - lTB - log(oma);
                            float Ti = exp((float)lT_i);
                            if (Ti > 1.0f) Ti = 1.0f;

                            lcr = gdr * ae * Ti; lcg = gdg * ae * Ti; lcb = gdb * ae * Ti;

                            // Quadrant RGB gradient accumulation (split analysis)
                            if (output_quadrant && grad_rgb_quadrant != nullptr) {
                                int q = (u_rot >= 0.0f ? 1 : 0) | (v_rot >= 0.0f ? 2 : 0);
                                atomicAdd(&grad_rgb_quadrant[((long long)i * 4 + q) * 3 + 0], lcr);
                                atomicAdd(&grad_rgb_quadrant[((long long)i * 4 + q) * 3 + 1], lcg);
                                atomicAdd(&grad_rgb_quadrant[((long long)i * 4 + q) * 3 + 2], lcb);
                            }

                            if (Ti > 1e-6f) {
                                float dcdar = Ti * (clr - car), dcdag = Ti * (clg - cag), dcdab = Ti * (clb - cab);
                                float dlae = gdr * dcdar + gdg * dcdag + gdb * dcdab + gda * 255.0f * Ti;
                                // Yield gradient: ∂L_yield/∂ae_k = yield_norm · [(1-T_k) + vis_below_acc/(1-ae_k)]
                                // 第一项 (1-T_k): j 扩大但 vis 没相应增加的惩罚（被遮挡时大）
                                // 第二项 vis_below_acc/(1-ae_k): j 遮挡下层形状 vis 的惩罚（顶层大）
                                if (yield_norm > 0.0f) dlae += yield_norm * (-Ti + (float)vis_below_acc / fmaxf(omaf, 1e-6f));
                                // Aux-Vis: ∂L_aux/∂ae_i = -λ·g_i·T_i（g_i 常数权重，stop-gradient）。
                                // 低 vis 形状 g_i 大→朝"上层透过率 T_i 高"方向移动（dlae<0 使 ae 增大→vis 上升）。
                                // 反向 bidir 调用时 aux_vis_lambda=0，不注入（与 yield 反向传 0 同理）。
                                if (aux_vis_lambda > 0.0f && d_aux_gate != nullptr) dlae += -aux_vis_lambda * d_aux_gate[i] * Ti;
                                // FA-Loss 梯度（通过 ar 链流回 cx/cy/rx/ry/angle/logits，不流回 opacity/color）
                                // VW-FA: dL_fa/d_ar_k(p) = fa_scale · (||I(p)-μ_k||² - fa_mean_t) · mask(p) · lerp(1, T_k(p), vis_w)
                                // T_k(p)=Ti 已在上方算出（上层残余透过率）；vis_w=1 时被遮挡 Ti≈0→不误伤，vis_w=0 退化为原 FA。
                                float fa_grad_ar = 0.0f;
                                if (fa_scale > 0.0f && fa_mask > 0.5f && d_fa_mu != nullptr) {
                                    float mu_kr = d_fa_mu[i * 3 + 0];
                                    float mu_kg = d_fa_mu[i * 3 + 1];
                                    float mu_kb = d_fa_mu[i * 3 + 2];
                                    float fdr = fa_Ir - mu_kr, fdg = fa_Ig - mu_kg, fdb = fa_Ib - mu_kb;
                                    float dist_sq = fdr * fdr + fdg * fdg + fdb * fdb;
                                    float vw = 1.0f + (Ti - 1.0f) * fa_vis_weight;  // lerp(1, Ti, vis_w)
                                    fa_grad_ar = fa_scale * (dist_sq - fa_mean_t) * fa_mask * vw;
                                }
                                // SP-Loss 梯度注入（locked 分支）：同 DARTS 分支公式
                                if (sp_active) {
                                    float W = sp_sum_ae - sp_total_alpha;
                                    float K = d_k_allowed[idx];
                                    float W_minus_K = W - K;
                                    if (W_minus_K > 0.0f) {
                                        float one_minus_ae = fmaxf(1.0f - ae, 1e-6f);
                                        // ∂W/∂ae_i = 1 - Π_{j≠i}(1-ae_j) = 1 - (1-C)/(1-ae_i)
                                        // 注意：必须用 1 - (...)/(1-ae_i)，旧版漏了 "1 -" 导致梯度错误
                                        float T_without_i = 1.0f - (1.0f - sp_total_alpha) / one_minus_ae;
                                        T_without_i = fmaxf(0.0f, fminf(1.0f, T_without_i));
                                        dlae += sp_norm * W_minus_K * T_without_i;
                                    }
                                }
                                // SP-GEO-Loss: 几何覆盖层数梯度，直接注入 ar 链（不进 dlae，不被 opacity 衰减）
                                float geo_grad_ar = 0.0f;
                                if (sp_geo_active) {
                                    float G = sp_sum_ar;
                                    float K = d_k_allowed[idx];
                                    float G_minus_K = G - K;
                                    if (G_minus_K > 0.0f) {
                                        geo_grad_ar = sp_norm * G_minus_K;   // (2λ/N)·max(0, G-K)
                                    }
                                }

                                float dadu, dadv;
                                bilinear_sample_gradient(tmpl, u, v, &dadu, &dadv);

                                float dadcx = dadu * (-coi / su) + dadv * (sii / sv);
                                float dadcy = dadu * (-sii / su) + dadv * (-coi / sv);
                                float dadrx = dadu * (-u / rxc);
                                float dadry = dadv * (-v / ryc);
                                float dadan = dadu * (v_rot / su * PI_F / 180.0f)
                                            + dadv * (-u_rot / sv * PI_F / 180.0f);

                                lcx = (dlae * opi + fa_grad_ar + geo_grad_ar) * dadcx;
                                lcy = (dlae * opi + fa_grad_ar + geo_grad_ar) * dadcy;
                                lrx = (dlae * opi + fa_grad_ar + geo_grad_ar) * dadrx;
                                lry = (dlae * opi + fa_grad_ar + geo_grad_ar) * dadry;
                                lan = (dlae * opi + fa_grad_ar + geo_grad_ar) * dadan;
                                lop = dlae * ar;

                                float sampled_all[MAX_TEMPLATE_TYPES];
                                #pragma unroll
                                for (int t = 0; t < num_types; t++) {
                                    float su_t, sv_t;
                                    get_template_scale(t, rxc, ryc, &su_t, &sv_t, num_builtins);
                                    float u_t = u_rot / su_t, v_t = v_rot / sv_t;
                                    const float* tmpl_t = d_blurred_templates + t * TEMPLATE_SIZE * TEMPLATE_SIZE;
                                    sampled_all[t] = bilinear_sample_template(tmpl_t, u_t, v_t);
                                }
                                // 锁定分支仍计算 grad_logits（退出机制），不乘 1/T（与 DARTS 分支一致）
                                for (int t = 0; t < num_types; t++) {
                                    float gl_t = (dlae * opi + fa_grad_ar + geo_grad_ar) * softmax_w[i * num_types + t] * (sampled_all[t] - ar);
                                    if (deterministic) {
                                        lgl[t] += gl_t;
                                    } else {
                                        atomicAdd(&grad_logits[i * num_types + t], gl_t);
                                    }
                                }
                            }

                            car = ae * clr + omaf * car;
                            cag = ae * clg + omaf * cag;
                            cab = ae * clb + omaf * cab;
                            lTB += log(oma);
                            // YMP: 广播当前形状的潜在能力 vis_full=opi·π·rx·ry 到 vis_below_acc（不衰减，供上层 k 让位）
                            if (yield_norm > 0.0f) vis_below_acc += (double)opi * (double)PI_F * (double)rxc * (double)ryc * (double)err_norm;
                        }
                    }
                }
            }

            // warp reduction（所有 thread 参与，valid_pixel=false 的贡献 0）
            unsigned warp_active = __activemask();
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                lcx += __shfl_down_sync(warp_active, lcx, off);
                lcy += __shfl_down_sync(warp_active, lcy, off);
                lrx += __shfl_down_sync(warp_active, lrx, off);
                lry += __shfl_down_sync(warp_active, lry, off);
                lan += __shfl_down_sync(warp_active, lan, off);
                lcr += __shfl_down_sync(warp_active, lcr, off);
                lcg += __shfl_down_sync(warp_active, lcg, off);
                lcb += __shfl_down_sync(warp_active, lcb, off);
                lop += __shfl_down_sync(warp_active, lop, off);
            }
            // R23: intra-warp reduction for grad_logits (deterministic only)
            if (deterministic) {
                for (int t = 0; t < num_types; t++)
                    for (int off = 16; off > 0; off >>= 1)
                        lgl[t] += __shfl_down_sync(warp_active, lgl[t], off);
            }
            if (lid == 0) {
                s_block[wid * 9 + 0] = lcx; s_block[wid * 9 + 1] = lcy;
                s_block[wid * 9 + 2] = lrx; s_block[wid * 9 + 3] = lry;
                s_block[wid * 9 + 4] = lan;
                s_block[wid * 9 + 5] = lcr; s_block[wid * 9 + 6] = lcg;
                s_block[wid * 9 + 7] = lcb; s_block[wid * 9 + 8] = lop;
                // R23: write lgl to shared memory for cross-warp reduction
                if (deterministic) {
                    for (int t = 0; t < num_types; t++)
                        s_gl[wid * MAX_TEMPLATE_TYPES + t] = lgl[t];
                }
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
                    // Chain rule: grad_logit = grad_linear * d_linear/d_logit
                    // For color: clr = 255*sigmoid(lc), so d_clr/d_lc = 255 * s_c * (1-s_c)
                    // For opacity: opi = sigmoid(lo), so d_opi/d_lo = s_o * (1-s_o)
                    float sc0 = sigmoid_logit(colors[i * 3 + 0]);
                    float sc1 = sigmoid_logit(colors[i * 3 + 1]);
                    float sc2 = sigmoid_logit(colors[i * 3 + 2]);
                    float so  = sigmoid_logit(opacity[i] / g_opacity_tau);
                    float g_cr = blcr * 255.0f * sigmoid_deriv_from_s(sc0);
                    float g_cg = blcg * 255.0f * sigmoid_deriv_from_s(sc1);
                    float g_cb = blcb * 255.0f * sigmoid_deriv_from_s(sc2);
                    float g_op = blop * sigmoid_deriv_from_s(so) / g_opacity_tau;
                    if (deterministic) {
                        // R23: 写 per-tile-per-shape partial，后续 reduce kernel 固定顺序归约
                        // 用 tile_idx（全局）而非 blockIdx.x（batch 内），避免 batch 间覆盖
                        int base = (tile_idx * N + i) * 9;
                        grad_partials[base + 0] = blcx; grad_partials[base + 1] = blcy;
                        grad_partials[base + 2] = blrx; grad_partials[base + 3] = blry;
                        grad_partials[base + 4] = blan;
                        grad_partials[base + 5] = g_cr;  grad_partials[base + 6] = g_cg;
                        grad_partials[base + 7] = g_cb;  grad_partials[base + 8] = g_op;
                    } else {
                        atomicAdd(&grad_cx[i], blcx); atomicAdd(&grad_cy[i], blcy);
                        atomicAdd(&grad_rx[i], blrx); atomicAdd(&grad_ry[i], blry);
                        atomicAdd(&grad_angle[i], blan);
                        atomicAdd(&grad_colors[i * 3 + 0], g_cr);
                        atomicAdd(&grad_colors[i * 3 + 1], g_cg);
                        atomicAdd(&grad_colors[i * 3 + 2], g_cb);
                        atomicAdd(&grad_opacity[i], g_op);
                    }
                }
                // R23: cross-warp reduction for grad_logits (deterministic only)
                if (deterministic) {
                    for (int t = 0; t < num_types; t++) {
                        float val = (lid < num_warps) ? s_gl[lid * MAX_TEMPLATE_TYPES + t] : 0.0f;
                        for (int off = 16; off > 0; off >>= 1)
                            val += __shfl_down_sync(bmask, val, off);
                        if (lid == 0)
                            grad_logits_partials[(tile_idx * N + i) * num_types + t] = val;
                    }
                }
            }
            __syncthreads();
        }
    }

    // ========== Pass 3: z gradient (soft depth only) — 局部化 ==========
    // 只在覆盖当前像素的形状间计算 z 梯度（cover_count 个，通常 << 64）。
    // cover 数组已保证所有 ae > 0，故无需再检 ae<=0；只需跳过 Ti<=0（理论不会发生）。
    // 公式（与 forward pass 一致，p_ji = sigmoid((z_j - z_i)/tau)）：
    //   ∂L/∂z_i = Σ_k ∂L/∂T_k · ∂T_k/∂z_i
    //   k ≠ i: ∂T_k/∂z_i = -(T_k/tau) · [ae_i · p_ik·(1-p_ik) / (1 - ae_i·p_ik)]
    //   k = i: ∂T_i/∂z_i =  (T_i/tau) · Σ_{j≠i} [ae_j · p_ji·(1-p_ji) / (1 - ae_j·p_ji)]
    // per-thread atomicAdd：每个线程（像素）独立计算并累加 z 梯度。
    // 不能用 warp reduction，因为不同线程的 cover_shape_id[ii] 是不同的形状，
    // reduction 会把不同像素对不同形状的梯度混在一起。
    if (use_sd_backward && grad_z != nullptr && cover_count > 1 && valid_pixel) {
        float inv_tau = 1.0f / tau_z;
        for (int ii = 0; ii < cover_count; ii++) {
            float zi = zi_arr_sd[ii];
            float ae_i = ae_arr_sd[ii];
            float gz = 0.0f;
            for (int k = 0; k < cover_count; k++) {
                if (Ti_arr_sd[k] <= 0.0f) continue;
                float dL_dTk = (gdr * (clr_arr_sd[k] - bg_color[0])
                              + gdg * (clg_arr_sd[k] - bg_color[1])
                              + gdb * (clb_arr_sd[k] - bg_color[2])
                              + gda * 255.0f) * ae_arr_sd[k];
                if (k != ii) {
                    float diff = (zi_arr_sd[k] - zi) * inv_tau;
                    diff = fmaxf(-20.0f, fminf(20.0f, diff));
                    float p = 1.0f / (1.0f + expf(-diff));
                    float om = 1.0f - ae_i * p;
                    if (om < 1e-15f) om = 1e-15f;
                    float dT_dz = (Ti_arr_sd[k] * inv_tau) *
                                  (ae_i * p * (1.0f - p) / om);
                    gz += dL_dTk * dT_dz;
                } else {
                    float sum = 0.0f;
                    for (int j = 0; j < cover_count; j++) {
                        if (j == ii) continue;
                        float diff2 = (zi - zi_arr_sd[j]) * inv_tau;
                        diff2 = fmaxf(-20.0f, fminf(20.0f, diff2));
                        float p2 = 1.0f / (1.0f + expf(-diff2));
                        float om2 = 1.0f - ae_arr_sd[j] * p2;
                        if (om2 < 1e-15f) om2 = 1e-15f;
                        sum += ae_arr_sd[j] * p2 * (1.0f - p2) / om2;
                    }
                    float dT_dz = -(Ti_arr_sd[ii] * inv_tau) * sum;
                    gz += dL_dTk * dT_dz;
                }
            }
            atomicAdd(&grad_z[cover_shape_id[ii]], gz);
        }
    }
}

extern "C" {

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
    float inv_temperature,
    bool reverse,
    const float* d_z, float tau_z, int soft_depth_max_m,
    float* grad_z,
    int N,
    float yield_lambda,
    const float* error_map,                    // [total_pixels] EM-YMP 误差图 (nullptr=YMP/反向 bidir)
    const float* d_mean_err,                  // [1] 全局平均误差 device 标量 (nullptr=YMP/反向 bidir)
    // === FA-Loss 注入参数 ===
    const float* d_fa_globals,                 // [3] {fa_scale, fa_mean_t, fa_vis_weight} (nullptr=禁用)
    const float* d_fa_mu,                      // [N*3] 每形状 μ_i (nullptr=禁用)
    const float* d_target,                     // [total_pixels*3] 目标图像
    const float* d_mask,                       // [total_pixels] mask
    // === Auxiliary Visibility Gradient 注入参数 ===
    float aux_vis_lambda,                      // 0=禁用（反向 bidir 传 0）
    const float* d_aux_gate,                   // [N] g_i 唤醒权重 (nullptr=禁用)
    // === Stacking Penalty Loss 注入参数 ===
    float sp_lambda,                           // 0=禁用（反向 bidir 传 0）
    const float* d_k_allowed,                  // [total_pixels] K_allowed(p) (nullptr=禁用)
    bool sp_geo_loss_active,                   // true=几何覆盖损失模式
    // === R23: 确定性归约参数 ===
    bool deterministic,
    float* grad_partials,                      // [num_blocks * N * 9] 确定性梯度 partials
    float* grad_logits_partials)               // [num_blocks * N * num_types] 确定性 logits partials
{
    // P3-2: tile 对齐 launch，每 block 处理一个 tile
    if (batch_tile_count == 0) return;
    int threads = tile_size * tile_size;  // 16*16 = 256
    int blocks = batch_tile_count;
    tile_backward_kernel<<<blocks, threads, 0, stream>>>(
        grad_output, srgb_out, px, py, cx, cy, rx, ry, angle,
        colors, opacity, bg_color, sina, cosa,
        d_blurred_templates,
        grad_cx, grad_cy, grad_rx, grad_ry,
        grad_angle, grad_colors, grad_opacity,
        d_tile_shape_lists, d_tile_shape_counts,
        num_tiles_x, num_tiles_y, tile_size, max_shapes_per_tile,
        batch_tile_start, batch_tile_count,
        W, H,
        grad_alpha, softmax_w, grad_logits,
        shape_types, render_mode, blur_sigma,
        output_quadrant, grad_rgb_quadrant, num_types, num_builtins,
        inv_temperature, reverse,
        d_z, tau_z, soft_depth_max_m, grad_z, N, yield_lambda, error_map, d_mean_err,
        d_fa_globals, d_fa_mu, d_target, d_mask,
        aux_vis_lambda, d_aux_gate,
        sp_lambda, d_k_allowed,
        sp_geo_loss_active,
        deterministic, grad_partials, grad_logits_partials);
}

} // extern "C" (tile backward)
