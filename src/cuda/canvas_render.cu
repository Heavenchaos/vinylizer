#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "cuda/canvas_render.h"
#include "common/types.h"
#include "common/logger.h"
#include "common/alpha_228.h"
#include "cuda/color_utils.cuh"
#include "common/stamp_loader.h"
#include "cuda/template_sampling.cuh"

#define PI_F 3.14159265358979323846f

using vinylizer::ALPHA_LUT_SIZE;

#define CHECK_CUDA(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        VIN_ERROR("[CUDA] %s:%d %s failed: %s", __FILE__, __LINE__, #call, cudaGetErrorString(_err)); \
    } \
} while(0)

__device__ __forceinline__ float alpha_228_lut(float rn, cudaTextureObject_t tex) {
    if (rn >= 1.0f) return 0.0f;
    float coord = rn * (1.0f - 1.0f / ALPHA_LUT_SIZE) + 0.5f / ALPHA_LUT_SIZE;
    return tex1D<float>(tex, coord);
}

__global__ void canvas_render_kernel(
    const float* __restrict__ px, const float* __restrict__ py,
    const float* __restrict__ cx, const float* __restrict__ cy,
    const float* __restrict__ rx, const float* __restrict__ ry,
    const float* __restrict__ cos_a, const float* __restrict__ sin_a,
    const float* __restrict__ colors,
    const float* __restrict__ opacity,
    const int* __restrict__ types,
    float* __restrict__ output,
    int N, int total_pixels,
    cudaTextureObject_t alpha_tex,
    cudaTextureObject_t* d_stamp_tex,
    int* d_stamp_indices,
    int num_stamps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pixels) return;

    float pxi = px[idx], pyi = py[idx];
    float rr = 0.0f, gg = 0.0f, bb = 0.0f, T = 1.0f;

    for (int i = N - 1; i >= 0; i--) {
        float dx = pxi - cx[i], dy = pyi - cy[i];
        float ac = fabsf(cos_a[i]), as = fabsf(sin_a[i]);
        float hx = rx[i] * ac + ry[i] * as;
        float hy = rx[i] * as + ry[i] * ac;
        if (fabsf(dx) > hx || fabsf(dy) > hy) continue;

        int type = types[i];
        float alpha;
        float coi = cos_a[i], sii = sin_a[i];

        int stamp_idx = (d_stamp_indices && i < N) ? d_stamp_indices[i] : -1;

        if (stamp_idx >= 0 && stamp_idx < num_stamps && d_stamp_tex) {
            // 外部图章：仿射采样（椭圆模式）
            float u_rot = dx * coi + dy * sii;
            float v_rot = -dx * sii + dy * coi;
            float u = (u_rot / rx[i] + 1.0f) * 0.5f;
            float v = (v_rot / ry[i] + 1.0f) * 0.5f;
            if (u >= 0.0f && u <= 1.0f && v >= 0.0f && v <= 1.0f) {
                alpha = tex2D<float>(d_stamp_tex[stamp_idx], u, v);
            } else {
                alpha = 0.0f;
            }
        } else if (type == 1) {
            float hw = rx[i] * 0.5f, hh = ry[i] * 0.5f;
            float u = dx * coi + dy * sii;
            float v = -dx * sii + dy * coi;
            bool inside = (fabsf(u) <= hw && fabsf(v) <= hh);
            alpha = inside ? 1.0f : 0.0f;
        } else if (type == 16) {
            float u = dx * coi + dy * sii;
            float v = -dx * sii + dy * coi;
            float dsq = (u / rx[i]) * (u / rx[i]) + (v / ry[i]) * (v / ry[i]);
            alpha = (dsq <= 1.0f) ? 1.0f : 0.0f;
        } else if (type == 103) {
            float hw = rx[i], hh = ry[i] * 0.5f;
            float v0x = 0.0f, v0y = -hh;
            float v1x = -hw, v1y = hh;
            float v2x = hw, v2y = hh;
            float v0_wx = cx[i] + v0x*coi - v0y*sii;
            float v0_wy = cy[i] + v0x*sii + v0y*coi;
            float v1_wx = cx[i] + v1x*coi - v1y*sii;
            float v1_wy = cy[i] + v1x*sii + v1y*coi;
            float v2_wx = cx[i] + v2x*coi - v2y*sii;
            float v2_wy = cy[i] + v2x*sii + v2y*coi;
            float d0 = (pxi - v1_wx) * (v2_wy - v1_wy) - (pyi - v1_wy) * (v2_wx - v1_wx);
            float d1 = (pxi - v2_wx) * (v0_wy - v2_wy) - (pyi - v2_wy) * (v0_wx - v2_wx);
            float d2 = (pxi - v0_wx) * (v1_wy - v0_wy) - (pyi - v0_wy) * (v1_wx - v0_wx);
            bool inside = (d0 >= 0.0f && d1 >= 0.0f && d2 >= 0.0f)
                       || (d0 <= 0.0f && d1 <= 0.0f && d2 <= 0.0f);
            alpha = inside ? 1.0f : 0.0f;
        } else {
            float u = dx * coi + dy * sii;
            float v = -dx * sii + dy * coi;
            float rxc = fmaxf(rx[i], 0.5f), ryc = fmaxf(ry[i], 0.5f);
            float dsq = (u / rxc) * (u / rxc) + (v / ryc) * (v / ryc);
            float rn = sqrtf(fmaxf(dsq, 0.0f));
            alpha = alpha_228_lut(rn, alpha_tex);
        }

        float ae = alpha * opacity[i];
        float w = ae * T;
        // sRGB→linear：颜色以 sRGB 存储，混合须在线性空间进行
        rr += w * srgb_to_linear(colors[i * 3 + 0]);
        gg += w * srgb_to_linear(colors[i * 3 + 1]);
        bb += w * srgb_to_linear(colors[i * 3 + 2]);
        T *= (1.0f - ae);
        if (T < 1e-3f) break;
    }

    // Composite over white background (align with diffbmp)
    // 白色 sRGB(255) 转线性后数值仍为 255
    rr += T * srgb_to_linear(255.0f);
    gg += T * srgb_to_linear(255.0f);
    bb += T * srgb_to_linear(255.0f);

    // linear→sRGB：输出回到 sRGB 空间
    output[idx * 3 + 0] = linear_to_srgb(rr);
    output[idx * 3 + 1] = linear_to_srgb(gg);
    output[idx * 3 + 2] = linear_to_srgb(bb);
}

namespace vinylizer {

namespace {
struct ScopedCanvasResources {
    cudaStream_t stream = nullptr;
    float* d_px = nullptr;
    float* d_py = nullptr;
    float* d_output = nullptr;
    float* d_cx = nullptr; float* d_cy = nullptr;
    float* d_rx = nullptr; float* d_ry = nullptr;
    float* d_cos_a = nullptr; float* d_sin_a = nullptr;
    float* d_colors = nullptr; float* d_opacity = nullptr;
    int* d_types = nullptr;
    cudaArray* alpha_array = nullptr;
    cudaTextureObject_t alpha_tex = 0;

    static constexpr int MAX_STAMPS = 64;
    cudaTextureObject_t stamp_tex[MAX_STAMPS] = {};
    cudaArray* stamp_arrays[MAX_STAMPS] = {};
    int num_stamps = 0;

    bool ok = false;

    ScopedCanvasResources(int w, int h, int N,
                          const std::vector<cv::Mat>& stamp_images = {}) {
        int total_pixels = w * h;
        cudaError_t err;

        err = cudaStreamCreate(&stream);
        if (err != cudaSuccess) { VIN_ERROR("cudaStreamCreate: %s", cudaGetErrorString(err)); return; }

        err = cudaMalloc(&d_px, total_pixels * sizeof(float));
        if (err != cudaSuccess) { VIN_ERROR("cudaMalloc d_px: %s", cudaGetErrorString(err)); return; }
        err = cudaMalloc(&d_py, total_pixels * sizeof(float));
        if (err != cudaSuccess) { VIN_ERROR("cudaMalloc d_py: %s", cudaGetErrorString(err)); return; }
        err = cudaMalloc(&d_output, total_pixels * 3 * sizeof(float));
        if (err != cudaSuccess) { VIN_ERROR("cudaMalloc d_output: %s", cudaGetErrorString(err)); return; }
        err = cudaMalloc(&d_cx, N * sizeof(float));
        if (err != cudaSuccess) { VIN_ERROR("cudaMalloc d_cx: %s", cudaGetErrorString(err)); return; }
        err = cudaMalloc(&d_cy, N * sizeof(float));
        if (err != cudaSuccess) { VIN_ERROR("cudaMalloc d_cy: %s", cudaGetErrorString(err)); return; }
        err = cudaMalloc(&d_rx, N * sizeof(float));
        if (err != cudaSuccess) { VIN_ERROR("cudaMalloc d_rx: %s", cudaGetErrorString(err)); return; }
        err = cudaMalloc(&d_ry, N * sizeof(float));
        if (err != cudaSuccess) { VIN_ERROR("cudaMalloc d_ry: %s", cudaGetErrorString(err)); return; }
        err = cudaMalloc(&d_cos_a, N * sizeof(float));
        if (err != cudaSuccess) { VIN_ERROR("cudaMalloc d_cos_a: %s", cudaGetErrorString(err)); return; }
        err = cudaMalloc(&d_sin_a, N * sizeof(float));
        if (err != cudaSuccess) { VIN_ERROR("cudaMalloc d_sin_a: %s", cudaGetErrorString(err)); return; }
        err = cudaMalloc(&d_colors, N * 3 * sizeof(float));
        if (err != cudaSuccess) { VIN_ERROR("cudaMalloc d_colors: %s", cudaGetErrorString(err)); return; }
        err = cudaMalloc(&d_opacity, N * sizeof(float));
        if (err != cudaSuccess) { VIN_ERROR("cudaMalloc d_opacity: %s", cudaGetErrorString(err)); return; }
        err = cudaMalloc(&d_types, N * sizeof(int));
        if (err != cudaSuccess) { VIN_ERROR("cudaMalloc d_types: %s", cudaGetErrorString(err)); return; }

        std::vector<float> h_px(total_pixels), h_py(total_pixels);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                int idx = y * w + x;
                h_px[idx] = static_cast<float>(x);
                h_py[idx] = static_cast<float>(y);
            }
        CHECK_CUDA(cudaMemcpyAsync(d_px, h_px.data(), total_pixels * sizeof(float), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(d_py, h_py.data(), total_pixels * sizeof(float), cudaMemcpyHostToDevice, stream));

        const auto& shared_lut = alpha_228_lut();
        cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
        err = cudaMallocArray(&alpha_array, &desc, ALPHA_LUT_SIZE, 1);
        if (err != cudaSuccess) { VIN_ERROR("cudaMallocArray: %s", cudaGetErrorString(err)); return; }
        err = cudaMemcpyToArray(alpha_array, 0, 0, shared_lut.data(), ALPHA_LUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) { VIN_ERROR("cudaMemcpyToArray: %s", cudaGetErrorString(err)); return; }
        cudaResourceDesc resDesc = {};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = alpha_array;
        cudaTextureDesc texDesc = {};
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 1;
        err = cudaCreateTextureObject(&alpha_tex, &resDesc, &texDesc, nullptr);
        if (err != cudaSuccess) { VIN_ERROR("cudaCreateTextureObject: %s", cudaGetErrorString(err)); return; }

        // 创建 stamp 纹理
        num_stamps = (int)stamp_images.size();
        for (int s = 0; s < num_stamps && s < MAX_STAMPS; s++) {
            const cv::Mat& img = stamp_images[s];
            cudaChannelFormatDesc chDesc = cudaCreateChannelDesc<float>();
            cudaError_t serr = cudaMallocArray(&stamp_arrays[s], &chDesc, TEMPLATE_SIZE, TEMPLATE_SIZE);
            if (serr != cudaSuccess) {
                VIN_ERROR("cudaMallocArray stamp[%d]: %s", s, cudaGetErrorString(serr));
                return;
            }

            // 提取 alpha（pydiffbmp 逻辑：背景去除 + 灰度反相）
            // 注意：stamp_images[s] 已在 json_to_image.cpp 中通过
            // resize_stamp_keep_aspect 预处理，此处为 TEMPLATE_SIZE×TEMPLATE_SIZE RGBA
            std::vector<float> h_stamp = extract_stamp_alpha(img);
            CHECK_CUDA(cudaMemcpyToArray(stamp_arrays[s], 0, 0, h_stamp.data(),
                                         TEMPLATE_SIZE * TEMPLATE_SIZE * sizeof(float), cudaMemcpyHostToDevice));

            cudaResourceDesc resDesc2 = {};
            resDesc2.resType = cudaResourceTypeArray;
            resDesc2.res.array.array = stamp_arrays[s];
            cudaTextureDesc texDesc2 = {};
            texDesc2.addressMode[0] = cudaAddressModeClamp;
            texDesc2.addressMode[1] = cudaAddressModeClamp;
            texDesc2.filterMode = cudaFilterModeLinear;
            texDesc2.readMode = cudaReadModeElementType;
            texDesc2.normalizedCoords = 1;
            serr = cudaCreateTextureObject(&stamp_tex[s], &resDesc2, &texDesc2, nullptr);
            if (serr != cudaSuccess) {
                VIN_ERROR("cudaCreateTextureObject stamp[%d]: %s", s, cudaGetErrorString(serr));
                return;
            }
        }

        ok = true;
    }

    ~ScopedCanvasResources() {
        for (int s = 0; s < num_stamps; s++) {
            if (stamp_tex[s]) cudaDestroyTextureObject(stamp_tex[s]);
            if (stamp_arrays[s]) cudaFreeArray(stamp_arrays[s]);
        }
        if (stream) { cudaStreamSynchronize(stream); cudaStreamDestroy(stream); }
        if (alpha_tex) cudaDestroyTextureObject(alpha_tex);
        if (alpha_array) cudaFreeArray(alpha_array);
        cudaFree(d_px); cudaFree(d_py); cudaFree(d_output);
        cudaFree(d_cx); cudaFree(d_cy); cudaFree(d_rx); cudaFree(d_ry);
        cudaFree(d_cos_a); cudaFree(d_sin_a);
        cudaFree(d_colors); cudaFree(d_opacity);
        cudaFree(d_types);
    }

    ScopedCanvasResources(const ScopedCanvasResources&) = delete;
    ScopedCanvasResources& operator=(const ScopedCanvasResources&) = delete;
};
} // anonymous namespace

cv::Mat canvas_render_gpu(const std::vector<Shape>& shapes, int w, int h,
                          const std::vector<std::string>& stamp_names,
                          const std::vector<cv::Mat>& stamp_images) {
    if (shapes.empty() || w <= 0 || h <= 0) {
        return cv::Mat::zeros(h, w, CV_8UC3);
    }

    // === Soft depth: 按 z 降序排序 shapes（z 大=底层在前, z 小=顶层在后）===
    // canvas_render_kernel 按 index 倒序 (N-1→0) 渲染，即后部（大 index=顶层）先画。
    // 排序后：底层（大 z）在小 index，顶层（小 z）在大 index，与 over compositing 一致。
    // 当所有 z 相等（默认 0）时，排序保持原相对顺序（stable_sort），向后兼容。
    std::vector<Shape> sorted_shapes(shapes);
    std::stable_sort(sorted_shapes.begin(), sorted_shapes.end(),
                     [](const Shape& a, const Shape& b) { return a.z > b.z; });

    int N = static_cast<int>(sorted_shapes.size());
    int total_pixels = w * h;

    ScopedCanvasResources res(w, h, N, stamp_images);
    if (!res.ok) {
        VIN_ERROR("canvas_render_gpu: failed to allocate CUDA resources");
        return cv::Mat::zeros(h, w, CV_8UC3);
    }

    std::vector<float> h_cx(N), h_cy(N), h_rx(N), h_ry(N);
    std::vector<float> h_cos_a(N), h_sin_a(N);
    std::vector<float> h_colors(N * 3), h_opacity(N);
    std::vector<int> h_types(N);
    for (int i = 0; i < N; ++i) {
        const auto& s = sorted_shapes[i];
        h_cx[i] = std::isfinite(s.cx) ? s.cx : 0.0f;
        h_cy[i] = std::isfinite(s.cy) ? s.cy : 0.0f;
        h_rx[i] = (std::isfinite(s.rx) && s.rx > 0.0f) ? s.rx : 1.0f;
        h_ry[i] = (std::isfinite(s.ry) && s.ry > 0.0f) ? s.ry : 1.0f;
        float angle = std::isfinite(s.angle) ? s.angle : 0.0f;
        float rad = angle * PI_F / 180.0f;
        h_cos_a[i] = cosf(rad);
        h_sin_a[i] = sinf(rad);
        h_colors[i * 3 + 0] = std::isfinite(s.color.r) ? std::clamp(s.color.r, 0.0f, 255.0f) : 0.0f;
        h_colors[i * 3 + 1] = std::isfinite(s.color.g) ? std::clamp(s.color.g, 0.0f, 255.0f) : 0.0f;
        h_colors[i * 3 + 2] = std::isfinite(s.color.b) ? std::clamp(s.color.b, 0.0f, 255.0f) : 0.0f;
        h_opacity[i] = std::isfinite(s.opacity) ? std::clamp(s.opacity, 0.0f, 1.0f) : 1.0f;
        h_types[i] = s.type;
    }

    // 构建 stamp_indices
    std::vector<int> h_stamp_indices(N, -1);
    for (int i = 0; i < N; ++i) {
        if (!sorted_shapes[i].stamp_name.empty()) {
            for (int s = 0; s < (int)stamp_names.size(); s++) {
                if (stamp_names[s] == sorted_shapes[i].stamp_name) {
                    h_stamp_indices[i] = s;
                    break;
                }
            }
        }
    }

    int* d_stamp_indices = nullptr;
    CHECK_CUDA(cudaMalloc(&d_stamp_indices, N * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_stamp_indices, h_stamp_indices.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    // 分配 device 端的 stamp_tex 数组
    int num_stamps = res.num_stamps;
    cudaTextureObject_t* d_stamp_tex = nullptr;
    if (num_stamps > 0) {
        CHECK_CUDA(cudaMalloc(&d_stamp_tex, num_stamps * sizeof(cudaTextureObject_t)));
        CHECK_CUDA(cudaMemcpy(d_stamp_tex, res.stamp_tex, num_stamps * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice));
    }

    CHECK_CUDA(cudaMemcpyAsync(res.d_cx,     h_cx.data(),     N * sizeof(float),     cudaMemcpyHostToDevice, res.stream));
    CHECK_CUDA(cudaMemcpyAsync(res.d_cy,     h_cy.data(),     N * sizeof(float),     cudaMemcpyHostToDevice, res.stream));
    CHECK_CUDA(cudaMemcpyAsync(res.d_rx,     h_rx.data(),     N * sizeof(float),     cudaMemcpyHostToDevice, res.stream));
    CHECK_CUDA(cudaMemcpyAsync(res.d_ry,     h_ry.data(),     N * sizeof(float),     cudaMemcpyHostToDevice, res.stream));
    CHECK_CUDA(cudaMemcpyAsync(res.d_cos_a,  h_cos_a.data(),  N * sizeof(float),     cudaMemcpyHostToDevice, res.stream));
    CHECK_CUDA(cudaMemcpyAsync(res.d_sin_a,  h_sin_a.data(),  N * sizeof(float),     cudaMemcpyHostToDevice, res.stream));
    CHECK_CUDA(cudaMemcpyAsync(res.d_colors, h_colors.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice, res.stream));
    CHECK_CUDA(cudaMemcpyAsync(res.d_opacity,h_opacity.data(),N * sizeof(float),     cudaMemcpyHostToDevice, res.stream));
    CHECK_CUDA(cudaMemcpyAsync(res.d_types,  h_types.data(),  N * sizeof(int),       cudaMemcpyHostToDevice, res.stream));

    const int CHUNK_PIXELS = 16384;
    std::vector<float> h_output(total_pixels * 3);
    int threads = 256;

    for (int chunk_start = 0; chunk_start < total_pixels; chunk_start += CHUNK_PIXELS) {
        int chunk_size = std::min(CHUNK_PIXELS, total_pixels - chunk_start);
        int blocks = (chunk_size + threads - 1) / threads;

        cudaGetLastError();  // 清除可能存在的 stale error（避免误报 kernel 启动失败）
        canvas_render_kernel<<<blocks, threads, 0, res.stream>>>(
            res.d_px + chunk_start, res.d_py + chunk_start,
            res.d_cx, res.d_cy, res.d_rx, res.d_ry,
            res.d_cos_a, res.d_sin_a,
            res.d_colors, res.d_opacity, res.d_types,
            res.d_output,
            N, chunk_size, res.alpha_tex,
            d_stamp_tex, d_stamp_indices, num_stamps);

        cudaError_t kerr = cudaGetLastError();
        if (kerr != cudaSuccess) {
            VIN_ERROR("canvas_render_kernel launch failed (chunk %d/%d): %s",
                      chunk_start / CHUNK_PIXELS,
                      (total_pixels + CHUNK_PIXELS - 1) / CHUNK_PIXELS,
                      cudaGetErrorString(kerr));
            return cv::Mat::zeros(h, w, CV_8UC3);
        }

        CHECK_CUDA(cudaMemcpyAsync(h_output.data() + chunk_start * 3,
                                    res.d_output, chunk_size * 3 * sizeof(float),
                                    cudaMemcpyDeviceToHost, res.stream));

        cudaError_t syncerr = cudaStreamSynchronize(res.stream);
        if (syncerr != cudaSuccess) {
            VIN_ERROR("cudaStreamSynchronize failed (chunk %d/%d): %s",
                      chunk_start / CHUNK_PIXELS,
                      (total_pixels + CHUNK_PIXELS - 1) / CHUNK_PIXELS,
                      cudaGetErrorString(syncerr));
            return cv::Mat::zeros(h, w, CV_8UC3);
        }
    }

    if (d_stamp_tex) cudaFree(d_stamp_tex);
    if (d_stamp_indices) cudaFree(d_stamp_indices);

    // canvas 输出的字节布局为 [R, G, B] 顺序（与 GPU 内核输出一致）
    // 注意: cv::Vec3b(R,G,B) 写入的字节顺序是 [R,G,B]，
    // 但 OpenCV CV_8UC3 约定为 [B,G,R]。
    // 因此此 cv::Mat 不可直接用于 cv::imwrite，需先 cvtColor(RGB2BGR)。
    cv::Mat result(h, w, CV_8UC3);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int idx = y * w + x;
            result.at<cv::Vec3b>(y, x) = cv::Vec3b(
                static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, h_output[idx * 3 + 0]))),
                static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, h_output[idx * 3 + 1]))),
                static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, h_output[idx * 3 + 2])))
            );
        }
    }

    return result;
}

} // namespace vinylizer
