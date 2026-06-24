#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cmath>

#include "cuda/canvas_render.h"
#include "common/types.h"
#include "common/logger.h"
#include "common/alpha_228.h"
#include "cuda/color_utils.cuh"

#define PI_F 3.14159265358979323846f

using vinylizer::ALPHA_LUT_SIZE;

__device__ __forceinline__ float alpha_228_lut(float rn, cudaTextureObject_t tex) {
    if (rn >= 1.0f) return 0.0f;
    float coord = rn * (1.0f - 1.0f / ALPHA_LUT_SIZE) + 0.5f / ALPHA_LUT_SIZE;
    return tex1D<float>(tex, coord);
}

__global__ void canvas_render_kernel(
    const float* __restrict__ px, const float* __restrict__ py,
    const float* __restrict__ cx, const float* __restrict__ cy,
    const float* __restrict__ rx, const float* __restrict__ ry,
    const float* __restrict__ angle,
    const float* __restrict__ colors,
    const float* __restrict__ opacity,
    const int* __restrict__ types,
    float* __restrict__ output,
    int N, int total_pixels,
    cudaTextureObject_t alpha_tex)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pixels) return;

    float pxi = px[idx], pyi = py[idx];
    float rr = 0.0f, gg = 0.0f, bb = 0.0f, T = 1.0f;

    for (int i = N - 1; i >= 0; i--) {
        int type = types[i];
        float alpha;

        if (type == 1) {
            float hw = rx[i] * 0.5f, hh = ry[i] * 0.5f;
            bool inside = (pxi >= cx[i] - hw && pxi <= cx[i] + hw &&
                           pyi >= cy[i] - hh && pyi <= cy[i] + hh);
            alpha = inside ? 1.0f : 0.0f;
        } else if (type == 16) {
            float rad = angle[i] * PI_F / 180.0f;
            float coi = cosf(rad), sii = sinf(rad);
            float dx = pxi - cx[i], dy = pyi - cy[i];
            float u = dx * coi + dy * sii;
            float v = -dx * sii + dy * coi;
            float dsq = (u / rx[i]) * (u / rx[i]) + (v / ry[i]) * (v / ry[i]);
            alpha = (dsq <= 1.0f) ? 1.0f : 0.0f;
        } else {
            float rad = angle[i] * PI_F / 180.0f;
            float coi = cosf(rad), sii = sinf(rad);
            float dx = pxi - cx[i], dy = pyi - cy[i];
            float u = dx * coi + dy * sii;
            float v = -dx * sii + dy * coi;
            float rxc = fmaxf(rx[i], 0.5f), ryc = fmaxf(ry[i], 0.5f);
            float dsq = (u / rxc) * (u / rxc) + (v / ryc) * (v / ryc);
            float rn = sqrtf(fmaxf(dsq, 0.0f));
            alpha = alpha_228_lut(rn, alpha_tex);
        }

        float ae = alpha * opacity[i];
        float w = ae * T;
        rr += w * srgb_to_linear(colors[i * 3 + 0]);
        gg += w * srgb_to_linear(colors[i * 3 + 1]);
        bb += w * srgb_to_linear(colors[i * 3 + 2]);
        T *= (1.0f - ae);
        if (T < 1e-3f) break;
    }

    output[idx * 3 + 0] = linear_to_srgb(rr);
    output[idx * 3 + 1] = linear_to_srgb(gg);
    output[idx * 3 + 2] = linear_to_srgb(bb);
}

namespace vinylizer {

cv::Mat canvas_render_gpu(const std::vector<Shape>& shapes, int w, int h) {
    if (shapes.empty() || w <= 0 || h <= 0) {
        return cv::Mat::zeros(h, w, CV_8UC3);
    }

    int N = static_cast<int>(shapes.size());
    int total_pixels = w * h;
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    std::vector<float> h_px(total_pixels), h_py(total_pixels);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int idx = y * w + x;
            h_px[idx] = static_cast<float>(x);
            h_py[idx] = static_cast<float>(y);
        }
    }

    float *d_px = nullptr, *d_py = nullptr;
    cudaMalloc(&d_px, total_pixels * sizeof(float));
    cudaMalloc(&d_py, total_pixels * sizeof(float));
    cudaMemcpyAsync(d_px, h_px.data(), total_pixels * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_py, h_py.data(), total_pixels * sizeof(float), cudaMemcpyHostToDevice, stream);

    float *d_cx = nullptr, *d_cy = nullptr, *d_rx = nullptr, *d_ry = nullptr;
    float *d_angle = nullptr, *d_colors = nullptr, *d_opacity = nullptr;
    int *d_types = nullptr;
    cudaMalloc(&d_cx, N * sizeof(float));
    cudaMalloc(&d_cy, N * sizeof(float));
    cudaMalloc(&d_rx, N * sizeof(float));
    cudaMalloc(&d_ry, N * sizeof(float));
    cudaMalloc(&d_angle, N * sizeof(float));
    cudaMalloc(&d_colors, N * 3 * sizeof(float));
    cudaMalloc(&d_opacity, N * sizeof(float));
    cudaMalloc(&d_types, N * sizeof(int));

    float *d_output = nullptr;
    cudaMalloc(&d_output, total_pixels * 3 * sizeof(float));

    std::vector<float> h_cx(N), h_cy(N), h_rx(N), h_ry(N);
    std::vector<float> h_angle(N), h_colors(N * 3), h_opacity(N);
    std::vector<int> h_types(N);
    for (int i = 0; i < N; ++i) {
        const auto& s = shapes[i];
        h_cx[i] = s.cx;
        h_cy[i] = s.cy;
        h_rx[i] = s.is_rect() ? s.rx : s.rx;
        h_ry[i] = s.is_rect() ? s.ry : s.ry;
        h_angle[i] = s.angle;
        h_colors[i * 3 + 0] = static_cast<float>(s.color.r);
        h_colors[i * 3 + 1] = static_cast<float>(s.color.g);
        h_colors[i * 3 + 2] = static_cast<float>(s.color.b);
        h_opacity[i] = s.opacity;
        h_types[i] = s.type;
    }

    cudaMemcpyAsync(d_cx, h_cx.data(), N * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_cy, h_cy.data(), N * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_rx, h_rx.data(), N * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_ry, h_ry.data(), N * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_angle, h_angle.data(), N * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_colors, h_colors.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_opacity, h_opacity.data(), N * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_types, h_types.data(), N * sizeof(int), cudaMemcpyHostToDevice, stream);

    // Alpha LUT texture (from shared LUT)
    const auto& shared_lut = alpha_228_lut();

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    cudaArray* alpha_array = nullptr;
    cudaMallocArray(&alpha_array, &desc, ALPHA_LUT_SIZE, 1);
    cudaMemcpyToArray(alpha_array, 0, 0, shared_lut.data(), ALPHA_LUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = alpha_array;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    cudaTextureObject_t alpha_tex = 0;
    cudaCreateTextureObject(&alpha_tex, &resDesc, &texDesc, nullptr);

    int threads = 256;
    int blocks = (total_pixels + threads - 1) / threads;
    canvas_render_kernel<<<blocks, threads, 0, stream>>>(
        d_px, d_py, d_cx, d_cy, d_rx, d_ry, d_angle,
        d_colors, d_opacity, d_types, d_output,
        N, total_pixels, alpha_tex);

    std::vector<float> h_output(total_pixels * 3);
    cudaMemcpyAsync(h_output.data(), d_output, total_pixels * 3 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

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

    cudaDestroyTextureObject(alpha_tex);
    cudaFreeArray(alpha_array);
    cudaFree(d_output);
    cudaFree(d_types);
    cudaFree(d_opacity); cudaFree(d_colors);
    cudaFree(d_angle);
    cudaFree(d_ry); cudaFree(d_rx); cudaFree(d_cy); cudaFree(d_cx);
    cudaFree(d_py); cudaFree(d_px);
    cudaStreamDestroy(stream);

    return result;
}

} // namespace vinylizer
