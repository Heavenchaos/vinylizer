// k-medoids 初始化 GPU 热核实现 —— 见 kmedoids_kernels.cuh 的设计说明。
//
// 与 CPU 版的数值一致性是本文件的验收核心：
//   - 比较用严格 '<'（最小索引胜），与 CPU assign_labels 相同
//   - 距离 (x-sx)^2+(y-sy)^2 全部用 float32，与 CPU 版同精度
//   - 不做任何归约（bincount / 累积分布 / 采样全部留在 CPU）
//   KM_DIST_NO_FMA=1 时改用 IEEE round-to-nearest 内建，强制不融合成 FMA，
//   用于验证/消除 GPU 与 CPU 的末位差异。

#include "cuda/kmedoids_kernels.cuh"
#include "common/logger.h"
#include <math_constants.h>

#ifndef KM_DIST_NO_FMA
#define KM_DIST_NO_FMA 0
#endif

namespace vinylizer {

namespace {

// ── 单块 shared memory 容纳的 site 数 ──
// 4096 sites × 2 floats × 4B = 32KB，低于默认 48KB 静态上限（无需 opt-in）。
// N <= 4096（含研究定版 N=3000）走"一次装载、纯 shared 读"；
// N > 4096 自动退化为按 4096 分块的循环装载（cooperative global→shared），
// 因此任意大 N 都可用，只是每点多几次分块同步。
constexpr int kKmChunk = 4096;

// 与 CPU 版 assign_labels 的 `dx*dx + dy*dy`（float32）对应。
__device__ __forceinline__ float km_dist2(float x, float y, float sx, float sy) {
#if KM_DIST_NO_FMA
    const float dx = __fsub_rn(x, sx);
    const float dy = __fsub_rn(y, sy);
    return __fadd_rn(__fmul_rn(dx, dx), __fmul_rn(dy, dy));
#else
    const float dx = x - sx;
    const float dy = y - sy;
    return dx * dx + dy * dy;
#endif
}

// ── 最近 site 分配：每线程一个点 ──
// sites 置于 shared memory（float2），按 kKmChunk 分块装载。分块循环对所有
// 线程一致（nchunks 与 N 无关地由 kKmChunk 决定），保证 __syncthreads 不被
// 线程发散跳过。
__global__ void km_assign_kernel(const float* __restrict__ gx,
                                 const float* __restrict__ gy,
                                 const float* __restrict__ sx,
                                 const float* __restrict__ sy,
                                 int P, int N,
                                 int* __restrict__ labels)
{
    extern __shared__ float2 s_site[];

    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = (p < P);

    float x = 0.0f, y = 0.0f;
    if (active) { x = gx[p]; y = gy[p]; }
    float best = CUDART_INF_F;
    int best_k = 0;

    const int nchunks = (N + kKmChunk - 1) / kKmChunk;
    for (int c = 0; c < nchunks; ++c) {
        const int base = c * kKmChunk;
        const int cnt = min(kKmChunk, N - base);
        for (int i = threadIdx.x; i < cnt; i += blockDim.x) {
            s_site[i] = make_float2(__ldg(&sx[base + i]), __ldg(&sy[base + i]));
        }
        __syncthreads();
        if (active) {
            for (int i = 0; i < cnt; ++i) {
                const float2 s = s_site[i];
                const float d = km_dist2(x, y, s.x, s.y);
                if (d < best) { best = d; best_k = base + i; }
            }
        }
        __syncthreads();
    }

    if (active) labels[p] = best_k;
}

// ── kmeans++ D² 初始化：d2[p] = ||c_p - s||² ──
__global__ void km_d2_init_kernel(const float* __restrict__ gx,
                                  const float* __restrict__ gy,
                                  float sx, float sy,
                                  float* __restrict__ d2, int P)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < P) d2[p] = km_dist2(gx[p], gy[p], sx, sy);
}

// ── kmeans++ D² 原地更新：d2[p] = min(d2[p], ||c_p - s||²) ──
__global__ void km_d2_update_kernel(const float* __restrict__ gx,
                                    const float* __restrict__ gy,
                                    float sx, float sy,
                                    float* __restrict__ d2, int P)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < P) {
        const float dk = km_dist2(gx[p], gy[p], sx, sy);
        if (dk < d2[p]) d2[p] = dk;
    }
}

constexpr int kBlock = 256;

inline int grid_for(int n) { return (n + kBlock - 1) / kBlock; }

} // namespace

// ── device 可用性 ──
int km_cuda_device_count() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) {
        cudaGetLastError();   // 清掉 sticky error，避免影响后续 CUDA 调用
        return 0;
    }
    return n;
}

struct KmGpuAssigner::Impl {
    int P = 0;
    int N = 0;
    float* d_gx = nullptr;
    float* d_gy = nullptr;
    float* d_sx = nullptr;
    float* d_sy = nullptr;
    float* d_d2 = nullptr;
    int* d_labels = nullptr;
    float* h_d2 = nullptr;   // pinned host 缓冲：D2H 走满带宽 fast path
};

KmGpuAssigner::KmGpuAssigner(int P, int N) : impl_(new Impl()) {
    impl_->P = P;
    impl_->N = N;
    bool good = true;
    auto alloc = [&](void** ptr, size_t bytes, const char* what) {
        const cudaError_t e = cudaMalloc(ptr, bytes);
        if (e != cudaSuccess) {
            VIN_ERROR("[km-gpu] %s failed: %s", what, cudaGetErrorString(e));
            good = false;
        }
    };
    alloc(reinterpret_cast<void**>(&impl_->d_gx), sizeof(float) * P, "cudaMalloc d_gx");
    alloc(reinterpret_cast<void**>(&impl_->d_gy), sizeof(float) * P, "cudaMalloc d_gy");
    alloc(reinterpret_cast<void**>(&impl_->d_sx), sizeof(float) * N, "cudaMalloc d_sx");
    alloc(reinterpret_cast<void**>(&impl_->d_sy), sizeof(float) * N, "cudaMalloc d_sy");
    alloc(reinterpret_cast<void**>(&impl_->d_d2), sizeof(float) * P, "cudaMalloc d_d2");
    alloc(reinterpret_cast<void**>(&impl_->d_labels), sizeof(int) * P, "cudaMalloc d_labels");
    if (cudaMallocHost(reinterpret_cast<void**>(&impl_->h_d2), sizeof(float) * P) != cudaSuccess) {
        VIN_ERROR("[km-gpu] cudaMallocHost h_d2 failed: %s",
                  cudaGetErrorString(cudaGetLastError()));
        good = false;
    }
    if (!good) {
        err_ = "cudaMalloc failed";
        cudaFree(impl_->d_gx); cudaFree(impl_->d_gy);
        cudaFree(impl_->d_sx); cudaFree(impl_->d_sy);
        cudaFree(impl_->d_d2); cudaFree(impl_->d_labels);
        if (impl_->h_d2) { cudaFreeHost(impl_->h_d2); impl_->h_d2 = nullptr; }
        impl_->d_gx = impl_->d_gy = impl_->d_sx = impl_->d_sy = nullptr;
        impl_->d_d2 = nullptr; impl_->d_labels = nullptr;
        return;
    }
    ok_ = true;
    err_ = "ok";
}

KmGpuAssigner::~KmGpuAssigner() {
    if (impl_) {
        cudaFree(impl_->d_gx);
        cudaFree(impl_->d_gy);
        cudaFree(impl_->d_sx);
        cudaFree(impl_->d_sy);
        cudaFree(impl_->d_d2);
        cudaFree(impl_->d_labels);
        if (impl_->h_d2) cudaFreeHost(impl_->h_d2);
        delete impl_;
    }
}

float* KmGpuAssigner::d2_host() {
    return impl_ ? impl_->h_d2 : nullptr;
}

void KmGpuAssigner::upload_coords(const float* gx, const float* gy, int P) {
    cudaMemcpy(impl_->d_gx, gx, sizeof(float) * P, cudaMemcpyHostToDevice);
    cudaMemcpy(impl_->d_gy, gy, sizeof(float) * P, cudaMemcpyHostToDevice);
}

void KmGpuAssigner::assign(const float* sx, const float* sy, int n, int* labels) {
    cudaMemcpy(impl_->d_sx, sx, sizeof(float) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(impl_->d_sy, sy, sizeof(float) * n, cudaMemcpyHostToDevice);
    const int shared_bytes = kKmChunk * sizeof(float2);
    km_assign_kernel<<<grid_for(impl_->P), kBlock, shared_bytes>>>(
        impl_->d_gx, impl_->d_gy, impl_->d_sx, impl_->d_sy,
        impl_->P, n, impl_->d_labels);
    cudaMemcpy(labels, impl_->d_labels, sizeof(int) * impl_->P, cudaMemcpyDeviceToHost);
}

void KmGpuAssigner::d2_init(float sx0, float sy0) {
    km_d2_init_kernel<<<grid_for(impl_->P), kBlock>>>(
        impl_->d_gx, impl_->d_gy, sx0, sy0, impl_->d_d2, impl_->P);
    cudaMemcpy(impl_->h_d2, impl_->d_d2, sizeof(float) * impl_->P, cudaMemcpyDeviceToHost);
}

void KmGpuAssigner::d2_update_download(float sxk, float syk) {
    km_d2_update_kernel<<<grid_for(impl_->P), kBlock>>>(
        impl_->d_gx, impl_->d_gy, sxk, syk, impl_->d_d2, impl_->P);
    cudaMemcpy(impl_->h_d2, impl_->d_d2, sizeof(float) * impl_->P, cudaMemcpyDeviceToHost);
}

} // namespace vinylizer
