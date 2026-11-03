#pragma once

// ============================================================================
// k-medoids 初始化的 GPU 热核（阶段 2b）
//
// 设计原则（保证与 CPU 版逐位一致）：
//   GPU 只做"确定性的选标签 / 选最小距离"这一步；**所有归约与采样都留在 CPU**。
//   - Lloyd：GPU assign_kernel → labels 拷回 host → 沿用 CPU 版 bincount/质心更新
//   - kmeans++：GPU d2_update_kernel → d2 拷回 host → 沿用 CPU 版累积采样
//   因此只要 labels / d2 与 CPU 逐位一致，整个初始化结果必然逐位一致。
//
// 本模块独立分配/释放自己的 device 缓冲，不触碰 DiffRenderer 的任何
// buffer / stream / 状态。
// ============================================================================

#include <cuda_runtime.h>

namespace vinylizer {

// 返回可用 CUDA 设备数（0 = 无可用设备或驱动异常）。
int km_cuda_device_count();

// KmGpuAssigner：一个 init 生命周期内复用的 device 缓冲持有者。
// 构造即分配，析构即释放（用完释放，不常驻）。
class KmGpuAssigner {
public:
    // P: 网格点数（决定 coords/labels/d2 缓冲大小）
    // N: site 数上限（决定 sites 缓冲大小）
    KmGpuAssigner(int P, int N);
    ~KmGpuAssigner();

    KmGpuAssigner(const KmGpuAssigner&) = delete;
    KmGpuAssigner& operator=(const KmGpuAssigner&) = delete;

    bool ok() const { return ok_; }
    const char* error() const { return err_; }

    // 上传 P 个网格点坐标（host float32，各 P 个元素）。每次 init 只需调用一次。
    void upload_coords(const float* gx, const float* gy, int P);

    // 上传前 n 个 site，计算 labels[p] = argmin_k ||c_p - s_k||²（float32），
    // 拷回 host labels（P 个 int）。平手判定用严格 '<' → 最小 site 索引胜，
    // 与 CPU 版 assign_labels 完全一致。
    void assign(const float* sx, const float* sy, int n, int* labels);

    // kmeans++ 用：d2[p] = ||c_p - (sx0,sy0)||²（设备端算，拷回本对象的 pinned 缓冲）。
    void d2_init(float sx0, float sy0);

    // kmeans++ 用：d2[p] = min(d2[p], ||c_p - (sxk,syk)||²)（原地更新，拷回 pinned 缓冲）。
    void d2_update_download(float sxk, float syk);

    // kmeans++ 的 d2 host 缓冲（P floats，pinned）。d2_init 之后有效，
    // 调用方直接读该指针（避免再多一次 pageable 中转拷贝）。
    float* d2_host();

private:
    struct Impl;
    Impl* impl_ = nullptr;
    bool ok_ = false;
    const char* err_ = "uninitialized";
};

} // namespace vinylizer
