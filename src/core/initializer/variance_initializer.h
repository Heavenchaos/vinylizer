#pragma once

#include "core/initializer/base_initializer.h"
#include <opencv2/core.hpp>
#include <random>

namespace vinylizer {

class VarianceInitializer : public BaseInitializer {
public:
    // target_rgb: 目标 RGB 图像 (H, W, 3), CV_8UC3, [0,255]
    // mask_small: 有效区域 mask (H, W), CV_8UC1, 0/255
    // W_opt, H_opt: 优化分辨率
    // detail_first: z-order 排序模式
    VarianceInitializer(const cv::Mat& target_rgb, const cv::Mat& mask_small,
                        int W_opt, int H_opt, bool detail_first);

    void initialize(OptParams& params, int num_template_types,
                    const OptimizerConfig& config,
                    std::mt19937& rng) override;

    ShapeInitParams sample_at(float cx, float cy,
                              std::mt19937& rng) const override;

    bool supports_sample_at() const override { return true; }

    // 设置梯度方向图（度，float32，H_opt x W_opt）。
    // 设置后 initialize/sample_at 的角度将平行于局部边缘（= 梯度方向 + 90°），
    // 而非完全随机，帮助有方向性结构（边缘/纹理）的图像更快收敛。
    void set_grad_dir(const cv::Mat& grad_dir) { grad_dir_ = grad_dir; }

private:
    // 采样位置处的边缘对齐角度（度）。无 grad_dir 或梯度过弱时返回 -1（调用方回退随机）。
    float edge_aligned_angle(int x, int y, std::mt19937& rng) const;

    cv::Mat variance_map_;   // 归一化方差图 (H_opt, W_opt), CV_32FC1, [0,1]
    cv::Mat target_small_;   // 缩放到 opt 分辨率的目标图像 (H_opt, W_opt), CV_8UC3
    cv::Mat grad_dir_;       // 梯度方向图 (H_opt, W_opt), CV_32FC1, 度（可为空）
    std::vector<float> sampling_prob_;  // 扁平化采样概率 [H_opt * W_opt]
    float min_r_ = 2.0f;
    float max_r_ = 128.0f;
    int W_opt_ = 0, H_opt_ = 0;
    bool detail_first_ = true;
    float init_opacity_ = 31.0f / 255.0f;  // 由 initialize() 从 config.init_opacity 缓存，供 sample_at() 复用

    // === 整数 Fenwick 树（无放回加权采样加速，O(P + N log P)）===
    // 把浮点概率乘以 2^32 转为 uint64 整数权重，Fenwick 树维护精确整数前缀和，
    // 避免 std::discrete_distribution 的 O(N*P) 瓶颈。
    // 整数运算满足结合律 → 严格确定性（同 seed → 同结果，数学保证）。
    std::vector<uint64_t> fenwick_tree_;  // 1-indexed, size P+1
    std::vector<uint64_t> fenwick_weights_; // 原始权重副本（用于删除时计算 delta）
    int fenwick_size_ = 0;

    void fenwick_build(const std::vector<float>& probs, int P);
    void fenwick_update(int idx, uint64_t delta);
    uint64_t fenwick_prefix_sum(int idx) const;
    int fenwick_find_kth(uint64_t k) const;  // 找最小的 idx 使 prefix_sum(idx) >= k
    void fenwick_remove(int idx);             // 把 idx 权重置 0
};

} // namespace vinylizer
