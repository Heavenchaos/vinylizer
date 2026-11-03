#include "core/initializer/variance_initializer.h"
#include "common/logger.h"

#include <opencv2/imgproc.hpp>
#include <cmath>
#include <random>
#include <numeric>
#include <cstdint>

namespace vinylizer {

VarianceInitializer::VarianceInitializer(
    const cv::Mat& target_rgb, const cv::Mat& mask_small,
    int W_opt, int H_opt, bool detail_first)
    : W_opt_(W_opt), H_opt_(H_opt), detail_first_(detail_first)
{
    // ── 1. 缩放目标图像到优化分辨率 ──
    cv::resize(target_rgb, target_small_, {W_opt, H_opt}, 0, 0, cv::INTER_AREA);

    // ── 2. 计算 RGB 三通道局部方差图 ──
    const int window_size = 7;
    cv::Mat target_float;
    target_small_.convertTo(target_float, CV_32FC3, 1.0 / 255.0);  // [0,1]

    variance_map_ = cv::Mat::zeros(H_opt, W_opt, CV_32FC1);

    for (int ch = 0; ch < 3; ++ch) {
        cv::Mat channel(H_opt, W_opt, CV_32FC1);
        cv::extractChannel(target_float, channel, ch);

        cv::Mat local_mean, local_mean_sq;
        cv::boxFilter(channel, local_mean, -1, {window_size, window_size}, {-1,-1}, true);
        cv::boxFilter(channel.mul(channel), local_mean_sq, -1, {window_size, window_size}, {-1,-1}, true);

        // Variance = E[X²] - E[X]²
        cv::Mat ch_var = local_mean_sq - local_mean.mul(local_mean);
        cv::add(variance_map_, ch_var, variance_map_);
    }

    // ── 3. 归一化方差图到 [0,1] ──
    double vmin, vmax;
    cv::minMaxLoc(variance_map_, &vmin, &vmax);
    if (vmax > 0) {
        variance_map_ = variance_map_ / static_cast<float>(vmax);
    }
    // mask 掉无效区域（padding 黑边）
    if (!mask_small.empty()) {
        variance_map_.setTo(0, ~mask_small);
    }
    VIN_INFO("Variance map: min=%.4f, max=%.4f, mean=%.4f",
             vmin / std::max(vmax, 1.0), 1.0, cv::mean(variance_map_)[0]);

    // ── 4. 计算采样概率 ──
    const float base_prob = 0.1f;
    int total = H_opt * W_opt;
    sampling_prob_.resize(total);

    for (int y = 0; y < H_opt; ++y) {
        for (int x = 0; x < W_opt; ++x) {
            float var = variance_map_.at<float>(y, x);
            sampling_prob_[y * W_opt + x] = base_prob + (1.0f - base_prob) * var;
        }
    }
    float prob_sum = std::accumulate(sampling_prob_.begin(), sampling_prob_.end(), 0.0f);
    if (prob_sum > 0) {
        for (auto& p : sampling_prob_) p /= prob_sum;
    }

    // ── 5. 计算半径范围（对齐 diffbmp std_r_init） ──
    // 注意：尝试过 r_min=4 + r_max=min(W,H)/4，导致性能退化 (21s→90s)
    // 且 PSNR 下降，因此只对齐 std_r_init=(2.0, 50.0)，r_max 限制为画布短边 1/2
    min_r_ = 2.0f;
    max_r_ = 50.0f;
    // 上限保护：不超过画布短边的一半
    max_r_ = std::min(max_r_, std::min(W_opt, H_opt) / 2.0f);
    if (max_r_ <= min_r_) max_r_ = min_r_ + 0.1f;
    VIN_INFO("VarianceInitializer: min_r=%.1f, max_r=%.1f", min_r_, max_r_);
}

void VarianceInitializer::initialize(OptParams& params, int /*num_template_types*/,
                                      const OptimizerConfig& config,
                                      std::mt19937& rng)
{
    // 允许通过 config 覆盖半径范围（半径消融实验：缩小初始形状以降低覆盖深度）
    min_r_ = config.var_min_r;
    max_r_ = config.var_max_r;
    max_r_ = std::min(max_r_, std::min(W_opt_, H_opt_) / 2.0f);
    if (max_r_ <= min_r_) max_r_ = min_r_ + 0.1f;
    init_opacity_ = config.init_opacity;
    VIN_INFO("VarianceInitializer (override): min_r=%.1f, max_r=%.1f, init_opacity=%.4f", min_r_, max_r_, init_opacity_);

    int N = params.N;
    std::uniform_real_distribution<float> dist_angle(0.0f, 360.0f);
    std::normal_distribution<float> color_noise(0.0f, 0.02f);   // std_c_init = 0.02
    std::normal_distribution<float> radius_noise(0.0f, 0.1f * (max_r_ - min_r_));

    // ── 无放回加权采样位置（整数 Fenwick 树，O(P + N log P)）──
    // 整数运算满足结合律 → 严格确定性（同 seed → 同结果）。
    struct ShapeTemp {
        int idx; float cx, cy, rx, angle, cr, cg, cb, var;
    };
    std::vector<ShapeTemp> temps(N);

    int total_pixels = W_opt_ * H_opt_;
    fenwick_build(sampling_prob_, total_pixels);

    // 64 位均匀随机数生成器，映射到 [0, total_sum)
    std::uniform_int_distribution<uint64_t> dist_u64;

    for (int i = 0; i < N; ++i) {
        // 获取当前总权重（Fenwick 树根 = prefix_sum(P)）
        uint64_t total_sum = fenwick_prefix_sum(fenwick_size_);
        if (total_sum == 0) {
            VIN_WARN("VarianceInitializer: all pixels exhausted at i=%d/%d", i, N);
            break;
        }

        // 生成 u ∈ [0, total_sum)
        uint64_t u = dist_u64(rng) % total_sum;  // 等价于 u ∈ [0, total_sum)

        // 在 Fenwick 树上找第 u+1 个元素（1-indexed kth）
        int flat_idx = fenwick_find_kth(u + 1);  // +1 转为 1-indexed

        // 从树中移除该像素（权重置 0）
        fenwick_remove(flat_idx);

        int y = flat_idx / W_opt_, x = flat_idx % W_opt_;

        // 位置
        temps[i].cx = static_cast<float>(x);
        temps[i].cy = static_cast<float>(y);

        // 方差→半径反比
        float var = variance_map_.at<float>(y, x);
        temps[i].var = var;
        float r = max_r_ - var * (max_r_ - min_r_);
        r += radius_noise(rng);
        r = std::clamp(r, min_r_, max_r_);
        temps[i].rx = r;

        // 角度：优先边缘对齐（平行局部边缘），无结构时随机
        float ea = edge_aligned_angle(x, y, rng);
        temps[i].angle = (ea >= 0.0f) ? ea : dist_angle(rng);

        // 颜色：精确像素 + 高斯噪声
        auto& pixel = target_small_.at<cv::Vec3b>(y, x);
        float cr = pixel[0] / 255.0f + color_noise(rng);
        float cg = pixel[1] / 255.0f + color_noise(rng);
        float cb = pixel[2] / 255.0f + color_noise(rng);
        temps[i].cr = std::clamp(cr, 0.0f, 1.0f) * 255.0f;
        temps[i].cg = std::clamp(cg, 0.0f, 1.0f) * 255.0f;
        temps[i].cb = std::clamp(cb, 0.0f, 1.0f) * 255.0f;
        temps[i].idx = i;
    }

    // ── z-order 排序 ──
    if (detail_first_) {
        // 升序：低方差（大半径）在前（底层），高方差（小半径）在后（顶层）
        std::sort(temps.begin(), temps.end(),
            [](const ShapeTemp& a, const ShapeTemp& b) { return a.var < b.var; });
    } else {
        // 降序
        std::sort(temps.begin(), temps.end(),
            [](const ShapeTemp& a, const ShapeTemp& b) { return a.var > b.var; });
    }

    // ── 写入 OptParams ──
    // Logit parameterization: color_r/g/b/a are logits (sigmoid space).
    // Initial opacity 由 --init-opacity 控制（init_opacity_ 已由 initialize 从 config 缓存）
    const float init_opacity_logit = vin_logit(init_opacity_);
    for (int i = 0; i < N; ++i) {
        params.cx[i] = temps[i].cx;
        params.cy[i] = temps[i].cy;
        params.rx[i] = temps[i].rx;
        params.ry[i] = temps[i].rx;   // rx == ry
        params.angle[i] = temps[i].angle;
        // temps[i].cr/cg/cb are linear [0,255]; convert to logit
        params.color_r[i] = vin_logit(temps[i].cr / 255.0f);
        params.color_g[i] = vin_logit(temps[i].cg / 255.0f);
        params.color_b[i] = vin_logit(temps[i].cb / 255.0f);
        params.color_a[i] = init_opacity_logit;
    }

    VIN_INFO("VarianceInitializer: initialized %d shapes (detail_first=%d)",
             N, static_cast<int>(detail_first_));
}

ShapeInitParams VarianceInitializer::sample_at(float cx, float cy,
                                                std::mt19937& rng) const
{
    ShapeInitParams p;

    int ix = std::clamp(static_cast<int>(cx), 0, W_opt_ - 1);
    int iy = std::clamp(static_cast<int>(cy), 0, H_opt_ - 1);

    // 方差→半径反比
    float var = variance_map_.at<float>(iy, ix);
    float r = max_r_ - var * (max_r_ - min_r_);

    std::normal_distribution<float> radius_noise(0.0f, 0.1f * (max_r_ - min_r_));
    r += radius_noise(rng);
    r = std::clamp(r, min_r_, max_r_);
    p.rx = r;
    p.ry = r;

    // 角度：优先边缘对齐（平行局部边缘），无结构时随机
    std::uniform_real_distribution<float> dist_angle(0.0f, 360.0f);
    float ea = edge_aligned_angle(ix, iy, rng);
    p.angle = (ea >= 0.0f) ? ea : dist_angle(rng);

    // 颜色：精确像素 + 高斯噪声，转换为 logit
    std::normal_distribution<float> color_noise(0.0f, 0.02f);
    auto& pixel = target_small_.at<cv::Vec3b>(iy, ix);
    float cr = pixel[0] / 255.0f + color_noise(rng);
    float cg = pixel[1] / 255.0f + color_noise(rng);
    float cb = pixel[2] / 255.0f + color_noise(rng);
    // Store as logit (sigmoid space) to match OptParams parameterization
    p.r = vin_logit(std::clamp(cr, 0.0f, 1.0f));
    p.g = vin_logit(std::clamp(cg, 0.0f, 1.0f));
    p.b = vin_logit(std::clamp(cb, 0.0f, 1.0f));

    // 透明度 logit: 由 --init-opacity 控制（init_opacity_ 由 initialize 缓存）
    p.a = vin_logit(init_opacity_);

    return p;
}

float VarianceInitializer::edge_aligned_angle(int x, int y, std::mt19937& rng) const {
    // 无梯度方向图 → 回退随机
    if (grad_dir_.empty()) return -1.0f;

    // 方差 = 局部结构强度代理。结构太弱（平坦区）时梯度方向不可靠，回退随机。
    float var = variance_map_.at<float>(y, x);
    if (var < 0.05f) return -1.0f;

    // 边缘方向 = 梯度方向 + 90°（形状长轴平行于边缘，跨越均匀区域而非边缘）
    float g = grad_dir_.at<float>(y, x);

    // 加 ±20° 噪声，避免所有形状角度完全一致导致的僵硬布局
    std::uniform_real_distribution<float> noise(-20.0f, 20.0f);
    float angle = g + 90.0f + noise(rng);

    // 归一化到 [0, 360)
    angle = std::fmod(angle, 360.0f);
    if (angle < 0.0f) angle += 360.0f;
    return angle;
}

// ── 整数 Fenwick 树实现 ──
// 树内部 1-indexed（fenwick_tree_ 大小 P+1，索引 1..P）。
// 对外接口（idx 参数）统一 0-based，与 sampling_prob_/fenwick_weights_ 一致。

void VarianceInitializer::fenwick_build(const std::vector<float>& probs, int P) {
    fenwick_size_ = P;
    fenwick_tree_.assign(P + 1, 0);
    fenwick_weights_.assign(P, 0);

    // 浮点概率 → 整数权重（乘以 2^32 保留精度）
    // 整数运算满足结合律 → 同 seed 同结果（严格确定性）
    const uint64_t scale = 1ULL << 32;
    for (int i = 0; i < P; ++i) {
        fenwick_weights_[i] = static_cast<uint64_t>(probs[i] * scale);
    }

    // O(P) 直接建树：每个节点累加自身权重后向父节点推送
    for (int i = 1; i <= P; ++i) {
        fenwick_tree_[i] += fenwick_weights_[i - 1];
        int parent = i + (i & -i);
        if (parent <= P) {
            fenwick_tree_[parent] += fenwick_tree_[i];
        }
    }
}

void VarianceInitializer::fenwick_update(int idx, uint64_t delta) {
    // idx: 0-based → 转 1-indexed
    int idx1 = idx + 1;
    for (; idx1 <= fenwick_size_; idx1 += idx1 & -idx1) {
        fenwick_tree_[idx1] += delta;
    }
}

uint64_t VarianceInitializer::fenwick_prefix_sum(int idx) const {
    // idx: 1-indexed（fenwick_size_ 表示求整个前缀和）
    uint64_t sum = 0;
    for (; idx > 0; idx -= idx & -idx) {
        sum += fenwick_tree_[idx];
    }
    return sum;
}

int VarianceInitializer::fenwick_find_kth(uint64_t k) const {
    // k: 1-indexed（第 k 个元素）
    // 返回: 0-based flat_idx
    // 找最大的 idx（1-indexed）使 prefix_sum(idx) < k，则 idx 即为 0-based 答案
    int idx = 0;
    // 找不超过 fenwick_size_ 的最大 2 的幂
    int bit = 1;
    while (bit * 2 <= fenwick_size_) bit <<= 1;
    for (; bit > 0; bit >>= 1) {
        int next = idx + bit;
        if (next <= fenwick_size_ && fenwick_tree_[next] < k) {
            idx = next;
            k -= fenwick_tree_[next];
        }
    }
    return idx;  // 0-based
}

void VarianceInitializer::fenwick_remove(int idx) {
    // idx: 0-based
    int idx1 = idx + 1;
    uint64_t w = fenwick_weights_[idx];
    if (w == 0) return;  // 已被移除
    fenwick_weights_[idx] = 0;
    // 沿祖先链减去 w
    for (; idx1 <= fenwick_size_; idx1 += idx1 & -idx1) {
        fenwick_tree_[idx1] -= w;
    }
}

} // namespace vinylizer
