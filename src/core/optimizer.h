#pragma once

#include "common/types.h"
#include "core/diff_renderer.h"
#include <opencv2/core.hpp>
#include <vector>
#include <string>
#include <random>
#include <functional>
#include <atomic>
#include <chrono>
#include <tuple>
#include <set>
#include <fstream>
#include <memory>  // std::unique_ptr
#include <cmath>

namespace vinylizer {

// 前向声明
class BaseInitializer;

// ============================================================
// Sigmoid / logit helpers for color/opacity parameterization.
// GPU-resident d_colors / d_opacity (and OptParams color_r/g/b/a) are
// stored in LOGIT space, aligned with diffbmp's sigmoid(tile_v) /
// sigmoid(tile_c) parameterization. ShapeParams (host-facing) keeps
// the legacy linear ranges: color_r/g/b in [0,255], opacity in [0,1].
// Conversions happen at the OptParams <-> ShapeParams boundary.
// ============================================================
inline float vin_sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}
// Clamp input to avoid +/-inf in logit (sigmoid never reaches 0 or 1 exactly).
inline float vin_logit(float p) {
    float pc = std::clamp(p, 1e-5f, 1.0f - 1e-5f);
    return std::log(pc / (1.0f - pc));
}

struct OptimizerConfig {
    int max_shapes = 500;
    int opt_resolution = 128;
    float lr_start = 1.0f;
    float lr_end = 0.1f;
    std::string lr_schedule = "linear";  // "linear", "cosine", "piecewise"
    int lr_warmup_iters = 0;   // --lr-warmup-iters: 前 N 步 lr 从 0 线性爬到 lr_start（0=关，严格 baseline）
    // 初始化策略
    std::string init_mode = "variance";    // "variance" | "kmedoids"
    bool detail_first = true;              // true=东亚风格（小细节在顶层）, false=西方风格
    float var_min_r = 2.0f;         // VarianceInitializer 半径下限（像素）
    float var_max_r = 50.0f;        // VarianceInitializer 半径上限（像素）
    float init_opacity = 31.0f / 255.0f;  // 初始 opacity（linear [0,1]），31/255≈0.12 与历史 baseline 一致
    // k-medoids 构造式几何初始化（init_mode == "kmedoids"）——研究定版参数
    int km_grid = 4;                // 网格步长 g（像素）
    float km_alpha = 5.0f;          // 误差权重指数 w = (0.1 + 0.9*var)^alpha
    float km_radius_scale = 2.5f;   // 半径缩放（CCVT 教训：2.5 使形状必须重叠）
    int km_iters = 15;              // 加权 Lloyd 轮数
    float km_opacity = 0.9f;        // km 家族统一初始 opacity（linear [0,1]）
    // k-medoids assign 核后端："auto"（有可用 CUDA 设备→cuda，否则 cpu）| "cpu" | "cuda"
    // CPU 路径原样保留（对拍基准 + fallback）。
    std::string km_backend = "auto";
    // --init-shapes: 从 optimize 输出 json 加载初始形状继续优化（空=走 initializer，默认关闭等价 baseline）
    std::string init_shapes_path;
    // --init-shapes-mode: continue=one-hot 锁定 json type（断点续训）；reset=type_logits 置 0（真 keep-init，DARTS 重搜）
    std::string init_shapes_mode = "continue";
    int total_iters = 5000;

    // 模板模糊 sigma 调度。默认 2.0 → 0.0 cosine：
    // 早期大模糊帮助建立大结构，后期退到完全锐利（末端精修硬边）。
    // 注意 sigma 是**模板 texel 单位**，映射到像素为 sigma_px = sigma * max(su,sv) / 64，
    // 即每个形状的模糊量正比于它自己的尺寸；sigma=0 时大形状才真正得到锐利边缘。
    // ⚠️ 实测（test.jpg N=1500 5000it）：2.0→0.5 与 1.0 恒定都能到 ~240.6–241.2，
    //    而 2.0→0.0（本默认）约 242.6。退到 0 会失去小形状的抗锯齿，故收益略负；
    //    如需更优，用 --blur-sigma 2.0,0.5 或 GUI 把「终止」设成 0.5。
    // 求值口径：`render --sigma` 必须等于训练末态 sigma 才复现报数（默认已同步为 0.0）。
    float blur_sigma_start = 2.0f;
    float blur_sigma_end = 0.0f;
    int blur_sigma_warmup = 0;
    std::string blur_sigma_schedule = "cosine";

    // DARTS type selection
    bool enable_darts = true;          // enable full-training DARTS type selection
    bool round_robin = false;          // round-robin type assignment (cycles through all stamps, implies no-darts)
    // DARTS 温度与锁定比例（内部常量钉死：温度恒 1.0，lock_ratio=1.0 仅最后一步锁）
    float darts_temp_start = 1.0f;
    float darts_temp_end = 1.0f;
    float darts_lock_ratio = 1.0f;

    // Adam optimizer
    float adam_beta2 = 0.95f;           // 二阶矩衰减系数。0.999 对 reloc/split 后 reset 不友好，0.95 最优
    float max_grad_norm = 30.0f;        // 全局梯度裁剪范数。100 几乎不触发，30 最优

    // 学习率因子（按参数组）。
    // 对齐 diffbmp gains（logit space 下的相对步长）：
    //   diffbmp gains: x/y/r=10.0, angle=1.0, rgb=1.0, alpha=1.5
    float lr_factor_xy = 10.0f;     // cx, cy
    float lr_factor_r = 10.0f;      // rx, ry
    float lr_factor_angle = 1.0f;   // angle
    float lr_factor_rgb = 1.0f;     // color_r, color_g, color_b (logit space)
    float lr_factor_a = 1.5f;       // color_a (opacity logit space)

    // Uniform scale constraint
    bool uniform_scale = false;               // 强制 rx=ry（同比例缩放）

    // Random seed for reproducibility
    int random_seed = 42;

    // R23: 确定性归约（内部钉死 1=两遍 kernel 确定性归约，同 seed bit-exact）
    int deterministic_reduce = 1;

    // RGBA MSE（alpha 通道参与 loss）：true=channels 4（透明区 rendered_alpha→0，FH6 真透明）
    // false=channels 3（alpha 梯度清零）。默认 ON（对齐旧仓库恒定 4 通道行为）
    bool alpha_loss = true;
};

// Optimizer parameters (SoA layout for GPU transfer)
struct OptParams {
    int N = 0;
    std::vector<float> cx, cy, rx, ry, angle;
    std::vector<float> color_r, color_g, color_b, color_a;  // logit space (sigmoid parameterization)
    std::vector<float> type_logits;
    std::vector<float> z;  // soft depth: learnable render-order scalar per shape

    void init(int n, int num_types = 4);
    ShapeParams get_shape(int i) const;
    void set_shape(int i, const ShapeParams& sp);

    // Convert all params to ShapeParams array (for GPU upload)
    std::vector<ShapeParams> to_shape_params() const {
        std::vector<ShapeParams> out(N);
        for (int i = 0; i < N; ++i) out[i] = get_shape(i);
        return out;
    }

    // In-place version: reuses pre-allocated buffer
    void to_shape_params(std::vector<ShapeParams>& out) const {
        out.resize(N);
        for (int i = 0; i < N; ++i) out[i] = get_shape(i);
    }
};

class GradientOptimizer {
public:
    GradientOptimizer(const OptimizerConfig& config, int shape_type = 16);
    ~GradientOptimizer();

    // Main entry: returns optimized shapes
    std::vector<Shape> optimize(
        const cv::Mat& target_rgb,
        const cv::Mat& mask,
        const cv::Mat& edge_map,
        const cv::Mat& saliency_map,
        const cv::Mat& dark_mask,
        const cv::Mat& posterized,
        const std::vector<std::tuple<int,int,int>>& palette,
        const cv::Mat& grad_dir,
        const Color& bg_color = {0, 0, 0, 255});

    void request_stop() { stop_requested_ = true; }

    void set_num_template_types(int n) { num_template_types_ = n; }
    void set_num_builtins(int n) { num_builtins_ = n; }
    void set_external_stamp_names(const std::vector<std::string>& names) {
        external_stamp_names_ = names;
    }
    void set_enable_builtins(bool v) { enable_builtins_ = v; }

    // Progress callback
    using ProgressCallback = std::function<void(float progress, float mse, int iter)>;
    void set_progress_callback(ProgressCallback cb) { progress_cb_ = cb; }

    // Pinned memory for async viz (GPU D2H → glTexSubImage2D in GUI)
    // Double-buffered: GPU alternates writing to A/B, GUI reads the completed one
    void set_viz_target(uint8_t* pinned_a, void* event_a,
                        uint8_t* pinned_b, void* event_b) {
        viz_pinned_[0] = pinned_a;
        viz_pinned_[1] = pinned_b;
        viz_event_[0] = static_cast<cudaEvent_t>(event_a);
        viz_event_[1] = static_cast<cudaEvent_t>(event_b);
        viz_initialized_ = true;
    }
    int viz_read_idx() const { return viz_read_idx_ptr_ ? viz_read_idx_ptr_->load() : -1; }
    void viz_read_done() { if (viz_read_idx_ptr_) viz_read_idx_ptr_->store(-1); }
    void set_viz_read_idx(std::atomic<int>* ptr) { viz_read_idx_ptr_ = ptr; }

private:
    // Initialization
    void shapes_to_params(const std::vector<Shape>& shapes);
    std::vector<Shape> params_to_shapes();

    // Optimization loop
    void gradient_optimize();
    float run_single_iter(OptParams& params,
                          int iter, int total_iters,
                          float* d_grad_history = nullptr, int write_idx = 0,
                          bool local_phase = false,
                          const std::vector<int>& frozen_indices = {},
                          float temperature = 1.0f);

    // Utility
    std::tuple<int,int,int> get_posterized_color(float x, float y);

    OptimizerConfig config_;
    cv::Mat target_rgb_, mask_, edge_map_, saliency_map_, dark_mask_;
    cv::Mat posterized_small_;
    cv::Mat grad_dir_small_;   // 梯度方向图（opt resolution），单位度 [0,360)
    std::vector<std::tuple<int,int,int>> palette_;

    // Optimization resolution
    float scale_ = 1.0f;
    int H_opt_ = 0, W_opt_ = 0;
    cv::Mat target_small_, mask_small_;

    // Renderer
    DiffRenderer renderer_;

    // Initializer (strategy pattern)
    std::unique_ptr<BaseInitializer> initializer_;

    // Current state
    OptParams params_;
    int adam_step_ = 0;
    float current_blur_sigma_ = 2.0f;
    // 当前生效的 opacity tau（供 clamp_opacity 计算 logit 下限）。
    float current_opacity_tau_ = 1.0f;
    std::atomic<bool> stop_requested_{false};
    ProgressCallback progress_cb_;

    // Async viz (non-blocking): GPU D2H → pinned memory, GUI polls with atomic flag
    // Double-buffered: GPU alternates writes, external atomic protects read index
    // Sync strategy: optimizer syncs previous frame's event before queuing next,
    //   so GUI never calls CUDA APIs (no cross-thread cudaEventQuery)
    uint8_t* viz_pinned_[2] = {nullptr, nullptr};
    cudaEvent_t viz_event_[2] = {nullptr, nullptr};
    int viz_write_idx_ = 0;
    std::atomic<int>* viz_read_idx_ptr_ = nullptr;
    std::chrono::steady_clock::time_point viz_last_time_;
    bool viz_initialized_ = false;
    bool viz_first_frame_ = true;
    int viz_synced_idx_ = -1;

    // Background color for canvas rendering
    Color bg_color_{0, 0, 0, 255};

    // Pre-allocated reusable buffers (avoid per-iter heap allocation)
    std::vector<ShapeParams> cached_shape_params_;

    // Random (seed from config)
    std::mt19937 rng_;

    // Shape type (228=soft_circle, 16=hard_ellipse, 1=rectangle, 103=triangle)
    int shape_type_ = 16;

    // Template type count and external stamp names
    int num_template_types_ = 4;
    int num_builtins_ = 4;
    std::vector<std::string> external_stamp_names_;
    bool enable_builtins_ = true;
};

} // namespace vinylizer
