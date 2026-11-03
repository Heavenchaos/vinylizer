#pragma once

#include "common/types.h"
#include <string>
#include <vector>
#include <atomic>
#include <functional>
#include <algorithm>
#include <cmath>
#include <opencv2/core.hpp>

namespace vinylizer {

class GradientOptimizer;

struct PipelineConfig {
    std::string input_path;
    std::string output_path;
    int num_shapes = 500;
    int canvas_w = 512;
    int canvas_h = 512;
    int total_iters = 5000;
    // 学习率退火：默认 1.0 → 0.0 cosine。
    // 承重项：Adam 在 bias-correction 满步长期步长≈lr·sign(g)（与梯度幅值无关），
    // lr 恒定不退火 ⇒ 参数在最优点附近常驻随机游走，loss 停在一个 ∝ lr² 的地板上
    // （实测纯色场：lr=1.0 恒定残差 28.6，退火到 0 后 2e-4）。不要改回恒定。
    float lr_start = 1.0f;
    float lr_end = 0.0f;
    std::string lr_schedule = "cosine";
    int lr_warmup_iters = 0;   // --lr-warmup-iters: 前 N 步 lr 从 0 线性爬到 lr_start（0=关）
    // 初始化策略（产品默认：构造式几何初始化 kmedoids）
    std::string init_mode = "kmedoids";    // "variance" | "kmedoids"
    bool detail_first = true;              // true=东亚风格, false=西方风格
    float var_min_r = 2.0f;                // VarianceInitializer 半径下限（像素）
    float var_max_r = 50.0f;               // VarianceInitializer 半径上限（像素）
    float init_opacity = 31.0f / 255.0f;   // 初始 opacity（linear [0,1]）
    // k-medoids 构造式几何初始化（研究定版参数，见 kmedoids_initializer.h）
    int km_grid = 4;
    float km_alpha = 5.0f;
    float km_radius_scale = 2.5f;
    int km_iters = 15;
    float km_opacity = 0.9f;
    // k-medoids assign 核后端："auto" | "cpu" | "cuda"（见 OptimizerConfig::km_backend）
    std::string km_backend = "auto";
    // --init-shapes: 从 optimize 输出 json 加载初始形状继续优化（空=走 initializer，默认关闭等价 baseline）
    std::string init_shapes_path;
    // --init-shapes-mode: continue=one-hot 锁定 json type（断点续训，默认）；reset=type_logits 置 0（真 keep-init，DARTS 重搜）
    std::string init_shapes_mode = "continue";
    int shape_type = 16;
    bool enable_darts = true;
    bool round_robin = false;
    // 透明 padding（防止形状超出画布边界）。
    // boundary_pad >= 0：显式绝对像素覆盖（CLI --pad，0=完全关闭）；
    // boundary_pad < 0：按图像尺寸比例自动（默认，见 effective_boundary_pad）。
    int boundary_pad = -1;
    // 自动模式：pad = clamp(round(max(W,H) * pad_ratio), pad_min, pad_max)。
    // 边框面积占比 = 1 - 1/(1+2c)^2（c = pad/max(W,H)）与图像尺寸无关；
    // 默认 0.053 让 test1（1186x1884）恰好保持旧的固定 100px，大图行为不变、小图自动收缩。
    float pad_ratio = 0.053f;
    int pad_min = 16;
    int pad_max = 400;
    // RGBA MSE（alpha 通道参与 loss）：驱动透明区 rendered_alpha→0，使形状 opacity 学成 0
    // → FH6 真透明。默认 ON（对齐旧仓库恒定 4 通道行为）
    bool alpha_loss = true;

    // 模板模糊 sigma 调度。默认 2.0 → 0.0 cosine（早期大模糊建结构，末期锐利精修）。
    // sigma 为**模板 texel 单位**，映射到像素 sigma_px = sigma * max(su,sv) / 64，
    // 即模糊量正比于形状尺寸；sigma=0 时大形状才真正锐利。
    // ⚠️ 实测：2.0→0.5 与 1.0 恒定 ≈240.6–241.2，2.0→0.0 ≈242.6（退到 0 损小形状抗锯齿）。
    // 求值口径：`render --sigma` 需等于训练末态 sigma（默认已同步为 0.0）。
    float blur_sigma_start = 2.0f;
    float blur_sigma_end = 0.0f;
    int blur_sigma_warmup = 0;
    std::string blur_sigma_schedule = "cosine";

    // DARTS type selection（内部常量钉死：温度恒 1.0，锁定比例 1.0=仅最后一步锁）
    float darts_temp_start = 1.0f;
    float darts_temp_end = 1.0f;
    float darts_lock_ratio = 1.0f;

    // Adam optimizer
    float adam_beta2 = 0.95f;          // 0.999 对 reset 后不友好，0.95 最优
    float max_grad_norm = 30.0f;       // 100 几乎不触发，30 最优

    // Uniform scale constraint
    bool uniform_scale = false;                // 强制 rx=ry（同比例缩放）

    // Random seed for reproducibility
    int random_seed = 42;

    // R23: 确定性归约（内部钉死 1=两遍 kernel 确定性归约，同 seed bit-exact）
    int deterministic_reduce = 1;

    // 学习率因子（按参数组），默认全 1（行为不变）
    float lr_factor_xy = 1.0f;
    float lr_factor_r = 1.0f;
    float lr_factor_angle = 1.0f;
    float lr_factor_rgb = 1.0f;
    float lr_factor_a = 1.0f;

    // 外部图章
    std::vector<std::string> stamp_files;
    bool enable_builtins = true;
    int num_template_types = 4;
    int num_builtins = 4;  // 0 = pure external stamps, 4 = with builtins

    mutable std::atomic<bool> stop_requested{false};

    // Pinned memory for GPU visualisation output (cudaMallocHost by GUI)
    // Double-buffered: GPU alternates writing to A/B, GUI reads from the completed one
    uint8_t* viz_pinned_a = nullptr;
    void* viz_event_a = nullptr;
    uint8_t* viz_pinned_b = nullptr;
    void* viz_event_b = nullptr;

    // Shared atomic: optimizer sets index (0/1) when frame is queued; GUI reads & resets to -1
    std::atomic<int>* viz_read_idx_atomic = nullptr;

    PipelineConfig() = default;
    PipelineConfig(const PipelineConfig& other)
        : input_path(other.input_path), output_path(other.output_path),
          num_shapes(other.num_shapes), canvas_w(other.canvas_w), canvas_h(other.canvas_h),
          total_iters(other.total_iters), lr_start(other.lr_start), lr_end(other.lr_end),
          lr_schedule(other.lr_schedule),
          lr_warmup_iters(other.lr_warmup_iters),
          init_mode(other.init_mode),
          detail_first(other.detail_first),
          var_min_r(other.var_min_r),
          var_max_r(other.var_max_r),
          init_opacity(other.init_opacity),
          km_grid(other.km_grid),
          km_alpha(other.km_alpha),
          km_radius_scale(other.km_radius_scale),
          km_iters(other.km_iters),
          km_opacity(other.km_opacity),
          km_backend(other.km_backend),
          init_shapes_path(other.init_shapes_path),
          init_shapes_mode(other.init_shapes_mode),
          shape_type(other.shape_type), enable_darts(other.enable_darts),
          round_robin(other.round_robin),
          boundary_pad(other.boundary_pad),
          pad_ratio(other.pad_ratio),
          pad_min(other.pad_min),
          pad_max(other.pad_max),
          alpha_loss(other.alpha_loss),
          blur_sigma_start(other.blur_sigma_start), blur_sigma_end(other.blur_sigma_end),
          blur_sigma_warmup(other.blur_sigma_warmup),
          blur_sigma_schedule(other.blur_sigma_schedule),
          darts_temp_start(other.darts_temp_start),
          darts_temp_end(other.darts_temp_end),
          darts_lock_ratio(other.darts_lock_ratio),
          adam_beta2(other.adam_beta2),
          max_grad_norm(other.max_grad_norm),
          uniform_scale(other.uniform_scale),
          random_seed(other.random_seed),
          deterministic_reduce(other.deterministic_reduce),
          lr_factor_xy(other.lr_factor_xy),
          lr_factor_r(other.lr_factor_r),
          lr_factor_angle(other.lr_factor_angle),
          lr_factor_rgb(other.lr_factor_rgb),
          lr_factor_a(other.lr_factor_a),
          stamp_files(other.stamp_files),
          enable_builtins(other.enable_builtins),
          num_template_types(other.num_template_types),
          num_builtins(other.num_builtins),
          stop_requested(false),
          // GPU→GUI 可视化双缓冲（裸指针/句柄，跨线程共享所有权不在本结构）
          viz_pinned_a(other.viz_pinned_a), viz_event_a(other.viz_event_a),
          viz_pinned_b(other.viz_pinned_b), viz_event_b(other.viz_event_b),
          viz_read_idx_atomic(other.viz_read_idx_atomic) {}
};

// 共享辅助函数：计算有效的 boundary padding。
// boundary_pad >= 0：显式绝对像素覆盖（CLI --pad，0=完全关闭）；
// boundary_pad < 0：按图像尺寸比例自动 clamp(round(max(W,H) * pad_ratio), pad_min, pad_max)。
inline int effective_boundary_pad(int W, int H, int boundary_pad,
                                  float pad_ratio, int pad_min, int pad_max) {
    if (boundary_pad >= 0) return boundary_pad;
    int pad = static_cast<int>(std::lround(static_cast<double>(std::max(W, H)) * pad_ratio));
    if (pad < pad_min) pad = pad_min;
    if (pad > pad_max) pad = pad_max;
    return pad;
}

inline int effective_boundary_pad(const PipelineConfig& config, int W, int H) {
    return effective_boundary_pad(W, H, config.boundary_pad, config.pad_ratio,
                                  config.pad_min, config.pad_max);
}

void run_pipeline(const PipelineConfig& config,
                  std::function<void(float,float,int)> progress_cb = nullptr,
                  std::function<void(GradientOptimizer*)> optimizer_ready_cb = nullptr);

// === Render-only mode ===
// 从 shapes.json 加载形状（optimize 输出格式），用 DiffRenderer 静态渲染。
// 用于形状移除敏感性分析：移除指定形状后渲染，计算 PSNR。
struct RenderOnlyConfig {
    std::string shapes_json_path;   // shapes.json 路径（optimize 输出格式）
    std::string target_raw_path;    // target.raw 路径（用于计算 PSNR，与 target_png_path 二选一）
    std::string target_png_path;    // target PNG 路径（与 target_raw_path 二选一，INTER_AREA resize 与 optimize 口径一致）
    std::string mask_raw_path;      // mask.raw 路径（用于有效像素 PSNR）
    std::string output_raw_path;    // 输出 rendered.raw 路径（可选）
    std::vector<std::string> stamp_files;  // 外部图章路径
    int num_template_types = 4;
    int num_builtins = 0;            // 0 = pure external stamps
    std::vector<int> remove_indices; // 要移除的形状索引（opacity 强制为 0）
    float alpha_upper_bound = 1.0f;  // alpha cap（须与训练一致，1.0=无 cap）
    // 模板模糊 sigma；必须与训练末态一致（1.0）才能复现 optimize 报告的 MSE。
    float blur_sigma = 1.0f;
    bool verbose = true;
};

struct RenderOnlyResult {
    int H = 0, W = 0, N = 0, N_used = 0;
    float psnr_db = 0.0f;            // 相对 target 的 PSNR（mask 内）
    float mse_0_255 = 0.0f;
    bool ok = false;
};

RenderOnlyResult render_only(const RenderOnlyConfig& config);

} // namespace vinylizer
