#pragma once

// 页 1「矢量化」的可调参数，以及唯一的 PipelineConfig 构造入口。
// GUI（optimize_page）与 --autotest 都走 build_pipeline_config()，保证
// 界面改参数 → 引擎收到的参数完全一致（不留第二份构造代码）。

#include "core/pipeline.h"
#include <algorithm>
#include <string>

namespace vinylizer {

// 透明 padding：GUI 不暴露绝对像素，默认按图像尺寸比例（与 CLI 的 effective_boundary_pad
// 共用同一口径）。默认 0.053 让 test1 级别大图 ≈100px，小图自动收缩，边框占比恒定；
// build_pipeline_config 与预览缓冲尺寸计算共用这些常量，避免两处漂移。
constexpr float kGuiPadRatio = 0.053f;
constexpr int kGuiPadMin = 16;
constexpr int kGuiPadMax = 400;

// 形状模板策略（面向玩家：默认自动四型搜索）
enum class TemplateMode {
    Auto = 0,     // 自动选择模板（四型搜索）
    SoftCircle,   // 软圆 228
    Ellipse,      // 椭圆 16
    Rect,         // 矩形 1
    Triangle,     // 三角 103
};

struct OptimizeParams {
    std::string input_path;
    int canvas_w = 0;             // = 原图宽（保持原生分辨率，只缩不放由 opt_resolution 控制）
    int canvas_h = 0;
    int num_shapes = 1500;        // 玩家手动输入；无预设档位
    TemplateMode template_mode = TemplateMode::Auto;
    std::string init_mode = "kmedoids";   // 智能布局初始化
    float km_alpha = 5.0f;                // 细节强度 α
    int km_grid = 4;                      // 网格
    float km_radius_scale = 2.5f;         // 半径系数
    int km_iters = 15;                    // Lloyd 迭代
    std::string km_backend = "auto";      // 计算后端
    int random_seed = 42;
    int total_iters = 5000;
    // 学习率退火（高级选项）。默认 1.0→0.0 + cosine：
    // Adam 的 bias-correction 满步长期步长≈lr·sign(g)，与梯度幅值无关；若 lr 恒定
    // 不衰减，参数在最优点附近做常驻随机游走，loss 会在"地板"上停住、末端无法精修
    // （实测纯色场残差 ∝ lr²；恒定 lr=1.0 残差 28.6，退火后 2e-4）。
    float lr_start = 1.0f;                // 起始学习率
    float lr_end = 0.0f;                  // 终止学习率（0.0 = 退火到零）
    std::string lr_schedule = "cosine";   // "cosine" | "linear" | "piecewise"
    int lr_warmup_iters = 0;              // 前 N 步从 0 线性爬到 lr_start（0=关；续训/微调时建议 5）
    // 模板模糊 sigma 退火（高级选项）。sigma 为**模板 texel 单位**，映射到像素是
    // sigma_px = sigma * max(su,sv) / 64 —— 即每个形状的模糊量正比于它自己的尺寸。
    // 默认 2.0 → 0.0 cosine：早期大模糊建结构、末期完全锐利。
    // ⚠️ 实测 test.jpg N=1500 5000it：2.0→0.5 与 1.0 恒定 ≈240.6–241.2，
    //    2.0→0.0 ≈242.6（退到 0 会失去小形状的抗锯齿）——想更优可把「终止」设 0.5。
    float blur_sigma_start = 2.0f;
    float blur_sigma_end = 0.0f;
    std::string blur_sigma_schedule = "cosine";
    int blur_sigma_warmup = 0;
    // DARTS 软/硬渲染切换点（进度比例）。1.0=只在最后一个迭代切硬渲染（默认）；
    // 0.0=全程硬渲染搜索。实测 0.4 与 1.0 统计上无差别，0.7 最差。
    float darts_lock_ratio = 1.0f;
};

inline void template_mode_to_engine(TemplateMode m, bool& enable_darts, int& shape_type) {
    switch (m) {
        case TemplateMode::Auto:       enable_darts = true;  shape_type = 16;  break;
        case TemplateMode::SoftCircle: enable_darts = false; shape_type = 228; break;
        case TemplateMode::Ellipse:    enable_darts = false; shape_type = 16;  break;
        case TemplateMode::Rect:       enable_darts = false; shape_type = 1;   break;
        case TemplateMode::Triangle:   enable_darts = false; shape_type = 103; break;
    }
}

// 唯一的产品默认口径：与 CLI 的默认参数等价（无外部图章 / 四内置模板 /
// lr 1.0→0.0 cosine 退火 / 透明 padding 按尺寸比例 / deterministic reduce 内部钉死 1）。
inline PipelineConfig build_pipeline_config(const OptimizeParams& p,
                                            const std::string& output_path) {
    PipelineConfig c;
    c.input_path = p.input_path;
    c.output_path = output_path;
    c.num_shapes = p.num_shapes;
    c.canvas_w = p.canvas_w > 0 ? p.canvas_w : 9999;
    c.canvas_h = p.canvas_h > 0 ? p.canvas_h : 9999;
    c.total_iters = p.total_iters;
    c.lr_start = p.lr_start;
    c.lr_end = p.lr_end;
    c.lr_schedule = p.lr_schedule;
    c.lr_warmup_iters = p.lr_warmup_iters;
    c.blur_sigma_start = p.blur_sigma_start;
    c.blur_sigma_end = p.blur_sigma_end;
    c.blur_sigma_schedule = p.blur_sigma_schedule;
    c.blur_sigma_warmup = p.blur_sigma_warmup;
    c.darts_lock_ratio = p.darts_lock_ratio;
    c.boundary_pad = -1;   // -1 = 按比例（防止形状超出画布边界），pad_ratio 见下
    c.pad_ratio = kGuiPadRatio;
    c.pad_min = kGuiPadMin;
    c.pad_max = kGuiPadMax;
    c.alpha_loss = true;   // RGBA MSE：透明区保持透明（玩家界面不暴露，恒定 ON）
    template_mode_to_engine(p.template_mode, c.enable_darts, c.shape_type);
    c.init_mode = p.init_mode;
    c.km_alpha = p.km_alpha;
    c.km_grid = p.km_grid;
    c.km_radius_scale = p.km_radius_scale;
    c.km_iters = p.km_iters;
    c.km_backend = p.km_backend;
    c.random_seed = p.random_seed;
    c.enable_builtins = true;
    c.num_builtins = 4;
    c.num_template_types = 4;
    return c;
}

// 优化器实际使用的画布尺寸（含透明 padding）以及内容区在该画布上的位置/尺寸。
// 必须与 pipeline.cpp（先给目标图加 boundary_pad）→ optimizer.cpp（opt_resolution
// 取 max(w,h,画布边) 定 scale）口径完全一致：实时预览的 pinned 缓冲要按 *画布*
// 尺寸分配（D2H 整块写入），而纹理只显示内容区（用 UNPACK_SKIP_* 取子矩形）。
inline void optimizer_canvas_view(const OptimizeParams& p,
                                  int& canvas_w, int& canvas_h,
                                  int& view_x, int& view_y, int& view_w, int& view_h) {
    const int cw = p.canvas_w > 0 ? p.canvas_w : 9999;
    const int ch = p.canvas_h > 0 ? p.canvas_h : 9999;
    const int img_w = std::max(4, cw);
    const int img_h = std::max(4, ch);
    // 与 pipeline.cpp 完全同一公式（GUI 走按比例；p.canvas_w/h = 原图尺寸）
    const int pad = effective_boundary_pad(img_w, img_h, -1,
                                           kGuiPadRatio, kGuiPadMin, kGuiPadMax);
    const int W_orig = img_w + 2 * pad;   // pipeline.cpp: copyMakeBorder 后的尺寸
    const int H_orig = img_h + 2 * pad;
    const int opt_res = std::max(std::max(cw, ch), std::max(W_orig, H_orig));
    const int max_dim = std::max(W_orig, H_orig);
    const float scale = std::min(1.0f, static_cast<float>(opt_res) / static_cast<float>(max_dim));
    canvas_w = std::max(4, static_cast<int>(W_orig * scale));
    canvas_h = std::max(4, static_cast<int>(H_orig * scale));
    view_x = std::min(canvas_w - 1, static_cast<int>(pad * scale));
    view_y = std::min(canvas_h - 1, static_cast<int>(pad * scale));
    view_w = std::max(1, std::min(canvas_w - view_x, static_cast<int>(img_w * scale)));
    view_h = std::max(1, std::min(canvas_h - view_y, static_cast<int>(img_h * scale)));
}

} // namespace vinylizer
