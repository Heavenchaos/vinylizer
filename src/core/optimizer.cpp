#include "core/optimizer.h"
#include "common/logger.h"
#include "common/canvas.h"
#include "common/path_utils.h"
#include "core/initializer/variance_initializer.h"
#include "core/initializer/kmedoids_initializer.h"

#include <opencv2/imgproc.hpp>
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <string>
#include <fstream>
#include <filesystem>
#include "json.hpp"

// 方向3：opacity 温度 τ 设置（定义在 template_sampling.cu，用 __constant__ memory）
extern "C" void cuda_set_opacity_tau(float tau);

// alpha_upper_bound 设置（对齐 diffbmp，定义在 template_sampling.cu，用 __constant__ memory）
extern "C" void cuda_set_alpha_upper_bound(float aub);

namespace vinylizer {

// ================================================================
// OptParams
// ================================================================

void OptParams::init(int n, int num_types) {
    N = n;
    cx.resize(n); cy.resize(n); rx.resize(n); ry.resize(n); angle.resize(n);
    color_r.resize(n); color_g.resize(n); color_b.resize(n); color_a.resize(n);
    type_logits.resize(n * num_types, 0.0f);
    z.resize(n, 0.0f);  // soft depth: 初始全零，由 gradient_optimize 设置具体值
}

ShapeParams OptParams::get_shape(int i) const {
    ShapeParams sp;
    sp.cx = cx[i]; sp.cy = cy[i]; sp.rx = rx[i]; sp.ry = ry[i];
    sp.angle = angle[i];
    // Logit -> linear (host-facing ShapeParams keeps [0,255] / [0,1] ranges)
    sp.color_r = 255.0f * vin_sigmoid(color_r[i]);
    sp.color_g = 255.0f * vin_sigmoid(color_g[i]);
    sp.color_b = 255.0f * vin_sigmoid(color_b[i]);
    sp.opacity = vin_sigmoid(color_a[i]);
    return sp;
}

void OptParams::set_shape(int i, const ShapeParams& sp) {
    cx[i] = sp.cx; cy[i] = sp.cy; rx[i] = sp.rx; ry[i] = sp.ry;
    angle[i] = sp.angle;
    // Linear -> logit
    color_r[i] = vin_logit(sp.color_r / 255.0f);
    color_g[i] = vin_logit(sp.color_g / 255.0f);
    color_b[i] = vin_logit(sp.color_b / 255.0f);
    color_a[i] = vin_logit(sp.opacity);
}

// ================================================================
// GradientOptimizer
// ================================================================

// 将 FH6 shape_type 值映射到模板索引
// num_builtins: 内置图元数量（4 或 0）
static int shape_type_to_template_idx(int shape_type, int num_builtins) {
    // 纯外部 stamps：整数 type 即 stamp 索引，直接透传。
    // 否则 FH6 编码 switch 会抢匹配（如 case 1 把 fingerprint 误映射到模板 2=square）。
    if (num_builtins == 0) return shape_type;
    switch (shape_type) {
        case 228: return 0;
        case 16:  return 1;
        case 1:   return 2;
        case 103: return 3;
        default:
            // shape_type >= num_builtins → external stamp, already in template index space
            if (shape_type >= num_builtins)
                return shape_type;
            // shape_type < num_builtins but not a builtin → fallback to 0
            return 0;
    }
}

GradientOptimizer::GradientOptimizer(const OptimizerConfig& config, int shape_type)
    : config_(config), shape_type_(shape_type), rng_(config.random_seed) {}

GradientOptimizer::~GradientOptimizer() = default;

std::vector<Shape> GradientOptimizer::optimize(
    const cv::Mat& target_rgb, const cv::Mat& mask,
    const cv::Mat& edge_map, const cv::Mat& saliency_map,
    const cv::Mat& dark_mask, const cv::Mat& posterized,
    const std::vector<std::tuple<int,int,int>>& palette,
    const cv::Mat& grad_dir, const Color& bg_color)
{
    target_rgb_ = target_rgb.clone();
    mask_ = mask.clone();
    edge_map_ = edge_map.clone();
    saliency_map_ = saliency_map.clone();
    dark_mask_ = dark_mask.clone();
    palette_ = palette;
    stop_requested_ = false;

    // Compute optimization resolution
    int H_orig = target_rgb.rows, W_orig = target_rgb.cols;
    int max_dim = std::max(H_orig, W_orig);
    scale_ = std::min(1.0f, static_cast<float>(config_.opt_resolution) / max_dim);
    H_opt_ = std::max(4, static_cast<int>(H_orig * scale_));
    W_opt_ = std::max(4, static_cast<int>(W_orig * scale_));

    VIN_INFO("Optimizer: %dx%d -> %dx%d (scale=%.3f)",
             W_orig, H_orig, W_opt_, H_opt_, scale_);

    // Resize target and mask
    cv::resize(target_rgb, target_small_, {W_opt_, H_opt_}, 0, 0, cv::INTER_AREA);
    cv::Mat mask_u8;
    mask.convertTo(mask_u8, CV_8U, 255);
    cv::resize(mask_u8, mask_small_, {W_opt_, H_opt_}, 0, 0, cv::INTER_NEAREST);
    mask_small_ = mask_small_ > 127;

    // Resize posterized
    if (!posterized.empty()) {
        cv::resize(posterized, posterized_small_, {W_opt_, H_opt_}, 0, 0, cv::INTER_NEAREST);
    }

    // Initialize renderer
    // R23: 确定性归约开关（必须在 init() 之前，影响 partials 缓冲分配）
    renderer_.set_deterministic_reduce(config_.deterministic_reduce != 0);
    // alpha 损失：RGBA MSE（channels=4），透明区 rendered_alpha→0 → FH6 真透明
    renderer_.set_alpha_loss_enabled(config_.alpha_loss);
    renderer_.init(W_opt_, H_opt_, config_.max_shapes, num_template_types_, num_builtins_);
    // 加载外部图章（必须在 init 之后、梯度优化之前调用）
    if (!external_stamp_names_.empty()) {
        VIN_INFO("Loading %d external stamps...", (int)external_stamp_names_.size());
        renderer_.load_external_stamps(external_stamp_names_);
    }
    bg_color_ = bg_color;
    renderer_.set_bg_color(static_cast<float>(bg_color.r), static_cast<float>(bg_color.g), static_cast<float>(bg_color.b));

    // Set target on GPU (RGB + mask + target_alpha)
    std::vector<float> h_target(H_opt_ * W_opt_ * 3);
    std::vector<float> h_mask_gpu(H_opt_ * W_opt_);
    std::vector<float> h_target_alpha(H_opt_ * W_opt_);
    for (int y = 0; y < H_opt_; ++y) {
        for (int x = 0; x < W_opt_; ++x) {
            int idx = y * W_opt_ + x;
            bool opaque = mask_small_.at<uint8_t>(y, x) > 0;
            if (opaque) {
                auto& c = target_small_.at<cv::Vec3b>(y, x);
                h_target[idx * 3 + 0] = c[0];
                h_target[idx * 3 + 1] = c[1];
                h_target[idx * 3 + 2] = c[2];
            } else {
                h_target[idx * 3 + 0] = 0.0f;
                h_target[idx * 3 + 1] = 0.0f;
                h_target[idx * 3 + 2] = 0.0f;
            }
            h_mask_gpu[idx] = opaque ? 1.0f : 0.0f;
            h_target_alpha[idx] = opaque ? 255.0f : 0.0f;
        }
    }
    renderer_.set_target(h_target.data(), h_mask_gpu.data(), h_target_alpha.data());


    // 初始化形状
    int N = config_.max_shapes;
    params_.init(N, num_template_types_);

    if (!config_.init_shapes_path.empty()) {
        // --init-shapes: 从 optimize 输出 json 加载初始形状，跳过 initializer。
        // 复用 render 命令路径的 json 读取逻辑（linear→logit 由 set_shape 完成）。
        std::ifstream sf = open_ifstream(config_.init_shapes_path);
        if (!sf.is_open()) {
            VIN_ERROR("--init-shapes: cannot open %s", config_.init_shapes_path.c_str());
            return {};
        }
        nlohmann::json data = nlohmann::json::parse(sf);
        nlohmann::json shapes_arr;
        if (data.is_array()) {
            shapes_arr = data;
        } else if (data.contains("shapes") && data["shapes"].is_array()) {
            shapes_arr = data["shapes"];
        } else {
            VIN_ERROR("--init-shapes: invalid JSON format (expected array or {shapes:[]})");
            return {};
        }
        int N_json = (int)shapes_arr.size();
        if (N_json == 0) {
            VIN_ERROR("--init-shapes: no shapes in JSON");
            return {};
        }

        // stamps 数组（用于 type 字符串→模板索引映射，与 _eval_r12.py convert_shapes 一致）
        std::vector<std::string> stamps;
        if (data.contains("stamps") && data["stamps"].is_array()) {
            for (auto& st : data["stamps"]) stamps.push_back(st.get<std::string>());
        }

        // N 由 json 形状数决定（-n 如提供则忽略，以 json 为准）
        if (N_json > config_.max_shapes) {
            // 容量不足时重新分配
            params_.init(N_json, num_template_types_);
        }
        params_.N = N_json;
        N = N_json;

        for (int i = 0; i < N; ++i) {
            auto& s = shapes_arr[i];
            ShapeParams sp;
            if (s.contains("data") && s["data"].is_array()) {
                // optimize 输出格式: data[5]={cx,cy,rx,ry,angle}, color[4]={r,g,b,opacity*255}
                sp.cx = s["data"][0].get<float>();
                sp.cy = s["data"][1].get<float>();
                sp.rx = s["data"][2].get<float>();
                sp.ry = s["data"][3].get<float>();
                sp.angle = s["data"][4].get<float>();
                sp.color_r = s["color"][0].get<float>();
                sp.color_g = s["color"][1].get<float>();
                sp.color_b = s["color"][2].get<float>();
                sp.opacity = s["color"][3].get<float>() / 255.0f;
            } else {
                // diagnosis 格式: cx/cy/rx/ry/angle/r/g/b/opacity 直接字段
                sp.cx = s["cx"].get<float>();
                sp.cy = s["cy"].get<float>();
                sp.rx = s["rx"].get<float>();
                sp.ry = s["ry"].get<float>();
                sp.angle = s["angle"].get<float>();
                sp.color_r = s["r"].get<float>();
                sp.color_g = s["g"].get<float>();
                sp.color_b = s["b"].get<float>();
                sp.opacity = s["opacity"].get<float>();
            }
            params_.set_shape(i, sp);  // linear→logit

            // type 索引：字符串→stamps.index；整数→shape_type_to_template_idx
            int tidx = -1;
            if (s.contains("type")) {
                if (s["type"].is_string()) {
                    const std::string& tn = s["type"].get<std::string>();
                    for (int k = 0; k < (int)stamps.size(); ++k) {
                        if (stamps[k] == tn) { tidx = k; break; }
                    }
                    // 退化：按 basename 匹配（兼容相对/绝对路径差异）
                    if (tidx < 0) {
                        auto basename = [](const std::string& p) {
                            size_t pos = p.find_last_of("/\\");
                            return (pos == std::string::npos) ? p : p.substr(pos + 1);
                        };
                        std::string tn_base = basename(tn);
                        for (int k = 0; k < (int)stamps.size(); ++k) {
                            if (basename(stamps[k]) == tn_base) { tidx = k; break; }
                        }
                    }
                } else if (s["type"].is_number()) {
                    tidx = shape_type_to_template_idx(s["type"].get<int>(), num_builtins_);
                }
            }
            if (tidx < 0 || tidx >= num_template_types_) tidx = 0;

            // type_logits: continue=one-hot 锁定 json type（断点续训）；reset=全 0 均匀（真 keep-init，DARTS 重搜）
            if (config_.init_shapes_mode == "reset") {
                for (int k = 0; k < num_template_types_; ++k) {
                    params_.type_logits[i * num_template_types_ + k] = 0.0f;
                }
            } else {
                // continue（默认，现状不变）：one-hot（正确类型=10，其他=-10）
                for (int k = 0; k < num_template_types_; ++k) {
                    params_.type_logits[i * num_template_types_ + k] = (k == tidx) ? 10.0f : -10.0f;
                }
            }
        }
        VIN_INFO("[init-shapes] loaded %d shapes from %s", N, config_.init_shapes_path.c_str());
    } else if (config_.init_mode == "kmedoids") {
        // 构造式几何：误差加权 k-medoids（对齐 Python --init-shapes --init-shapes-mode reset）
        // type 不在本策略内：type_logits 保持 params_.init() 的全 0，由 DARTS 重搜
        // 内容区 ROI（优化分辨率坐标）：mask_small_ 由"padding 后的 alpha"阈值化而来
        // （padding 区 alpha=0 → mask=0），其包围盒即原图内容区（= [pad, pad+W) 等）。
        // 网格只铺在内容区内 → 形状中心不会落进透明 padding 区。
        // 再把 mask_small_ 本身交给 initializer：候选点按【逐像素 mask】排除内容区
        // 内的透明像素（bbox 内的洞/外部透明），权重场对透明像素置 0，
        // 且 KM 读到的 RGB 与下方 GPU target 口径一致（透明像素 RGB=0）。
        cv::Rect content = cv::boundingRect(mask_small_);
        if (content.width <= 0 || content.height <= 0)
            content = cv::Rect(0, 0, W_opt_, H_opt_);
        initializer_ = std::make_unique<KMedoidsInitializer>(
            target_rgb_, mask_small_, W_opt_, H_opt_,
            config_.km_grid, config_.km_alpha, config_.km_radius_scale,
            config_.km_iters, config_.km_opacity, content);
        initializer_->initialize(params_, num_template_types_, config_, rng_);
    } else {
        // 默认 "variance"
        auto var_init = std::make_unique<VarianceInitializer>(
            target_rgb_, mask_small_, W_opt_, H_opt_, config_.detail_first);
        // 传递梯度方向图（resize 到 opt 分辨率），用于边缘对齐的角度初始化
        if (!grad_dir.empty()) {
            cv::Mat grad_dir_small;
            cv::resize(grad_dir, grad_dir_small, {W_opt_, H_opt_}, 0, 0, cv::INTER_LINEAR);
            var_init->set_grad_dir(grad_dir_small);
        }
        initializer_ = std::move(var_init);
        initializer_->initialize(params_, num_template_types_, config_, rng_);
    }

    // [DEBUG] 打印前 5 个 shape 的初始化参数
    VIN_INFO("[init-dbg] N=%d nt=%d nb=%d eb=%d", N, num_template_types_, num_builtins_, enable_builtins_);
    for (int i = 0; i < 5 && i < N; ++i) {
        VIN_INFO("[init-dbg] shape[%d] cx=%.2f cy=%.2f rx=%.2f ry=%.2f angle=%.2f cr=%.4f cg=%.4f cb=%.4f ca=%.4f",
                 i, params_.cx[i], params_.cy[i], params_.rx[i], params_.ry[i], params_.angle[i],
                 params_.color_r[i], params_.color_g[i], params_.color_b[i], params_.color_a[i]);
    }

    // 类型分配（与策略正交，保持不变）
    {
        if (config_.round_robin) {
            std::vector<int> types(N);
            for (int i = 0; i < N; ++i) {
                // H2 修复：diffbmp 是 top-first（shape 0=顶层），其 round-robin 为
                // types[i] = flip(arange(N)%p)[i] = (N-1-i)%p。vinylizer 是 bottom-first
                // （shape 0=底层），索引 j 对应 diffbmp 索引 i=N-1-j，代入得
                // types[j] = (N-1-(N-1-j))%p = j%p。旧公式 (N-1-i)%p 是把 top-first
                // 公式直接套在 bottom-first 上的镜像 bug（每层 stamp 配错，多重集不变）。
                // 修复后与 R21 init json（z 序反转后恰为 j%p 约定）对齐。
                types[i] = i % num_template_types_;
            }
            renderer_.upload_types(types.data(), N);
        } else if (!config_.enable_darts) {
            // shape_type_ 可能是 FH6 值（16/1/103/228）或外部图章索引，
            // 统一转为模板索引后再上传，避免 GPU to_template_idx 歧义。
            int tidx = shape_type_to_template_idx(shape_type_, num_builtins_);
            if (tidx >= num_template_types_) tidx = 0;
            std::vector<int> types(N, tidx);
            renderer_.upload_types(types.data(), N);
        }
    }
    renderer_.upload_type_logits(params_.type_logits.data(), N);
    renderer_.set_render_mode(config_.enable_darts ? 1 : 0);

    // Run optimization
    gradient_optimize();

    // Convert back to shapes (best params already restored inside gradient_optimize)
    return params_to_shapes();
}

// ================================================================
// Initialization
// ================================================================



std::tuple<int,int,int> GradientOptimizer::get_posterized_color(float x, float y) {
    if (!posterized_small_.empty() && !palette_.empty()) {
        int yi = std::clamp(static_cast<int>(y), 0, H_opt_ - 1);
        int xi = std::clamp(static_cast<int>(x), 0, W_opt_ - 1);
        int label = posterized_small_.at<uint8_t>(yi, xi);
        if (label < static_cast<int>(palette_.size())) {
            return palette_[label];
        }
    }
    // 回退：采样目标像素邻域均值（原 get_local_color 逻辑内联）
    int r = 5;
    int x0 = std::max(0, static_cast<int>(x) - r);
    int x1 = std::min(W_opt_, static_cast<int>(x) + r + 1);
    int y0 = std::max(0, static_cast<int>(y) - r);
    int y1 = std::min(H_opt_, static_cast<int>(y) + r + 1);

    cv::Mat local = target_small_(cv::Range(y0, y1), cv::Range(x0, x1));
    cv::Mat local_mask = mask_small_(cv::Range(y0, y1), cv::Range(x0, x1));

    int sum_r = 0, sum_g = 0, sum_b = 0, count = 0;
    for (int py = 0; py < local.rows; ++py) {
        for (int px = 0; px < local.cols; ++px) {
            if (local_mask.at<uint8_t>(py, px) > 0) {
                auto& c = local.at<cv::Vec3b>(py, px);
                sum_r += c[0]; sum_g += c[1]; sum_b += c[2];
                ++count;
            }
        }
    }
    if (count > 0)
        return {sum_r / count, sum_g / count, sum_b / count};
    else
        return {0, 0, 0};
}


float GradientOptimizer::run_single_iter(
    OptParams& params,
    int iter, int total_iters,
    float* d_grad_history, int write_idx,
    bool local_phase, const std::vector<int>& frozen_indices,
    float temperature)
{
    int N = params.N;

    float progress = static_cast<float>(iter) / std::max(total_iters - 1, 1);

    float mse;
    int update_mask = 15;  // 默认全量（bit0 位置 + bit1 RGBA + bit2 logits + bit3 尺度，= baseline）
    mse = renderer_.render_forward_backward(N, d_grad_history, write_idx, temperature);
    if (std::isnan(mse)) {
        VIN_WARN("NaN MSE at iter %d, skipping Adam update", iter);
        return mse;
    }

    // Update frozen mask on GPU
    if (!frozen_indices.empty()) {
        std::vector<int> frozen_mask(N, 0);
        for (int i : frozen_indices) frozen_mask[i] = 1;
        renderer_.set_frozen_mask(frozen_mask.data(), N);
    }

    // Learning rate schedule
    float lr;
    if (config_.lr_schedule == "linear") {
        lr = config_.lr_start + (config_.lr_end - config_.lr_start) * progress;
    } else if (config_.lr_schedule == "piecewise") {
        float lr_mid = (config_.lr_start + config_.lr_end) * 0.5f;
        if (progress < 0.5f) {
            lr = config_.lr_start;
        } else if (progress < 0.8f) {
            float t2 = (progress - 0.5f) / 0.3f;
            lr = config_.lr_start + (lr_mid - config_.lr_start) * t2;
        } else {
            float t3 = (progress - 0.8f) / 0.2f;
            lr = lr_mid + (config_.lr_end - lr_mid) * t3;
        }
    } else {
        // cosine（默认）
        lr = config_.lr_end + (config_.lr_start - config_.lr_end) *
             0.5f * (1.0f + std::cos(static_cast<float>(M_PI) * progress));
    }

    // LR warmup（--lr-warmup-iters）：前 W 步 lr 从 0 线性爬到 lr_start，
    // 越过 Adam bias-correction 满步长期（前几步实际步长≈lr·sign(g)，与梯度幅值无关）。
    // from-scratch 无必要（起点在高斜坡，满步长=大步下坡）；continue/fine-tune 关键
    // （起点在最优盆地，满步长踢出低谷，实测 lr=1.0 continue 初期 MSE 155→2029）。
    // 默认 0=关 → 严格 baseline 等价；det1 确定性不受影响（纯数值乘法）。
    if (config_.lr_warmup_iters > 0 && iter < config_.lr_warmup_iters) {
        lr = config_.lr_start * (float)(iter + 1) / (float)config_.lr_warmup_iters;
    }


    float angle_factor = config_.lr_factor_angle;
    float logits_factor = config_.enable_darts ? 1.0f : 0.0f;
    float r_factor = config_.lr_factor_r;
    float xy_factor = config_.lr_factor_xy;
    float a_factor = config_.lr_factor_a;


    // Fused GPU Adam: gradient clip + Adam update + param clamp + STE quantize
    // 解耦模式：位置相位 update_mask=5（几何+logits），颜色相位 =2（RGBA），baseline=7。
    // angle_factor/logits_factor 在位置相位已被量纲缩放（见上 v4 修正）。
    adam_step_++;
    renderer_.fused_adam_update(
        N, adam_step_, lr, 0.9f, config_.adam_beta2, 1e-8f,
        config_.max_grad_norm,
        W_opt_, H_opt_,
        xy_factor, r_factor, angle_factor,
        config_.lr_factor_rgb, a_factor,
        logits_factor, update_mask,
        0.0f,
        config_.uniform_scale);






    // Uniform scale enforcement: rx -> ry (applied after every Adam step)
    if (config_.uniform_scale) {
        renderer_.apply_uniform_scale(N);
    }

#ifndef __NODEBUG__
    // v7 debug: type distribution and logits statistics every 10 iters (async DMA)
    static int debug_interval = 0;
    static std::vector<float> debug_logits_buf;
    debug_interval++;
    if (debug_interval % 10 == 0) {
        int N = params.N;
        int nt = num_template_types_;

        // Try to process previous async download result
        debug_logits_buf.resize(N * nt);
        if (renderer_.try_process_type_logits(N, debug_logits_buf.data())) {
            std::vector<int> count(nt, 0);
            std::vector<float> sum_w(nt, 0.0f);
            float max_logit = -1e10f, min_logit = 1e10f;
            for (int i = 0; i < N; ++i) {
                // argmax
                int best_t = 0;
                float best_l = debug_logits_buf[i * nt + 0];
                for (int t = 1; t < nt; ++t) {
                    float l = debug_logits_buf[i * nt + t];
                    if (l > best_l) { best_l = l; best_t = t; }
                }
                count[best_t]++;

                // softmax
                float max_l = debug_logits_buf[i * nt + 0];
                for (int t = 1; t < nt; ++t)
                    max_l = fmaxf(max_l, debug_logits_buf[i * nt + t]);
                float sum_exp = 0.0f;
                for (int t = 0; t < nt; ++t)
                    sum_exp += expf(debug_logits_buf[i * nt + t] - max_l);
                for (int t = 0; t < nt; ++t)
                    sum_w[t] += expf(debug_logits_buf[i * nt + t] - max_l) / sum_exp;

                for (int t = 0; t < nt; ++t) {
                    float l = debug_logits_buf[i * nt + t];
                    if (l > max_logit) max_logit = l;
                    if (l < min_logit) min_logit = l;
                }
            }
            VIN_INFO("  [v7] iter %d, mse=%.1f, type_counts:", iter, mse);
            for (int t = 0; t < nt; ++t) {
                if (count[t] > 0)
                    VIN_INFO("    t%d: %d", t, count[t]);
            }
            VIN_INFO("  [v7] logits: min=%.2f max=%.2f", min_logit, max_logit);
        }

        // Kick off async download for next time
        renderer_.download_type_logits_async(N);
    }
#endif

    return mse;
}

void GradientOptimizer::gradient_optimize() {
    int N = params_.N;
    int total_iters = config_.total_iters;

    // (grad history ring buffer removed with relocation)

    // Upload initial parameters to GPU-resident buffers
    {
        params_.to_shape_params(cached_shape_params_);
        renderer_.upload_params_gpu(cached_shape_params_.data(), N);
        std::vector<float> logits(N * num_template_types_, 0.0f);
        if (config_.round_robin) {
            // 为每个 shape 设置 logits 偏置，确保 force_lock argmax 匹配 round-robin 分配
            // H2 修复：与上方 upload_types 同步改为 i%p（bottom-first 正确映射，推导见上）
            for (int i = 0; i < N; ++i) {
                int tidx = i % num_template_types_;
                for (int t = 0; t < num_template_types_; t++) {
                    logits[i * num_template_types_ + t] = (t == tidx) ? 1.0f : 0.0f;
                }
            }
        } else if (config_.enable_darts) {
            // DARTS: uniform initialization (all logits = 0, equal softmax weights)
            // --init-shapes: 复用已加载的 type_logits（one-hot），不从零开始
            if (!config_.init_shapes_path.empty()) {
                for (int i = 0; i < N * num_template_types_; ++i)
                    logits[i] = params_.type_logits[i];
            }
        } else {
            // Fixed type: lock to single type via large logit bias
            int tidx = shape_type_to_template_idx(shape_type_, num_builtins_);
            if (tidx >= num_template_types_) tidx = 0;
            for (int i = 0; i < N; ++i) {
                for (int t = 0; t < num_template_types_; t++) {
                    logits[i * num_template_types_ + t] = (t == tidx) ? 1.0f : 0.0f;
                }
            }
        }
        renderer_.upload_type_logits(logits.data(), N);
    }

#if 0  // best_snapshot 机制暂时禁用
    // Allocate pinned memory and event for async param snapshots
    {
        cudaError_t e = cudaMallocHost(&snap_pinned_, N * (9 + num_template_types_) * sizeof(float));
        if (e != cudaSuccess) {
            VIN_WARN("cudaMallocHost failed for snapshots: %s", cudaGetErrorString(e));
            snap_pinned_ = nullptr;
        }
    }
    cudaEventCreate(&snap_event_);
    best_mse_ = 1e20f;
    has_best_ = false;
    snap_pending_ = false;
#endif

    VIN_INFO("Starting optimization: %d total iters, lr=%.3f->%.3f schedule=%s warmup=%d",
             total_iters, config_.lr_start, config_.lr_end,
             config_.lr_schedule.c_str(), config_.lr_warmup_iters);

    renderer_.set_render_mode(config_.enable_darts ? 1 : 0);  // DARTS or PURE

    // Set initial blur sigma
    // 双模糊退火：target_blur=true 时调度起点改用 blur_sigma_max（模板与目标共用）
    const float sigma_start_eff = config_.blur_sigma_start;
    current_blur_sigma_ = sigma_start_eff;
    renderer_.set_blur_sigma(current_blur_sigma_);
    int sigma_anneal_iters = total_iters - config_.blur_sigma_warmup;
    VIN_INFO("Sigma annealing: %.2f -> %.2f, warmup=%d, schedule=%s",
             sigma_start_eff, config_.blur_sigma_end,
             config_.blur_sigma_warmup, config_.blur_sigma_schedule.c_str());


    for (int iter = 0; iter < total_iters; ++iter) {
        if (stop_requested_) break;

        N = params_.N;  // 增量出生会增长 params_.N，每轮同步局部 N

#if 0  // best_snapshot 机制暂时禁用
        // Consume previous async snapshot (best-MSE tracking)
        if (snap_pending_ && snap_pinned_ && N > 0) {
            cudaEventSynchronize(snap_event_);
            if (!std::isnan(cycle_mse_) && cycle_mse_ < best_mse_) {
                best_mse_ = cycle_mse_;
                best_params_flat_.resize(N * (9 + num_template_types_));
                // Unified layout: snap_pinned_ is [cx,cy,rx,ry,angle,colors_interleaved,opacity,logits]
                // best_params_flat_ now uses the same interleaved colors layout (P1-5)
                memcpy(best_params_flat_.data(), snap_pinned_, N * (9 + num_template_types_) * sizeof(float));
                has_best_ = true;
            }
            snap_pending_ = false;
        }
#endif

        // DARTS temperature annealing (cosine schedule)
        // 注：E102 实验证实"温度在锁定点前提前退火"对性能中性（5 图 -0.05），
        // 锁定切换时的 MSE 跳升并非主要损失来源，故保持原始全程余弦退火。
        float temperature = 1.0f;
        if (config_.enable_darts) {
            float temp_progress = static_cast<float>(iter) / std::max(total_iters - 1, 1);
            float cos_t = 0.5f * (1.0f + std::cos(static_cast<float>(M_PI) * temp_progress));
            temperature = config_.darts_temp_end
                        + (config_.darts_temp_start - config_.darts_temp_end) * cos_t;

            // DARTS 锁定切换：进度超过 darts_lock_ratio 后切换到 PURE 模式，
            // 使形状参数在 argmax 渲染下继续优化，消除优化-渲染不一致。
            // 切换时打印一次日志。
            bool should_lock = (temp_progress >= config_.darts_lock_ratio);
            int desired_render_mode = should_lock ? 0 : 1;
            if (desired_render_mode != renderer_.get_render_mode()) {
                VIN_INFO("[DARTS-lock] iter %d/%d (%.0f%%) switching to PURE mode (force_lock)",
                         iter, total_iters, temp_progress * 100.0f);
                renderer_.set_render_mode(desired_render_mode);
            }
        }




        float mse = run_single_iter(params_,
                                    iter, total_iters,
                                    nullptr, 0,
                                    false, {}, temperature);





            // Async D2H copy to pinned memory (double-buffered, rate-limited ~30fps)
            // 优化器线程在排队新帧前同步上一帧的 cudaEvent（~1ms 阻塞），
            // 完成后通过共享原子通知 GUI，GUI 无需调用 CUDA API
            if (viz_initialized_) {
                auto now = std::chrono::steady_clock::now();
                bool first_iter = (iter == 0);
                bool interval_ok = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - viz_last_time_).count() >= 30;
                if (first_iter || interval_ok) {
                    // 发布上一帧：只有真的排过队（viz_synced_idx_ >= 0）才同步它的 event。
                    // 修复：旧代码首帧分支不写 viz_synced_idx_，第二帧会读 viz_event_[-1]
                    // （越界），并可能把 -1 写进 read_idx（GUI 丢帧）。
                    if (!viz_first_frame_ && viz_synced_idx_ >= 0) {
                        cudaEventSynchronize(viz_event_[viz_synced_idx_]);
                        viz_read_idx_ptr_->store(viz_synced_idx_);
                    }

                    int w = viz_write_idx_;
                    renderer_.async_copy_canvas(viz_pinned_[w], viz_event_[w]);

                    if (viz_first_frame_) {
                        cudaEventSynchronize(viz_event_[w]);
                        viz_read_idx_ptr_->store(w);
                        viz_first_frame_ = false;
                    }
                    viz_synced_idx_ = w;

                    viz_write_idx_ = 1 - w;
                    viz_last_time_ = now;
                    static int viz_queue_count = 0;
                    if (++viz_queue_count <= 5) {
                        VIN_INFO("viz: queued frame %d to buffer %d, iter %d",
                             viz_queue_count, w, iter);
                    }
                }
            }


            if (iter % 50 == 0) {
            VIN_INFO("  iter %d/%d, mse=%.1f, sigma=%.2f", iter, total_iters,
                     mse, current_blur_sigma_);
            }

            if (progress_cb_) {
                progress_cb_(static_cast<float>(iter) / total_iters,
                            mse, iter);
            }



            // Sigma annealing: update blur_sigma based on schedule
            {
                float sigma;
                if (iter < config_.blur_sigma_warmup) {
                    sigma = sigma_start_eff;
                } else {
                    float t = static_cast<float>(iter - config_.blur_sigma_warmup) /
                              std::max(sigma_anneal_iters, 1);
                    t = std::min(t, 1.0f);

                    if (config_.blur_sigma_schedule == "linear") {
                        sigma = sigma_start_eff + (config_.blur_sigma_end - sigma_start_eff) * t;
                    } else if (config_.blur_sigma_schedule == "cosine") {
                        float cos_t = 0.5f * (1.0f + std::cos(static_cast<float>(M_PI) * t));
                        sigma = config_.blur_sigma_end + (sigma_start_eff - config_.blur_sigma_end) * cos_t;
                    } else if (config_.blur_sigma_schedule == "piecewise") {
                        // Phase 1: 0-50% hold at start, Phase 2: 50-80% linear to mid, Phase 3: 80-100% linear to end
                        float sigma_mid = (sigma_start_eff + config_.blur_sigma_end) * 0.5f;
                        if (t < 0.5f) {
                            sigma = sigma_start_eff;
                        } else if (t < 0.8f) {
                            float t2 = (t - 0.5f) / 0.3f;
                            sigma = sigma_start_eff + (sigma_mid - sigma_start_eff) * t2;
                        } else {
                            float t3 = (t - 0.8f) / 0.2f;
                            sigma = sigma_mid + (config_.blur_sigma_end - sigma_mid) * t3;
                        }
                    } else {
                        // Default to cosine
                        float cos_t = 0.5f * (1.0f + std::cos(static_cast<float>(M_PI) * t));
                        sigma = config_.blur_sigma_end + (sigma_start_eff - config_.blur_sigma_end) * cos_t;
                    }
                }
                renderer_.set_blur_sigma(sigma);
                current_blur_sigma_ = sigma;
    }

#if 0  // best_snapshot 机制暂时禁用
    // Consume last async snapshot
    if (snap_pending_ && snap_pinned_ && N > 0) {
        cudaEventSynchronize(snap_event_);
        if (!std::isnan(cycle_mse_) && cycle_mse_ < best_mse_) {
            best_mse_ = cycle_mse_;
            best_params_flat_.resize(N * (9 + num_template_types_));
            // Unified layout: direct block copy (interleaved colors, same as snap_pinned_)
            memcpy(best_params_flat_.data(), snap_pinned_, N * (9 + num_template_types_) * sizeof(float));
            has_best_ = true;
        }
        snap_pending_ = false;
    }
#endif

#if 0  // best_snapshot 机制暂时禁用
    // Restore best params to CPU + GPU if a better snapshot was found
    if (has_best_ && N > 0) {
        for (int i = 0; i < N; ++i) {
            params_.cx[i] = best_params_flat_[0 * N + i];
            params_.cy[i] = best_params_flat_[1 * N + i];
            params_.rx[i] = best_params_flat_[2 * N + i];
            params_.ry[i] = best_params_flat_[3 * N + i];
            params_.angle[i] = best_params_flat_[4 * N + i];
            // Colors now interleaved in best_params_flat_ (unified with snap_pinned_ layout, P1-5)
            // Logit parameterization: best_params_flat_ stores logits (from GPU snapshot),
            // copied directly to OptParams (also logit space), no conversion needed.
            params_.color_r[i] = best_params_flat_[5 * N + i * 3 + 0];
            params_.color_g[i] = best_params_flat_[5 * N + i * 3 + 1];
            params_.color_b[i] = best_params_flat_[5 * N + i * 3 + 2];
            params_.color_a[i] = best_params_flat_[8 * N + i];
        }
        int nt = num_template_types_;
        memcpy(params_.type_logits.data(), &best_params_flat_[9 * N], N * nt * sizeof(float));
        params_.to_shape_params(cached_shape_params_);
        renderer_.upload_params_gpu(cached_shape_params_.data(), N);
        renderer_.upload_type_logits(params_.type_logits.data(), N);
        {
            std::vector<int> types(N);
            int num_builtins = num_builtins_;
            for (int i = 0; i < N; ++i) {
                // argmax over logits
                int best_t = 0;
                float best_l = params_.type_logits[i * nt + 0];
                for (int t = 1; t < nt; ++t) {
                    if (params_.type_logits[i * nt + t] > best_l) {
                        best_l = params_.type_logits[i * nt + t];
                        best_t = t;
                    }
                }
                // Map to FH6 shape type (builtins) or keep as index (external stamps)
                if (best_t < num_builtins) {
                    switch (best_t) {
                        case 0: types[i] = SHAPE_SOFT_ELLIPSE_228; break;
                        case 1: types[i] = SHAPE_ROTATED_ELLIPSE; break;
                        case 2: types[i] = SHAPE_RECTANGLE; break;
                        case 3: types[i] = SHAPE_TRIANGLE; break;
                        default: types[i] = SHAPE_ROTATED_ELLIPSE; break;
                    }
                } else {
                    types[i] = best_t;
                }
            }
            renderer_.lock_types(types.data(), N);
        }
        VIN_INFO("Restored best snapshot params: MSE=%.1f", best_mse_);
    }
#endif

#if 0  // best_snapshot 机制暂时禁用
    // Free snapshot resources
    if (snap_pinned_) {
        cudaFreeHost(snap_pinned_);
        snap_pinned_ = nullptr;
    }
    if (snap_event_) {
        cudaEventDestroy(snap_event_);
        snap_event_ = nullptr;
    }
#endif


    // === 死形状统计：utilization = vis / vis_full < 0.05 视为死形状（与 reloc 判定一致）===
    {
        renderer_.compute_visibility(N);
        std::vector<float> h_vis(N), h_vis_full(N);
        renderer_.download_visibility(h_vis.data(), N);
        renderer_.download_visibility_full(h_vis_full.data(), N);
        int n_dead = 0;
        for (int i = 0; i < N; ++i) {
            float util = (h_vis_full[i] > 1e-3f) ? (h_vis[i] / h_vis_full[i]) : 0.0f;
            if (util < 0.05f) n_dead++;
        }
        VIN_INFO("[dead-shapes] utilization<0.05: %d/%d (%.1f%%)",
                 n_dead, N, 100.0f * n_dead / N);
    }




    // Cleanup — sync render stream only (avoid OpenGL-blocking device sync)
    renderer_.sync_stream();

    // Drain last pending viz frame so GUI shows final result
    if (!viz_first_frame_ && viz_initialized_ && viz_synced_idx_ >= 0) {
        cudaEventSynchronize(viz_event_[viz_synced_idx_]);
        viz_read_idx_ptr_->store(viz_synced_idx_);
    }

    }  // for (iter)
}

std::vector<Shape> GradientOptimizer::params_to_shapes() {
    // Download params from GPU before converting
    params_.to_shape_params(cached_shape_params_);
    renderer_.download_params_gpu(cached_shape_params_.data(), params_.N);
    for (int i = 0; i < params_.N; ++i) params_.set_shape(i, cached_shape_params_[i]);

    int nt = num_template_types_;
    std::vector<float> logits(params_.N * nt);
    renderer_.download_type_logits(logits.data(), params_.N);

    // [DARTS-debug] 统计 type_logits 的 softmax 分布（temperature=0.5，对齐优化末期）
    {
        const float temp = config_.darts_temp_end;
        std::vector<float> max_probs(params_.N, 0.0f);
        std::vector<float> entropy(params_.N, 0.0f);
        for (int i = 0; i < params_.N; ++i) {
            float max_l = logits[i * nt];
            for (int t = 1; t < nt; ++t) max_l = std::max(max_l, logits[i * nt + t]);
            float sum = 0.0f;
            std::vector<float> probs(nt);
            for (int t = 0; t < nt; ++t) {
                float e = std::exp((logits[i * nt + t] - max_l) / temp);
                probs[t] = e; sum += e;
            }
            float max_p = 0.0f, ent = 0.0f;
            for (int t = 0; t < nt; ++t) {
                probs[t] /= sum;
                max_p = std::max(max_p, probs[t]);
                if (probs[t] > 1e-6f) ent -= probs[t] * std::log(probs[t]);
            }
            max_probs[i] = max_p;
            entropy[i] = ent;
        }
        float mean_maxp = 0.0f, mean_ent = 0.0f;
        int n_locked = 0;  // max_p > 0.9 视为已锁定
        std::vector<int> type_hist(nt, 0);  // argmax 类型分布直方图
        for (int i = 0; i < params_.N; ++i) {
            mean_maxp += max_probs[i];
            mean_ent += entropy[i];
            if (max_probs[i] > 0.9f) n_locked++;
            int argmax_t = 0;
            float best = logits[i * nt];
            for (int t = 1; t < nt; ++t) {
                if (logits[i * nt + t] > best) { best = logits[i * nt + t]; argmax_t = t; }
            }
            type_hist[argmax_t]++;
        }
        mean_maxp /= params_.N;
        mean_ent /= params_.N;
        std::string hist_str;
        for (int t = 0; t < nt; ++t) {
            char buf[32];
            snprintf(buf, sizeof(buf), "T%d=%d(%.0f%%)", t, type_hist[t], 100.0f * type_hist[t] / params_.N);
            hist_str += buf; hist_str += " ";
        }
        VIN_INFO("[DARTS-debug] type_logits softmax (temp=%.2f): mean_max_prob=%.3f mean_entropy=%.3f locked(>0.9)=%d/%d",
                 temp, mean_maxp, mean_ent, n_locked, params_.N);
        VIN_INFO("[DARTS-debug] type distribution (argmax): %s", hist_str.c_str());
    }

    std::vector<Shape> shapes;
    float inv_scale = 1.0f / scale_;

    int nan_count = 0, inf_count = 0, ok_count = 0;
    float min_angle = 1e30f, max_angle = -1e30f;

    for (int i = 0; i < params_.N; ++i) {
        // Logit -> linear: OptParams color_r/g/b/a are logits, convert to host ranges.
        float opacity = vin_sigmoid(params_.color_a[i]);
        float r = 255.0f * vin_sigmoid(params_.color_r[i]);
        float g = 255.0f * vin_sigmoid(params_.color_g[i]);
        float b = 255.0f * vin_sigmoid(params_.color_b[i]);

        // Determine best template type from logits (argmax)
        int best_type = 0;
        float best_logit = logits[i * nt + 0];
        for (int t = 1; t < nt; ++t) {
            if (logits[i * nt + t] > best_logit) {
                best_logit = logits[i * nt + t];
                best_type = t;
            }
        }

        // Map best_type to shape type / stamp name
        int shape_type;
        std::string stamp_name;
        int num_builtins = num_builtins_;
        if (best_type < num_builtins) {
            // 内置图元：映射到 FH6 shape type
            switch (best_type) {
                case 0: shape_type = SHAPE_SOFT_ELLIPSE_228; break;
                case 1: shape_type = SHAPE_ROTATED_ELLIPSE; break;
                case 2: shape_type = SHAPE_RECTANGLE; break;
                case 3: shape_type = SHAPE_TRIANGLE; break;
                default: shape_type = SHAPE_ROTATED_ELLIPSE; break;
            }
        } else {
            // 外部图章：type 设为第一个外部图章对应的 template index，stamp_name 用于 JSON 输出
            shape_type = best_type;
            int stamp_idx = best_type - num_builtins;
            if (stamp_idx >= 0 && stamp_idx < static_cast<int>(external_stamp_names_.size())) {
                stamp_name = external_stamp_names_[stamp_idx];
            }
        }

        Shape shape;
        shape.type = shape_type;
        shape.stamp_name = stamp_name;
        shape.cx = params_.cx[i] * inv_scale;
        shape.cy = params_.cy[i] * inv_scale;
        shape.rx = params_.rx[i] * inv_scale;
        shape.ry = params_.ry[i] * inv_scale;

        float angle_val = fmod(params_.angle[i], 360.0f);
        if (std::isnan(angle_val)) {
            nan_count++;
        } else if (std::isinf(angle_val)) {
            inf_count++;
        } else {
            ok_count++;
            min_angle = std::min(min_angle, angle_val);
            max_angle = std::max(max_angle, angle_val);
        }
        shape.angle = angle_val;

        shape.color = Color(r, g, b, 255.0f);
        shape.opacity = opacity;
        shape.z = params_.z[i];  // soft depth: 用于 preview 渲染排序
        shapes.push_back(shape);
    }

    VIN_INFO("params_to_shapes: angle stats — ok=%d nan=%d inf=%d  range=[%.2f, %.2f]",
             ok_count, nan_count, inf_count, min_angle, max_angle);

    // 进一步检查其他参数是否有 NaN
    int nan_cx = 0, nan_cy = 0, nan_rxry = 0;
    for (int i = 0; i < params_.N; ++i) {
        if (!std::isfinite(params_.cx[i])) nan_cx++;
        if (!std::isfinite(params_.cy[i])) nan_cy++;
        if (!std::isfinite(params_.rx[i]) || !std::isfinite(params_.ry[i])) nan_rxry++;
    }
    if (nan_cx > 0 || nan_cy > 0 || nan_rxry > 0) {
        VIN_WARN("params_to_shapes: other NaN params — cx=%d cy=%d rx/ry=%d",
                 nan_cx, nan_cy, nan_rxry);
    }

    return shapes;
}

} // namespace vinylizer
