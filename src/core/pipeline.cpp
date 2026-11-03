#include "core/pipeline.h"
#include "common/logger.h"
#include "common/path_utils.h"
#include "common/json_to_image.h"
#include "preprocess/preprocessor.h"
#include "core/optimizer.h"
#include "core/diff_renderer.h"
#include "output/json_writer.h"
#include <json.hpp>

// alpha_upper_bound setter（定义在 template_sampling.cu，用 __constant__ memory）
extern "C" void cuda_set_alpha_upper_bound(float aub);

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <limits>
#include <fstream>

namespace vinylizer {

// ---- 辅助函数（与 optimizer.cpp 一致，统一 FH6 type → template index 映射）----
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
            if (shape_type >= num_builtins) return shape_type;
            return 0;
    }
}

// 取路径 basename（用于字符串 type 匹配，兼容相对/绝对路径差异）
static std::string path_basename(const std::string& p) {
    size_t pos = p.find_last_of("/\\");
    return (pos == std::string::npos) ? p : p.substr(pos + 1);
}

void run_pipeline(const PipelineConfig& config,
                  std::function<void(float,float,int)> progress_cb,
                  std::function<void(GradientOptimizer*)> optimizer_ready_cb) {
    VIN_INFO("=== Vinylizer Pipeline ===");
    VIN_INFO("Input: %s", config.input_path.c_str());

    cv::Mat input = imread_utf8(config.input_path);
    if (input.empty()) {
        VIN_ERROR("Failed to load image: %s", config.input_path.c_str());
        return;
    }
    VIN_INFO("Image loaded: %dx%d, channels=%d", input.cols, input.rows, input.channels());

    // ── 透明 padding：防止形状超出画布边界 ──
    // 默认按图像尺寸比例（大图行为与旧固定 100px 一致，小图自动收缩）；--pad 可显式覆盖绝对值。
    int orig_w = input.cols, orig_h = input.rows;  // padding 前的原始尺寸
    int pad = effective_boundary_pad(config, orig_w, orig_h);
    if (config.boundary_pad < 0) {
        VIN_INFO("Boundary padding: proportional mode (ratio=%.4f, min=%d, max=%d) -> %dpx",
                 config.pad_ratio, config.pad_min, config.pad_max, pad);
    }
    if (pad > 0) {
        // 统一转 4 通道 BGRA（3 通道图补 alpha=255 全不透明）
        cv::Mat input_bgra;
        if (input.channels() == 3) {
            cv::cvtColor(input, input_bgra, cv::COLOR_BGR2BGRA);
        } else if (input.channels() == 4) {
            input_bgra = input.clone();
        } else {
            // 灰度等：先转 BGR 再转 BGRA
            cv::Mat tmp;
            cv::cvtColor(input, tmp, cv::COLOR_GRAY2BGR);
            cv::cvtColor(tmp, input_bgra, cv::COLOR_BGR2BGRA);
        }

        // 加透明 padding（BGRA=(0,0,0,0) = 透明黑）
        cv::copyMakeBorder(input_bgra, input,
                           pad, pad, pad, pad,
                           cv::BORDER_CONSTANT,
                           cv::Scalar(0, 0, 0, 0));
        VIN_INFO("Applied transparent padding: %dpx (canvas now %dx%d)",
                 pad, input.cols, input.rows);
    }

    Preprocessor preprocessor;
    auto prep_result = preprocessor.process(input);

    cv::Mat target_rgb;
    if (input.channels() == 4) {
        cv::cvtColor(input, target_rgb, cv::COLOR_BGRA2RGB);
    } else if (input.channels() == 3) {
        cv::cvtColor(input, target_rgb, cv::COLOR_BGR2RGB);
    } else {
        cv::cvtColor(input, target_rgb, cv::COLOR_GRAY2RGB);
    }

    int img_w = target_rgb.cols, img_h = target_rgb.rows;

    VIN_INFO("Background color: (255, 255, 255) — white background (align with diffbmp)");

    OptimizerConfig opt_config;
    opt_config.max_shapes = config.num_shapes;
    opt_config.opt_resolution = std::max({config.canvas_w, config.canvas_h, img_w, img_h});
    opt_config.total_iters = config.total_iters;
    opt_config.lr_start = config.lr_start;
    opt_config.lr_end = config.lr_end;
    opt_config.blur_sigma_start = config.blur_sigma_start;
    opt_config.blur_sigma_end = config.blur_sigma_end;
    opt_config.blur_sigma_warmup = config.blur_sigma_warmup;
    opt_config.blur_sigma_schedule = config.blur_sigma_schedule;
    opt_config.lr_schedule = config.lr_schedule;
    opt_config.lr_warmup_iters = config.lr_warmup_iters;
    opt_config.enable_darts = config.enable_darts;
    opt_config.round_robin = config.round_robin;
    opt_config.darts_temp_start = config.darts_temp_start;
    opt_config.darts_temp_end = config.darts_temp_end;
    opt_config.darts_lock_ratio = config.darts_lock_ratio;
    opt_config.adam_beta2 = config.adam_beta2;
    opt_config.max_grad_norm = config.max_grad_norm;
    opt_config.uniform_scale = config.uniform_scale;
    opt_config.init_mode = config.init_mode;
    opt_config.detail_first = config.detail_first;
    opt_config.var_min_r = config.var_min_r;
    opt_config.var_max_r = config.var_max_r;
    opt_config.init_opacity = config.init_opacity;
    opt_config.km_grid = config.km_grid;
    opt_config.km_alpha = config.km_alpha;
    opt_config.km_radius_scale = config.km_radius_scale;
    opt_config.km_iters = config.km_iters;
    opt_config.km_opacity = config.km_opacity;
    opt_config.km_backend = config.km_backend;
    opt_config.init_shapes_path = config.init_shapes_path;
    opt_config.init_shapes_mode = config.init_shapes_mode;
    opt_config.random_seed = config.random_seed;
    opt_config.deterministic_reduce = config.deterministic_reduce;
    opt_config.alpha_loss = config.alpha_loss;
    opt_config.lr_factor_xy = config.lr_factor_xy;
    opt_config.lr_factor_r = config.lr_factor_r;
    opt_config.lr_factor_angle = config.lr_factor_angle;
    opt_config.lr_factor_rgb = config.lr_factor_rgb;
    opt_config.lr_factor_a = config.lr_factor_a;

    GradientOptimizer optimizer(opt_config, config.shape_type);
    optimizer.set_num_template_types(config.num_template_types);
    optimizer.set_num_builtins(config.num_builtins);
    optimizer.set_external_stamp_names(config.stamp_files);
    optimizer.set_enable_builtins(config.enable_builtins);

    if (optimizer_ready_cb) {
        optimizer_ready_cb(&optimizer);
    }

    // 传递 pinned memory + event（GUI 线程用 cudaEventQuery 非阻塞等待）
    if (config.viz_pinned_a && config.viz_event_a &&
        config.viz_pinned_b && config.viz_event_b) {
        optimizer.set_viz_target(
            config.viz_pinned_a, config.viz_event_a,
            config.viz_pinned_b, config.viz_event_b);
    }
    if (config.viz_read_idx_atomic) {
        optimizer.set_viz_read_idx(config.viz_read_idx_atomic);
    }

    if (progress_cb) {
        optimizer.set_progress_callback([&](float pct, float mse, int iter) {
            if (config.stop_requested) optimizer.request_stop();
            progress_cb(pct, mse, iter);
        });
    }

    auto shapes = optimizer.optimize(
        target_rgb, prep_result.opaque_mask,
        prep_result.edge_map, prep_result.saliency_map,
        prep_result.dark_mask, prep_result.posterized,
        prep_result.palette, prep_result.grad_dir,
        Color(255, 255, 255, 255));  // 白色背景，与 Canvas/json_to_image 一致

    VIN_INFO("Optimization produced %d shapes", static_cast<int>(shapes.size()));

    // ── 还原坐标：减去 padding 偏移 ──
    if (pad > 0) {
        for (auto& s : shapes) {
            s.cx -= pad;
            s.cy -= pad;
        }
        VIN_INFO("Shifted %d shapes by (-%d, -%d) to original coords",
                 static_cast<int>(shapes.size()), pad, pad);
    }

    // ── 过滤"全透明形状"：对渲染无任何贡献，但会占用 FH6 有限的图层额度 ──
    // 阈值 = 0.5/255：FH6 的 alpha 是 8bit（输出 JSON 写的是 opacity*255），
    // 低于该值的形状在游戏里量化后 alpha==0，即完全不可见。
    // 注意：输出形状数会因此小于请求的 N（日志里明确报告过滤个数）。
    {
        const int n_before = static_cast<int>(shapes.size());
        const float kMinOpacity = 0.5f / 255.0f;
        shapes.erase(std::remove_if(shapes.begin(), shapes.end(),
                                    [kMinOpacity](const Shape& s) {
                                        return s.opacity < kMinOpacity;
                                    }),
                     shapes.end());
        const int n_removed = n_before - static_cast<int>(shapes.size());
        if (n_removed > 0) {
            VIN_INFO("Filtered %d fully-transparent shapes (opacity < %.5f = 0.5/255): "
                     "%d -> %d shapes", n_removed, kMinOpacity,
                     n_before, static_cast<int>(shapes.size()));
        }
    }

    write_json(config.output_path, shapes, orig_w, orig_h, config.stamp_files);

    // 从 JSON 重新渲染预览图（确保 PNG 与 JSON 完全一致）
    std::string preview_path = config.output_path + ".preview.png";
    VIN_INFO("Pipeline: calling json_to_image for preview: %s", config.output_path.c_str());
    cv::Mat preview_rgb = json_to_image(config.output_path);
    VIN_INFO("Pipeline: json_to_image returned, empty=%d, size=%dx%d",
             preview_rgb.empty(), preview_rgb.cols, preview_rgb.rows);
    if (!preview_rgb.empty()) {
        // OpenCV 要求 BGR 顺序 → 显式转换
        cv::Mat preview_bgr;
        cv::cvtColor(preview_rgb, preview_bgr, cv::COLOR_RGB2BGR);
        std::vector<uint8_t> buf;
        cv::imencode(".png", preview_bgr, buf);
        write_file_bytes(preview_path, buf.data(), buf.size());
        VIN_INFO("Preview saved (from JSON): %s", preview_path.c_str());
    } else {
        VIN_WARN("Cannot render preview from JSON");
    }

    VIN_INFO("=== Pipeline Complete ===");
}

// === Render-only mode ===
RenderOnlyResult render_only(const RenderOnlyConfig& cfg) {
    RenderOnlyResult result;
    VIN_INFO("=== Render-only Mode ===");

    // 1. 加载 shapes.json
    std::ifstream sf = open_ifstream(cfg.shapes_json_path);
    if (!sf.is_open()) {
        VIN_ERROR("Cannot open shapes.json: %s", cfg.shapes_json_path.c_str());
        return result;
    }
    nlohmann::json data = nlohmann::json::parse(sf);
    // shapes.json 可能是 list（diagnosis 格式）或 dict with "shapes"（output.json 格式）
    nlohmann::json shapes_arr;
    if (data.is_array()) {
        shapes_arr = data;
    } else if (data.contains("shapes") && data["shapes"].is_array()) {
        shapes_arr = data["shapes"];
    } else {
        VIN_ERROR("Invalid shapes.json format (expected array or {shapes:[]})");
        return result;
    }
    int N = (int)shapes_arr.size();
    result.N = N;
    if (N == 0) {
        VIN_ERROR("No shapes in JSON");
        return result;
    }

    // 2. 读 H, W：优先从 shapes.json 的 canvas_w/canvas_h（optimize 输出格式），
    //    fallback 到同目录 meta.json（diagnosis 格式）
    int H = 0, W = 0;
    if (data.contains("canvas_w") && data.contains("canvas_h")) {
        W = data["canvas_w"].get<int>();
        H = data["canvas_h"].get<int>();
    }
    if (H <= 0 || W <= 0) {
        std::string diag_dir = cfg.shapes_json_path;
        size_t last_slash = diag_dir.find_last_of("/\\");
        if (last_slash != std::string::npos) diag_dir = diag_dir.substr(0, last_slash);
        std::ifstream mf(diag_dir + "/meta.json");
        nlohmann::json meta;
        if (mf.is_open()) mf >> meta;
        H = meta.value("H", 0), W = meta.value("W", 0);
    }
    if (H <= 0 || W <= 0) {
        VIN_ERROR("Cannot read H/W from shapes.json (canvas_w/h) or meta.json");
        return result;
    }
    result.H = H; result.W = W;

    // 3. 构造 ShapeParams（linear 值，upload_params_gpu 内部转 logit）
    //    支持 optimize 输出格式（data[5]+color[4]+type）和 diagnosis 格式（扁平字段+type_argmax）
    std::vector<ShapeParams> shapes(N);
    std::vector<int> types(N);
    std::vector<bool> removed(N, false);
    for (int i : cfg.remove_indices) {
        if (i >= 0 && i < N) removed[i] = true;
    }

    bool is_output_format = (!shapes_arr.empty() && shapes_arr[0].contains("data"));
    for (int i = 0; i < N; ++i) {
        auto& s = shapes_arr[i];
        if (is_output_format) {
            // optimize 输出格式: data=[cx,cy,rx,ry,angle], color=[r,g,b,a], type=str|int
            const auto& d = s["data"];
            shapes[i].cx = d[0].get<float>();
            shapes[i].cy = d[1].get<float>();
            shapes[i].rx = d[2].get<float>();
            shapes[i].ry = d[3].get<float>();
            shapes[i].angle = d[4].get<float>();
            const auto& c = s["color"];
            shapes[i].color_r = c[0].get<float>();
            shapes[i].color_g = c[1].get<float>();
            shapes[i].color_b = c[2].get<float>();
            shapes[i].opacity = removed[i] ? 0.0f : c[3].get<float>() / 255.0f;
            // type: 字符串→stamp_files basename 匹配；整数→shape_type_to_template_idx
            if (s["type"].is_string()) {
                std::string stamp_name = s["type"].get<std::string>();
                std::string sn_base = path_basename(stamp_name);
                int tidx = -1;
                for (int k = 0; k < (int)cfg.stamp_files.size(); ++k) {
                    if (path_basename(cfg.stamp_files[k]) == sn_base) { tidx = k; break; }
                }
                types[i] = (tidx >= 0) ? tidx : 0;
            } else {
                types[i] = shape_type_to_template_idx(s["type"].get<int>(), cfg.num_builtins);
            }
        } else {
            // diagnosis 格式: cx/cy/rx/ry/angle/r/g/b/opacity/type_argmax
            shapes[i].cx = s["cx"].get<float>();
            shapes[i].cy = s["cy"].get<float>();
            shapes[i].rx = s["rx"].get<float>();
            shapes[i].ry = s["ry"].get<float>();
            shapes[i].angle = s["angle"].get<float>();
            shapes[i].color_r = s["r"].get<float>();
            shapes[i].color_g = s["g"].get<float>();
            shapes[i].color_b = s["b"].get<float>();
            shapes[i].opacity = removed[i] ? 0.0f : s["opacity"].get<float>();
            types[i] = s["type_argmax"].get<int>();
        }
    }
    result.N_used = N - (int)cfg.remove_indices.size();

    if (cfg.verbose) {
        VIN_INFO("Loaded %d shapes (%d used, %d removed), canvas %dx%d",
                 N, result.N_used, N - result.N_used, W, H);
    }

    // 4. 初始化 DiffRenderer
    DiffRenderer renderer;
    renderer.init(W, H, N, cfg.num_template_types, cfg.num_builtins);
    if (!cfg.stamp_files.empty()) {
        renderer.load_external_stamps(cfg.stamp_files);
    }
    renderer.set_bg_color(255.0f, 255.0f, 255.0f);
    renderer.set_blur_sigma(cfg.blur_sigma);  // [研究插桩移植] 默认 0.0=无模糊（原始行为）
    // alpha_upper_bound（对齐训练时的 cap，render 评估须与训练一致）
    if (cfg.alpha_upper_bound < 1.0f) {
        cuda_set_alpha_upper_bound(cfg.alpha_upper_bound);
    }
    renderer.set_render_mode(0);    // PURE mode (force_lock 所有类型)
    renderer.lock_types(types.data(), N);
    renderer.upload_params_gpu(shapes.data(), N);
    // 上传 type_logits（构造 one-hot：正确类型=10，其他=-10，确保 softmax argmax 正确）
    {
        std::vector<float> type_logits(N * cfg.num_template_types, -10.0f);
        for (int i = 0; i < N; ++i) {
            int t = std::max(0, std::min(types[i], cfg.num_template_types - 1));
            type_logits[i * cfg.num_template_types + t] = 10.0f;
        }
        renderer.upload_type_logits(type_logits.data(), N);
    }

    // 5. 加载 target：优先 --target-png（INTER_AREA，与 optimize 口径一致），fallback -t raw
    std::vector<float> h_target(H * W * 3, 0.0f);
    std::vector<float> h_mask(H * W, 1.0f);
    bool has_target = false;
    if (!cfg.target_png_path.empty()) {
        // PNG 路径：imread + INTER_AREA resize（与 optimize/prune 完全一致口径）
        cv::Mat img = imread_utf8(cfg.target_png_path, cv::IMREAD_UNCHANGED);
        if (img.empty()) {
            VIN_ERROR("Cannot load target PNG: %s", cfg.target_png_path.c_str());
            return result;
        }
        cv::Mat target_rgb;
        if (img.channels() == 4)      cv::cvtColor(img, target_rgb, cv::COLOR_BGRA2RGB);
        else if (img.channels() == 3) cv::cvtColor(img, target_rgb, cv::COLOR_BGR2RGB);
        else                          cv::cvtColor(img, target_rgb, cv::COLOR_GRAY2RGB);
        cv::Mat target_small;
        cv::resize(target_rgb, target_small, {W, H}, 0, 0, cv::INTER_AREA);
        for (int p = 0; p < H * W; ++p) {
            h_target[p * 3 + 0] = target_small.at<cv::Vec3b>(p / W, p % W)[0];
            h_target[p * 3 + 1] = target_small.at<cv::Vec3b>(p / W, p % W)[1];
            h_target[p * 3 + 2] = target_small.at<cv::Vec3b>(p / W, p % W)[2];
        }
        has_target = true;
    } else if (!cfg.target_raw_path.empty()) {
        // raw 路径：直接读 [H*W*3 float]，不做 resize（兼容旧脚本）
        std::ifstream tf(cfg.target_raw_path, std::ios::binary);
        if (!tf.is_open()) {
            VIN_ERROR("Cannot open target.raw: %s", cfg.target_raw_path.c_str());
            return result;
        }
        tf.read(reinterpret_cast<char*>(h_target.data()), h_target.size() * sizeof(float));
        has_target = true;
    }
    if (!cfg.mask_raw_path.empty()) {
        std::ifstream mf2(cfg.mask_raw_path, std::ios::binary);
        if (mf2.is_open()) mf2.read(reinterpret_cast<char*>(h_mask.data()),
                                    h_mask.size() * sizeof(float));
    }
    renderer.set_target(h_target.data(), h_mask.data(), nullptr);

    // 6. 渲染（render_forward_backward 完成 forward + backward，我们只用 forward 的 d_output）
    renderer.render_forward_backward(N, nullptr, 0, 0.5f, false);

    // 7. download_output
    std::vector<float> rendered(H * W * 3);
    renderer.download_output(rendered.data());

    // 8. 输出渲染结果（可选）：扩展名 .png → uint8 PNG（人眼可读，canvas 尺寸、无 resize，
    //    与 optimize 训练画布逐像素对齐）；其他（含 .raw）→ 原样 float dump。
    //    铁律：PSNR 始终从上方 float buffer 计算（第 9 步），绝不回读输出文件——
    //    uint8 量化偏差会破坏与 _eval_r12.py 的 Δ=0.000 口径门禁。
    if (!cfg.output_raw_path.empty()) {
        size_t dot = cfg.output_raw_path.find_last_of('.');
        std::string ext = (dot == std::string::npos) ? "" : cfg.output_raw_path.substr(dot + 1);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == "png") {
            cv::Mat img(H, W, CV_8UC3);
            for (int y = 0; y < H; ++y) {
                for (int x = 0; x < W; ++x) {
                    int p = y * W + x;
                    for (int c = 0; c < 3; ++c) {
                        float v = std::max(0.0f, std::min(255.0f, rendered[p * 3 + c]));
                        // rendered 为 RGB 内存序，OpenCV imencode 期望 BGR → 翻转通道
                        img.at<cv::Vec3b>(y, x)[2 - c] = (uchar)std::lroundf(v);
                    }
                }
            }
            std::vector<uchar> enc;
            std::vector<int> params{cv::IMWRITE_PNG_COMPRESSION, 3};
            if (cv::imencode(".png", img, enc, params)) {
                write_file_bytes(cfg.output_raw_path, enc.data(), enc.size());
                VIN_INFO("Rendered PNG saved: %s (%dx%d, canvas size, no resize)",
                         cfg.output_raw_path.c_str(), W, H);
            } else {
                VIN_ERROR("PNG encode failed: %s", cfg.output_raw_path.c_str());
            }
        } else {
            // 其他扩展名（含 .raw）→ float32 dump。
            // 通道序必须是 BGR：`rendered` 是 RGB 内存序，而本工程里"读回渲染结果"的
            // 一切下游（cv2.imread 的默认约定、以及 optimize 的训练损失本身）都按 BGR
            // 解读。此前这里原样 dump，导致 `-o x.raw` 与 `-o x.png` 对同一份 shapes
            // 给出不同的 MSE（实测 1102 vs 246，差 4.5 倍），且按像素核对可见红蓝互换
            // —— 那正是"预览比导出的颜色鲜艳"的真正原因。
            std::vector<float> bgr(rendered.size());
            for (size_t p = 0; p < rendered.size(); p += 3) {
                bgr[p + 0] = rendered[p + 2];
                bgr[p + 1] = rendered[p + 1];
                bgr[p + 2] = rendered[p + 0];
            }
            std::ofstream out(cfg.output_raw_path, std::ios::binary);
            out.write(reinterpret_cast<char*>(bgr.data()), bgr.size() * sizeof(float));
            VIN_INFO("Rendered saved: %s (float32 BGR, canvas size)", cfg.output_raw_path.c_str());
        }
    }

    // 9. 计算 PSNR（相对 target，mask 内；复用已加载的 h_target/h_mask）
    if (has_target) {
        double se = 0.0; long long cnt = 0;
        for (int p = 0; p < H * W; ++p) {
            if (h_mask[p] < 0.5f) continue;
            for (int c = 0; c < 3; ++c) {
                float d = rendered[p * 3 + c] - h_target[p * 3 + c];
                se += (double)d * d;
                cnt++;
            }
        }
        if (cnt > 0) {
            double mse = se / cnt;
            result.mse_0_255 = (float)mse;
            result.psnr_db = (float)(10.0 * log10(255.0 * 255.0 / std::max(mse, 1e-12)));
            if (cfg.verbose) {
                VIN_INFO("PSNR = %.4f dB (MSE=%.4f, valid_pixels=%lld/%d)",
                         result.psnr_db, result.mse_0_255, cnt / 3, H * W);
            }
        }
    }

    result.ok = true;
    return result;
}


} // namespace vinylizer
