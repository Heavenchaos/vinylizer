#include <opencv2/opencv.hpp>
#include <json.hpp>
#include <CLI11.hpp>

#include "common/types.h"
#include "common/logger.h"
#include "common/canvas.h"
#include "core/pipeline.h"
#include "output/json_writer.h"
#include "inject/inject.h"

#include <iostream>
#include <sstream>
#include <string>
#include <atomic>

int main(int argc, char* argv[]) {
    vinylizer::AsyncLogger::instance().set_level(vinylizer::LogLevel::INFO);

    CLI::App app{"Vinylizer C++ - FH6 Livery Vectorizer", "vinylizer"};

    // ── Subcommands ──────────────────────────────────────────
    auto* opt_cmd = app.add_subcommand("optimize", "Run vectorization optimization");
#ifdef _WIN32
    auto* inject_cmd = app.add_subcommand("inject", "Inject shapes into FH6 memory");
#endif
    auto* render_cmd = app.add_subcommand("render", "Static render from shapes.json (no optimization)");

    // ── Optimize options ─────────────────────────────────────
    std::string input_path;
    std::string output_path = "output.json";
    int num_shapes = 1500;
    // 9999 = 哨兵：opt_resolution = max(w,h,img) 只缩不放，任何 ≤9999 的图 scale=1.0 → 原生尺寸
    int canvas_w = 9999;
    int canvas_h = 9999;
    int total_iters = 5000;
    std::string lr_str = "1.0,0.0";   // 默认：lr 从 1.0 退火到 0.0（见下方 --lr-schedule 默认 cosine）
    std::string shape_type_str = "16";
    std::string lr_schedule = "cosine";
    int lr_warmup_iters = 0;           // --lr-warmup-iters: 前 N 步 lr 从 0 线性爬到 lr_start（0=关）
    float darts_lock_ratio = 1.0f;     // --darts-lock-ratio: 进度≥该值后切 PURE（硬 argmax）渲染
    std::string blur_sigma_str = "2.0,0.0";   // --blur-sigma start,end（模板 texel 单位）
    std::string blur_sigma_schedule = "cosine";
    int blur_sigma_warmup = 0;
    bool no_darts = false;
    bool round_robin = false;
    // 外部图章
    std::string stamp_files_str;
    bool enable_builtins_flag = false;
    int boundary_pad = -1;             // -1 = 按图像尺寸比例自动（见 --pad-ratio）；>=0 = 显式绝对 px
    float pad_ratio = 0.053f;          // 自动模式比例：pad = clamp(round(max(W,H)*ratio), min, max)
    int pad_min = 16;                  // 自动模式下限（px）
    int pad_max = 400;                 // 自动模式上限（px）
    bool alpha_loss = true;            // RGBA MSE（alpha 通道参与 loss）：默认 ON（对齐旧仓库恒定 4 通道）

    // Adam
    float adam_beta2 = 0.95f;          // 0.999 对 reset 后不友好，0.95 最优
    float max_grad_norm = 30.0f;        // 100 几乎不触发，30 最优

    // Uniform scale constraint
    bool uniform_scale = false;

    // 初始化策略
    std::string init_mode = "variance";
    bool detail_first = true;
    float var_min_r = 2.0f;
    float var_max_r = 50.0f;
    float init_opacity = 31.0f / 255.0f;   // 初始 opacity（linear [0,1]）
    // k-medoids 构造式几何初始化（研究定版参数）
    int km_grid = 4;
    float km_alpha = 5.0f;
    float km_radius_scale = 2.5f;
    int km_iters = 15;
    float km_opacity = 0.9f;
    std::string km_backend = "auto";
    // --init-shapes: 从 optimize 输出 json 加载初始形状继续优化（空=走 initializer，默认关闭等价 baseline）
    std::string init_shapes_path;
    std::string init_shapes_mode = "continue";  // continue=one-hot 锁定 type；reset=type_logits 置 0（真 keep-init）

    // Random seed
    int random_seed = 42;

    // 学习率因子（按参数组）。
    // 注意：color/opacity 已改为 logit 参数化，梯度含 sigmoid' 链式项，
    // 且 angle 的梯度在弧度空间（非 diffbmp 的 logit 空间）。
    std::string lr_factor_str = "1.0,1.0,1.0,1.0,1.0";

    opt_cmd->add_option("-i,--input", input_path, "Input image path")->required()->check(CLI::ExistingFile);
    opt_cmd->add_option("-o,--output", output_path, "Output JSON path");
    opt_cmd->add_option("-n,--num-shapes", num_shapes, "Number of shapes")->default_val("1500");
    opt_cmd->add_option("--width", canvas_w, "Canvas width (9999=native image size)")->default_val("9999");
    opt_cmd->add_option("--height", canvas_h, "Canvas height (9999=native image size)")->default_val("9999");
    opt_cmd->add_option("--total-iters", total_iters, "Total optimization iterations")->default_val("5000");
    opt_cmd->add_option("--lr", lr_str,
        "Learning rate schedule (start,end). Default 1.0,0.0 = anneal to zero "
        "(required: a constant lr leaves Adam's sign-step random walk running forever, "
        "so the loss plateaus and the last iterations cannot refine)")->default_val("1.0,0.0");
    opt_cmd->add_option("--lr-schedule", lr_schedule,
        "LR annealing schedule: cosine|linear|piecewise")->default_val("cosine");
    opt_cmd->add_option("--lr-warmup-iters", lr_warmup_iters,
        "LR warmup: linearly ramp lr from 0 to lr_start over first N iters "
        "(0=off; protects continue/fine-tune from Adam bias-correction full steps)")->default_val("0");
    opt_cmd->add_option("--shape-type", shape_type_str,
        "Shape type: 16=ellipse, 1=rect, 103=triangle, 228=soft_circle, or stamp filename")
        ->default_val("16");
    opt_cmd->add_option("--darts-lock-ratio", darts_lock_ratio,
        "Progress at which DARTS switches from the soft 4-type mixture to PURE hard-argmax "
        "rendering, so shapes keep being optimised under the exact model that gets exported. "
        "1.0=lock only on the last iteration (default), 0.0=pure search from the start. "
        "Measured: 0.4 and 1.0 are statistically indistinguishable; 0.7 was worst")->default_val("1.0");
    opt_cmd->add_option("--blur-sigma", blur_sigma_str,
        "Template blur sigma annealing (start,end), in TEMPLATE texels. "
        "Pixel-space blur = sigma * max(su,sv)/64, i.e. proportional to shape size. "
        "Default 2.0,0.0 = start soft (build structure), end perfectly sharp. "
        "Measured: 2.0,0.5 scores slightly better than 2.0,0.0 on test.jpg")->default_val("2.0,0.0");
    opt_cmd->add_option("--blur-sigma-schedule", blur_sigma_schedule,
        "Sigma annealing schedule: cosine|linear|piecewise|none")->default_val("cosine");
    opt_cmd->add_option("--blur-sigma-warmup", blur_sigma_warmup,
        "Hold the starting sigma for the first N iterations before annealing")->default_val("0");
    opt_cmd->add_flag("--no-darts", no_darts, "Disable DARTS type selection (use fixed --shape-type)");
    opt_cmd->add_flag("--round-robin", round_robin,
        "Round-robin type assignment: cycle through all stamps (implies --no-darts)");
    opt_cmd->add_option("--stamp-files", stamp_files_str,
        "External stamp image files (comma-separated; 相对当前工作目录 cwd)");
    opt_cmd->add_flag("--enable-builtins", enable_builtins_flag,
        "Keep built-in 4 primitives alongside external stamps");
    opt_cmd->add_option("--pad", boundary_pad,
        "Transparent padding pixels: explicit absolute override (0=disable). "
        "Default -1 = proportional: clamp(round(max(W,H)*pad-ratio), pad-min, pad-max)")
        ->default_val("-1");
    opt_cmd->add_option("--pad-ratio", pad_ratio,
        "Proportional padding ratio (used when --pad<0). Default 0.053 keeps the old "
        "100px on the 1186x1884 reference image; border-area fraction is size-independent")
        ->default_val("0.053");
    opt_cmd->add_option("--pad-min", pad_min,
        "Proportional padding lower bound in px (default 16)")->default_val("16");
    opt_cmd->add_option("--pad-max", pad_max,
        "Proportional padding upper bound in px (default 400)")->default_val("400");
    opt_cmd->add_flag("--alpha-loss,!--no-alpha-loss", alpha_loss,
        "RGBA MSE: alpha channel participates in the loss so transparent regions stay "
        "transparent (FH6). Default: on (matches legacy 4-channel behavior; "
        "--no-alpha-loss restores RGB-only MSE)");

    opt_cmd->add_option("--adam-beta2", adam_beta2,
        "Adam beta2 (2nd moment decay). 0.999 slow to adapt after reset, 0.95 optimal")->default_val("0.95");
    opt_cmd->add_option("--max-grad-norm", max_grad_norm,
        "Global gradient clip L2 norm. 100 never triggers (no clip), 30 optimal")->default_val("30");

    opt_cmd->add_flag("--uniform-scale", uniform_scale,
        "Force rx=ry (uniform scale constraint, like DiffBMP)");
    opt_cmd->add_option("--init-mode", init_mode,
        "Shape initialization strategy: 'kmedoids' (default, error-weighted "
        "k-medoids constructive geometry) | 'variance' (DiffBMP-style)")
        ->default_val("kmedoids")
        ->check(CLI::IsMember({"variance", "kmedoids"}));
    opt_cmd->add_option("--km-grid", km_grid,
        "kmedoids: grid step g in px (default 4)")->default_val("4");
    opt_cmd->add_option("--km-alpha", km_alpha,
        "kmedoids: weight exponent w=(0.1+0.9*var)^alpha (default 5)")->default_val("5");
    opt_cmd->add_option("--km-radius-scale", km_radius_scale,
        "kmedoids: radius scale applied to sqrt(cnt*cell_area/pi) (default 2.5)")
        ->default_val("2.5");
    opt_cmd->add_option("--km-iters", km_iters,
        "kmedoids: weighted Lloyd iterations (default 15)")->default_val("15");
    opt_cmd->add_option("--km-opacity", km_opacity,
        "kmedoids: initial opacity (linear 0-1, research-fixed 0.9)")
        ->default_val("0.9");
    opt_cmd->add_option("--km-backend", km_backend,
        "kmedoids: assign/d2 kernel backend: 'auto' (cuda if a CUDA device is "
        "available, else cpu) | 'cpu' | 'cuda'. CPU path is the bit-exact reference.")
        ->default_val("auto")
        ->check(CLI::IsMember({"auto", "cpu", "cuda"}));
    opt_cmd->add_option("--var-min-r", var_min_r,
        "VarianceInitializer min radius in px (default 2.0)")
        ->default_val("2.0");
    opt_cmd->add_option("--var-max-r", var_max_r,
        "VarianceInitializer max radius in px (default 50.0)")
        ->default_val("50.0");
    opt_cmd->add_option("--init-opacity", init_opacity,
        "Initial opacity (linear 0-1). 0.1216=baseline(31/255)")
        ->default_val("0.1215686274509804");
    opt_cmd->add_option("--init-shapes", init_shapes_path,
        "Load initial shapes from optimize-output JSON and continue optimizing "
        "(skip VarianceInitializer). Empty=default initializer (exactly baseline). "
        "N is determined by JSON shape count; -n is ignored if provided.")
        ->default_val("");
    opt_cmd->add_option("--init-shapes-mode", init_shapes_mode,
        "init-shapes mode: continue=one-hot lock json type (default, for resuming); "
        "reset=type_logits=0 uniform (true keep-init, DARTS re-search)")
        ->default_val("continue")
        ->check(CLI::IsMember({"continue", "reset"}));
    opt_cmd->add_flag("--detail-first,!--no-detail-first", detail_first,
        "z-order: small details on top (default: true, East-Asian style)");
    opt_cmd->add_option("--seed", random_seed,
        "Random seed for reproducibility (default: 42)")->default_val("42");
    opt_cmd->add_option("--lr-factor", lr_factor_str,
        "Per-param LR factors: cx_cy,rx_ry,angle,RGB,A (default: 1.0,1.0,1.0,1.0,1.0)")
        ->default_val("1.0,1.0,1.0,1.0,1.0");

    // ── Inject options (Windows-only: FH6 内存注入，Linux/云上无此功能) ─────
#ifdef _WIN32
    std::string json_path;
    int pid = 0;
    int layer_count = 0;

    inject_cmd->add_option("-j,--json", json_path, "JSON file path")->required()->check(CLI::ExistingFile);
    inject_cmd->add_option("-l,--layers", layer_count, "Number of placeholder layers in FH6")->required();
    inject_cmd->add_option("-p,--pid", pid, "FH6 process ID (0=auto)")->default_val("0");
#endif

    // ── Render-only options ─────────────────────────────────
    std::string render_shapes_path;
    std::string render_target_png_path;
    std::string render_mask_path;
    std::string render_output_path;
    std::string render_stamp_files_str;
    std::string render_remove_str;  // 逗号分隔的索引，或 a-b 范围
    int render_num_types = 4;
    int render_num_builtins = 0;
    float render_sigma = 0.0f;   // 模板模糊 sigma：必须 = 训练末态 sigma（训练默认已改为 2.0→0.0）

    render_cmd->add_option("-s,--shapes", render_shapes_path, "Path to shapes.json (optimize 输出)")->required()->check(CLI::ExistingFile);
    render_cmd->add_option("--target-png", render_target_png_path, "Target PNG path（INTER_AREA resize 与 optimize 口径一致）")->check(CLI::ExistingFile);
    render_cmd->add_option("-m,--mask", render_mask_path, "Path to mask.raw (for valid-pixel PSNR)");
    render_cmd->add_option("-o,--output", render_output_path,
        "Output path. .png = human-readable image (canvas size, no resize); anything else (e.g. .raw) = float32 H*W*3 dump. PSNR is always computed from the in-memory float buffer, never from this file.");
    render_cmd->add_option("--stamp-files", render_stamp_files_str, "External stamp image files (comma-separated; 相对当前工作目录 cwd)");
    render_cmd->add_option("--num-types", render_num_types, "Number of template types")->default_val("4");
    render_cmd->add_option("--num-builtins", render_num_builtins, "Number of built-in types (0=pure external)")->default_val("0");
    render_cmd->add_option("--sigma", render_sigma,
        "Template blur sigma for evaluation. MUST equal the training final sigma or the "
        "reported MSE will not match training. Training default is now 2.0->0.0, so the "
        "matching value is 0.0 (pass --sigma 1.0 to reproduce the historical caliber)")->default_val("0.0");
    render_cmd->add_option("--remove", render_remove_str, "Shape indices to remove (opacity=0). Comma-separated, supports a-b ranges, e.g. '0-149,300,500'");

    app.require_subcommand(1);

    CLI11_PARSE(app, argc, argv);

    if (opt_cmd->parsed()) {
        vinylizer::PipelineConfig config;
        config.input_path = input_path;
        config.output_path = output_path;
        config.num_shapes = num_shapes;
        config.canvas_w = canvas_w;
        config.canvas_h = canvas_h;
        config.total_iters = total_iters;

        // Parse lr schedule
        auto comma = lr_str.find(',');
        if (comma != std::string::npos) {
            config.lr_start = std::stof(lr_str.substr(0, comma));
            config.lr_end = std::stof(lr_str.substr(comma + 1));
        }
        config.lr_warmup_iters = lr_warmup_iters;
        config.lr_schedule = lr_schedule;
        // 解析 stamp-files
        std::vector<std::string> stamp_files;
        if (!stamp_files_str.empty()) {
            std::stringstream ss(stamp_files_str);
            std::string item;
            while (std::getline(ss, item, ',')) {
                item.erase(0, item.find_first_not_of(" \t"));
                item.erase(item.find_last_not_of(" \t") + 1);
                if (!item.empty()) stamp_files.push_back(item);
            }

            // 路径语义：所有相对路径均相对进程当前工作目录 (cwd) 解析，
            // 不做"相对输入图片目录"的隐式拼接（避免跨命令路径基准不一致）。
        }

        // 解析 shape-type（可能是数字或文件名）
        int shape_type = 16;
        try {
            shape_type = std::stoi(shape_type_str);
        } catch (...) {
            // 非数字：尝试匹配 stamp_files 中的文件名（支持完整路径或 basename）
            shape_type = -1;
            for (int i = 0; i < static_cast<int>(stamp_files.size()); ++i) {
                // 获取 stamp file 的 basename
                std::string stamp_basename = stamp_files[i];
                size_t last_sep = stamp_basename.find_last_of("/\\");
                if (last_sep != std::string::npos)
                    stamp_basename = stamp_basename.substr(last_sep + 1);
                // 比较输入的 shape_type_str 与 stamp basename 或完整路径
                if (shape_type_str == stamp_files[i] || shape_type_str == stamp_basename) {
                    shape_type = 4 + i;  // 内置 4 个 + 第 i 个外部图章
                    break;
                }
            }
            if (shape_type == -1) {
                VIN_WARN("--shape-type '%s' not recognized as number or stamp file, using default 16",
                         shape_type_str.c_str());
                shape_type = 16;
            }
        }

        bool enable_builtins = stamp_files.empty() ? true : enable_builtins_flag;
        int num_types = 4;
        if (!stamp_files.empty()) {
            num_types = enable_builtins ? (4 + (int)stamp_files.size()) : (int)stamp_files.size();
            // 如果 shape_type 是外部图章索引（>= 4），且未启用内置图元，需要调整索引
            // 内置图元启用时：索引 = 4 + i；禁用时：索引 = i
            if (shape_type >= 4 && !enable_builtins) {
                shape_type -= 4;
            }
        }

        config.shape_type = shape_type;
        config.stamp_files = stamp_files;
        config.enable_builtins = enable_builtins;
        config.num_template_types = num_types;
        config.num_builtins = enable_builtins ? 4 : 0;
        config.enable_darts = !no_darts && !round_robin;
        config.round_robin = round_robin;
        config.darts_lock_ratio = darts_lock_ratio;
        {
            const auto c2 = blur_sigma_str.find(',');
            if (c2 != std::string::npos) {
                config.blur_sigma_start = std::stof(blur_sigma_str.substr(0, c2));
                config.blur_sigma_end   = std::stof(blur_sigma_str.substr(c2 + 1));
            }
        }
        config.blur_sigma_schedule = blur_sigma_schedule;
        config.blur_sigma_warmup = blur_sigma_warmup;
        config.alpha_loss = alpha_loss;
        config.boundary_pad = boundary_pad;
        config.pad_ratio = pad_ratio;
        config.pad_min = pad_min;
        config.pad_max = pad_max;
        config.adam_beta2 = adam_beta2;
        config.max_grad_norm = max_grad_norm;
        config.uniform_scale = uniform_scale;
        config.init_mode = init_mode;
        config.detail_first = detail_first;
        config.var_min_r = var_min_r;
        config.var_max_r = var_max_r;
        config.init_opacity = init_opacity;
        config.km_grid = km_grid;
        config.km_alpha = km_alpha;
        config.km_radius_scale = km_radius_scale;
        config.km_iters = km_iters;
        config.km_opacity = km_opacity;
        config.km_backend = km_backend;
        config.init_shapes_path = init_shapes_path;
        config.init_shapes_mode = init_shapes_mode;
        config.random_seed = random_seed;

        // 解析 --lr-factor: cx_cy, rx_ry, angle, RGB, A
        {
            std::vector<float> factors;
            std::stringstream ss(lr_factor_str);
            std::string token;
            while (std::getline(ss, token, ',')) {
                token.erase(0, token.find_first_not_of(" \t"));
                token.erase(token.find_last_not_of(" \t") + 1);
                factors.push_back(std::stof(token));
            }
            if (factors.size() == 5) {
                config.lr_factor_xy = factors[0];
                config.lr_factor_r = factors[1];
                config.lr_factor_angle = factors[2];
                config.lr_factor_rgb = factors[3];
                config.lr_factor_a = factors[4];
            }
        }

        VIN_INFO("Starting optimization: input=%s, shapes=%d, total_iters=%d, seed=%d",
                 input_path.c_str(), num_shapes, total_iters, random_seed);

        vinylizer::run_pipeline(config,
            [&](float pct, float mse, int iter) {
                std::cout << "\rProgress: " << int(pct * 100) << "%  MSE: "
                          << int(mse) << "  iter: " << iter << "     " << std::flush;
            });

        VIN_INFO("Optimization complete: output=%s", output_path.c_str());

    }
#ifdef _WIN32
    else if (inject_cmd->parsed()) {
        vinylizer::InjectConfig config;
        config.json_path = json_path;
        config.process_id = pid;
        config.layer_count = layer_count;

        VIN_INFO("Injecting shapes: json=%s, pid=%d", json_path.c_str(), pid);

        if (vinylizer::inject_shapes(config)) {
            VIN_INFO("Injection successful");
        } else {
            VIN_ERROR("Injection failed");
            return 1;
        }
    }
#endif
    else if (render_cmd->parsed()) {
        // 解析 --remove 索引字符串（支持 "0-149,300,500" 格式）
        std::vector<int> remove_indices;
        if (!render_remove_str.empty()) {
            std::stringstream ss(render_remove_str);
            std::string token;
            while (std::getline(ss, token, ',')) {
                size_t dash = token.find('-');
                if (dash != std::string::npos && dash > 0 && dash < token.size() - 1) {
                    int a = std::stoi(token.substr(0, dash));
                    int b = std::stoi(token.substr(dash + 1));
                    for (int i = std::min(a, b); i <= std::max(a, b); ++i) remove_indices.push_back(i);
                } else {
                    remove_indices.push_back(std::stoi(token));
                }
            }
        }

        vinylizer::RenderOnlyConfig config;
        config.shapes_json_path = render_shapes_path;
        config.target_png_path = render_target_png_path;
        config.mask_raw_path = render_mask_path;
        config.output_raw_path = render_output_path;
        config.num_template_types = render_num_types;
        config.num_builtins = render_num_builtins;
        config.remove_indices = remove_indices;
        config.blur_sigma = render_sigma;

        // 解析 stamp files
        if (!render_stamp_files_str.empty()) {
            std::stringstream ss(render_stamp_files_str);
            std::string stamp;
            while (std::getline(ss, stamp, ',')) {
                // trim
                stamp.erase(0, stamp.find_first_not_of(" \t"));
                stamp.erase(stamp.find_last_not_of(" \t") + 1);
                if (!stamp.empty()) config.stamp_files.push_back(stamp);
            }
        }

        auto result = vinylizer::render_only(config);
        if (!result.ok) {
            VIN_ERROR("Render-only failed");
            return 1;
        }
        // 输出单行结果供脚本解析
        std::cout << "RENDER_RESULT psnr=" << result.psnr_db
                  << " mse=" << result.mse_0_255
                  << " N=" << result.N
                  << " used=" << result.N_used
                  << " removed=" << (result.N - result.N_used)
                  << " H=" << result.H << " W=" << result.W << std::endl;
    }

    return 0;
}
