#include "core/pipeline.h"
#include "common/logger.h"
#include "common/canvas.h"
#include "common/path_utils.h"
#include "preprocess/preprocessor.h"
#include "core/optimizer.h"
#include "output/json_writer.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>

namespace vinylizer {

static cv::Mat imread_utf8(const std::string& path, int flags = cv::IMREAD_UNCHANGED) {
    auto buf = read_file_bytes(path);
    if (buf.empty()) return {};
    return cv::imdecode(buf, flags);
}

static bool imwrite_utf8(const std::string& path, const cv::Mat& img,
                          const std::vector<int>& params = {}) {
    std::vector<uint8_t> buf;
    std::string ext = path.substr(path.rfind('.'));
    if (!cv::imencode(ext, img, buf, params)) return false;
    return write_file_bytes(path, buf.data(), buf.size());
}

void run_pipeline(const PipelineConfig& config) {
    VIN_INFO("=== Vinylizer Pipeline ===");
    VIN_INFO("Input: %s", config.input_path.c_str());

    cv::Mat input = imread_utf8(config.input_path);
    if (input.empty()) {
        VIN_ERROR("Failed to load image: %s", config.input_path.c_str());
        return;
    }
    VIN_INFO("Image loaded: %dx%d, channels=%d", input.cols, input.rows, input.channels());

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

    auto [bg_r, bg_g, bg_b] = compute_bg_color_from_result(
        prep_result.posterized, prep_result.opaque_mask,
        prep_result.palette, prep_result.rgb_image);
    VIN_INFO("Background color: (%d, %d, %d)", bg_r, bg_g, bg_b);

    OptimizerConfig opt_config;
    opt_config.max_shapes = config.num_shapes;
    opt_config.opt_resolution = std::max({config.canvas_w, config.canvas_h, img_w, img_h});
    opt_config.total_iters = config.num_cycles * 200;
    opt_config.lr_start = config.lr_start;
    opt_config.lr_end = config.lr_end;

    GradientOptimizer optimizer(opt_config);

    auto shapes = optimizer.optimize(
        target_rgb, prep_result.opaque_mask,
        prep_result.edge_map, prep_result.saliency_map,
        prep_result.dark_mask, prep_result.posterized,
        prep_result.palette, prep_result.grad_dir,
        Color(bg_r, bg_g, bg_b, 255));

    VIN_INFO("Optimization produced %d shapes", static_cast<int>(shapes.size()));

    write_json(config.output_path, shapes, img_w, img_h);

    Canvas canvas(img_w, img_h);
    canvas.add_shapes(shapes);
    std::string preview_path = config.output_path + ".preview.png";
    imwrite_utf8(preview_path, canvas.clone_image());
    VIN_INFO("Preview saved: %s", preview_path.c_str());

    VIN_INFO("=== Pipeline Complete ===");
}

} // namespace vinylizer
