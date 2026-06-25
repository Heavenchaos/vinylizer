#include "gui/optimize_worker.h"
#include "common/logger.h"
#include "common/path_utils.h"
#include "core/optimizer.h"
#include "core/pipeline.h"
#include "preprocess/preprocessor.h"
#include "output/json_writer.h"

#include <QElapsedTimer>
#include <opencv2/opencv.hpp>

namespace vinylizer {

OptimizeWorker::OptimizeWorker(const PipelineConfig& config, QObject* parent)
    : QThread(parent), config_(config) {}

void OptimizeWorker::run() {
    // Reset CUDA context to ensure clean state (in case previous run corrupted it)
    cudaDeviceReset();

    // 1. Load image (preserve alpha channel if present)
    auto img_bytes = read_file_bytes(config_.input_path);
    if (img_bytes.empty()) {
        VIN_ERROR("Cannot load image: %s", config_.input_path.c_str());
        emit done();
        return;
    }
    cv::Mat img = cv::imdecode(img_bytes, cv::IMREAD_UNCHANGED);
    if (img.empty()) {
        VIN_ERROR("Cannot decode image: %s", config_.input_path.c_str());
        emit done();
        return;
    }

    // Normalize image format: force 8-bit, valid channel count
    if (img.depth() != CV_8U) {
        VIN_WARN("Image depth is %d, converting to 8-bit", img.depth());
        double scale = (img.depth() == CV_16U) ? 1.0 / 257.0 : 255.0 / 65535.0;
        img.convertTo(img, CV_8U, scale);
    }
    if (img.channels() == 1) {
        cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
    } else if (img.channels() != 3 && img.channels() != 4) {
        VIN_ERROR("Unsupported channel count: %d", img.channels());
        emit done();
        return;
    }

    // 2. Compute opt_resolution and scale
    int W = img.cols, H = img.rows;
    int opt_res = std::max(config_.canvas_w, config_.canvas_h);
    float scale = std::min(1.0f, static_cast<float>(opt_res) / std::max(W, H));
    int Wo = std::max(1, static_cast<int>(W * scale));
    int Ho = std::max(1, static_cast<int>(H * scale));

    // 3. Resize image (alpha-aware for 4-channel images)
    cv::Mat img_small;
    if (img.channels() == 4) {
        // Separate BGR and alpha, resize independently to preserve alpha boundaries
        std::vector<cv::Mat> channels;
        cv::split(img, channels);  // B, G, R, A
        cv::Mat alpha = channels[3].clone();
        channels.pop_back();  // B, G, R
        std::vector<cv::Mat> small_channels(4);
        for (int c = 0; c < 3; c++) {
            cv::resize(channels[c], small_channels[c], cv::Size(Wo, Ho), 0, 0, cv::INTER_AREA);
        }
        cv::resize(alpha, small_channels[3], cv::Size(Wo, Ho), 0, 0, cv::INTER_NEAREST);
        cv::merge(small_channels, img_small);
    } else {
        cv::resize(img, img_small, cv::Size(Wo, Ho), 0, 0, cv::INTER_AREA);
    }

    // 4. Preprocess
    Preprocessor preprocessor;
    auto result = preprocessor.process(img_small);

    // 5. Compute bg_color from posterized label counts
    auto [br, bg, bb] = compute_bg_color_from_result(
        result.posterized, result.opaque_mask, result.palette, result.rgb_image);
    Color bg_color(static_cast<uint8_t>(br), static_cast<uint8_t>(bg), static_cast<uint8_t>(bb), 255);

    // 6. Create OptimizerConfig
    OptimizerConfig opt_cfg;
    opt_cfg.max_shapes = config_.num_shapes;
    opt_cfg.opt_resolution = std::max(Wo, Ho);
    opt_cfg.lr_start = config_.lr_start;
    opt_cfg.lr_end = config_.lr_end;
    opt_cfg.total_iters = config_.num_cycles * 200;

    GradientOptimizer optimizer(opt_cfg);

    // 7. Set progress callback
    QElapsedTimer frame_timer;
    frame_timer.start();

    optimizer.set_progress_callback([&](float progress, float mse, int iter) {
        if (this->stop_requested_) {
            optimizer.request_stop();
            return;
        }
        emit this->progress(progress, mse, iter);

        // Send frame at ~10fps
        if (frame_timer.elapsed() > 100) {
            cv::Mat rgb = optimizer.get_canvas(); // RGB (direct from CUDA)
            if (!rgb.empty()) {
                QImage qimg(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
                emit this->frame_ready(qimg.copy());
            }
            frame_timer.start();
        }
    });

    // 8. Run optimization
    auto shapes = optimizer.optimize(
        result.rgb_image, result.opaque_mask, result.edge_map,
        result.saliency_map, result.dark_mask, result.posterized,
        result.palette, result.grad_dir, bg_color);

    // 9. Scale shapes from opt resolution to original resolution
    float sx = static_cast<float>(W) / static_cast<float>(Wo);
    float sy = static_cast<float>(H) / static_cast<float>(Ho);
    for (auto& s : shapes) {
        s.cx *= sx; s.cy *= sy; s.rx *= sx; s.ry *= sy;
    }

    // 10. Write JSON
    if (!shapes.empty()) {
        write_json(config_.output_path, shapes, W, H);
    }

    // 11. Done
    emit done();
}

} // namespace vinylizer
