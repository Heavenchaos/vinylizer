#pragma once

#include "common/types.h"
#include "core/diff_renderer.h"
#include <opencv2/core.hpp>
#include <vector>
#include <string>
#include <random>
#include <functional>
#include <atomic>
#include <mutex>

namespace vinylizer {

struct OptimizerConfig {
    int max_shapes = 500;
    int opt_resolution = 128;
    float lr_start = 1.0f;
    float lr_end = 0.1f;
    float init_radius_hi = 10.0f;   // % of long edge
    float init_radius_lo = 3.0f;
    float reloc_radius_hi = 1.5f;
    float reloc_radius_lo = 0.5f;
    int total_iters = 2000;

    // Cyclic Relocation
    int reloc_cycle_global = 150;
    int reloc_cycle_local = 50;
    int reloc_stability_window = 50;
    float reloc_stability_threshold = 0.10f;
    float reloc_useless_threshold = 0.01f;
};

// Optimizer parameters (SoA layout for GPU transfer)
struct OptParams {
    int N = 0;
    std::vector<float> cx, cy, rx, ry, angle;
    std::vector<float> color_r, color_g, color_b, color_a;  // color_a = opacity * 255

    void init(int n);
    ShapeParams get_shape(int i) const;
    void set_shape(int i, const ShapeParams& sp);

    // Convert all params to ShapeParams array (for GPU upload)
    std::vector<ShapeParams> to_shape_params() const {
        std::vector<ShapeParams> out(N);
        for (int i = 0; i < N; ++i) out[i] = get_shape(i);
        return out;
    }
};

class GradientOptimizer {
public:
    GradientOptimizer(const OptimizerConfig& config);

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

    // Progress callback
    using ProgressCallback = std::function<void(float progress, float mse, int iter)>;
    void set_progress_callback(ProgressCallback cb) { progress_cb_ = cb; }

    // Get current canvas for visualization (renders on demand)
    cv::Mat get_canvas() const;

private:
    // Initialization
    cv::Mat build_importance_map();
    void initialize_shapes();
    void shapes_to_params(const std::vector<Shape>& shapes);
    std::vector<Shape> params_to_shapes();

    // Optimization loop
    void gradient_optimize();
    float run_single_iter(OptParams& params,
                          std::vector<float>& grads, int iter, int total_iters,
                          float* d_grad_history = nullptr, int write_idx = 0,
                          bool local_phase = false,
                          const std::vector<int>& frozen_indices = {});

    // Cyclic Relocation
    void find_relocation_candidates(const std::vector<float>& grads,
                                    const std::vector<float>& visibility,
                                    std::vector<int>& candidates,
                                    int& n_useful, int& n_useless, int& n_unstable);
    std::vector<int> relocate_shapes(OptParams& params,
                                       const std::vector<int>& indices);
    float run_local_phase(OptParams& params,
                          const std::vector<int>& reloc_indices, int num_iters);

    // Utility
    std::tuple<int,int,int> get_local_color(float x, float y);
    std::tuple<int,int,int> get_posterized_color(float x, float y);

    OptimizerConfig config_;
    cv::Mat target_rgb_, mask_, edge_map_, saliency_map_, dark_mask_;
    cv::Mat posterized_small_;
    std::vector<std::tuple<int,int,int>> palette_;

    // Optimization resolution
    float scale_ = 1.0f;
    int H_opt_ = 0, W_opt_ = 0;
    cv::Mat target_small_, mask_small_;

    // Renderer
    DiffRenderer renderer_;

    // Current state
    OptParams params_;
    int adam_step_ = 0;
    std::atomic<bool> stop_requested_{false};
    ProgressCallback progress_cb_;

    // Async visualization pipeline (matching Python v5)
    uint8_t* viz_pinned_ = nullptr;     // CPU pinned memory [H_opt * W_opt * 3] uint8 RGB
    cudaEvent_t viz_event_ = nullptr;   // CUDA event for async sync
    bool viz_first_frame_ = true;       // first frame marker
    mutable std::mutex viz_mutex_;      // protects canvas access
    cv::Mat canvas_rgb_[2];             // double-buffered RGB canvas
    std::atomic<int> viz_write_idx_{0}; // which buffer is being written

    // Relocation state
    int reloc_total_ = 0;
    bool reloc_converged_ = false;

    // Best-MSE cycle snapshot tracking
    float best_mse_ = 1e20f;
    bool has_best_ = false;
    std::vector<float> best_params_flat_;
    float* snap_pinned_ = nullptr;
    cudaEvent_t snap_event_ = nullptr;
    float cycle_mse_ = 0.0f;
    bool snap_pending_ = false;

    // Background color for canvas rendering
    Color bg_color_{0, 0, 0, 255};

    // GPU gradient history ring buffer
    float* d_grad_history_ = nullptr;  // [WINDOW, N]
    int history_write_idx_ = 0;
    int history_count_ = 0;

    // Random
    std::mt19937 rng_{42};
};

} // namespace vinylizer
