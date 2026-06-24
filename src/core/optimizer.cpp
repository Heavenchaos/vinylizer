#include "core/optimizer.h"
#include "common/logger.h"
#include "common/canvas.h"

#include <opencv2/imgproc.hpp>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <chrono>

namespace vinylizer {

// ================================================================
// OptParams
// ================================================================

void OptParams::init(int n) {
    N = n;
    cx.resize(n); cy.resize(n); rx.resize(n); ry.resize(n); angle.resize(n);
    color_r.resize(n); color_g.resize(n); color_b.resize(n); color_a.resize(n);
}

ShapeParams OptParams::get_shape(int i) const {
    ShapeParams sp;
    sp.cx = cx[i]; sp.cy = cy[i]; sp.rx = rx[i]; sp.ry = ry[i];
    sp.angle = angle[i];
    sp.color_r = color_r[i]; sp.color_g = color_g[i]; sp.color_b = color_b[i];
    sp.opacity = color_a[i] / 255.0f;
    return sp;
}

void OptParams::set_shape(int i, const ShapeParams& sp) {
    cx[i] = sp.cx; cy[i] = sp.cy; rx[i] = sp.rx; ry[i] = sp.ry;
    angle[i] = sp.angle;
    color_r[i] = sp.color_r; color_g[i] = sp.color_g;
    color_b[i] = sp.color_b; color_a[i] = sp.opacity * 255.0f;
}

// ================================================================
// GradientOptimizer
// ================================================================

GradientOptimizer::GradientOptimizer(const OptimizerConfig& config)
    : config_(config) {}

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
    renderer_.init(W_opt_, H_opt_, config_.max_shapes);
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
                // Transparent pixels: target RGB = bg_color (so rendered bg matches)
                h_target[idx * 3 + 0] = static_cast<float>(bg_color.r);
                h_target[idx * 3 + 1] = static_cast<float>(bg_color.g);
                h_target[idx * 3 + 2] = static_cast<float>(bg_color.b);
            }
            h_mask_gpu[idx] = opaque ? 1.0f : 0.0f;
            h_target_alpha[idx] = opaque ? 255.0f : 0.0f;
        }
    }
    renderer_.set_target(h_target.data(), h_mask_gpu.data(), h_target_alpha.data());

    // Initialize shapes
    initialize_shapes();

    // Run optimization
    gradient_optimize();

    // Convert back to shapes (best params already restored inside gradient_optimize)
    return params_to_shapes();
}

// ================================================================
// Initialization
// ================================================================

cv::Mat GradientOptimizer::build_importance_map() {
    cv::Mat importance = cv::Mat::ones(H_opt_, W_opt_, CV_32FC1);

    if (!edge_map_.empty()) {
        cv::Mat edge_small;
        cv::resize(edge_map_, edge_small, {W_opt_, H_opt_}, 0, 0, cv::INTER_LINEAR);
        cv::dilate(edge_small, edge_small, cv::getStructuringElement(cv::MORPH_RECT, {3, 3}));
        cv::Mat weighted;
        edge_small.convertTo(weighted, CV_32FC1, 3.0 / 255.0, 1.0);
        importance = importance.mul(weighted);
    }

    if (!saliency_map_.empty()) {
        cv::Mat sal_small;
        cv::resize(saliency_map_, sal_small, {W_opt_, H_opt_}, 0, 0, cv::INTER_LINEAR);
        cv::Mat clamped;
        cv::min(cv::max(sal_small, 0.5), 15.0, clamped);
        importance = importance.mul(clamped);
    }

    if (!dark_mask_.empty()) {
        cv::Mat dark_small;
        cv::resize(dark_mask_, dark_small, {W_opt_, H_opt_}, 0, 0, cv::INTER_LINEAR);
        cv::Mat weighted;
        dark_small.convertTo(weighted, CV_32FC1, 1.5 / 255.0, 1.0);
        importance = importance.mul(weighted);
    }

    importance.setTo(0, ~mask_small_);
    double total = cv::sum(importance)[0];
    if (total > 0) importance /= total;

    return importance;
}

void GradientOptimizer::initialize_shapes() {
    int N = config_.max_shapes;
    params_.init(N);

    cv::Mat importance = build_importance_map();

    int long_edge = std::max(W_opt_, H_opt_);
    float size_lo = long_edge * (config_.init_radius_lo / 100.0f);
    float size_hi = long_edge * (config_.init_radius_hi / 100.0f);
    std::uniform_real_distribution<float> dist_size(size_lo, size_hi);
    std::uniform_real_distribution<float> dist_angle(0.0f, 360.0f);

    // Weighted sampling from importance map
    cv::Mat flat_importance = importance.reshape(1, 1);
    std::vector<float> probs(flat_importance.cols);
    flat_importance.copyTo(cv::Mat(1, flat_importance.cols, CV_32FC1, probs.data()));

    double prob_sum = std::accumulate(probs.begin(), probs.end(), 0.0);
    if (prob_sum > 0) {
        std::discrete_distribution<int> dist_pos(probs.begin(), probs.end());
        for (int i = 0; i < N; ++i) {
            int idx = dist_pos(rng_);
            int y = idx / W_opt_, x = idx % W_opt_;
            params_.cx[i] = static_cast<float>(x);
            params_.cy[i] = static_cast<float>(y);
            params_.rx[i] = dist_size(rng_);
            params_.ry[i] = dist_size(rng_);
            params_.angle[i] = dist_angle(rng_);
            auto [r, g, b] = get_local_color(static_cast<float>(x), static_cast<float>(y));
            params_.color_r[i] = r;
            params_.color_g[i] = g;
            params_.color_b[i] = b;
            params_.color_a[i] = 255.0f;
        }
    }

    VIN_INFO("Initialized %d shapes", N);
}

std::tuple<int,int,int> GradientOptimizer::get_local_color(float x, float y) {
    int r = 5;
    int x0 = std::max(0, static_cast<int>(x) - r);
    int x1 = std::min(W_opt_, static_cast<int>(x) + r + 1);
    int y0 = std::max(0, static_cast<int>(y) - r);
    int y1 = std::min(H_opt_, static_cast<int>(y) + r + 1);

    cv::Mat local = target_small_(cv::Range(y0, y1), cv::Range(x0, x1));
    cv::Mat local_mask = mask_small_(cv::Range(y0, y1), cv::Range(x0, x1));

    cv::Scalar mean = cv::mean(local, local_mask);
    return {static_cast<int>(mean[0]), static_cast<int>(mean[1]), static_cast<int>(mean[2])};
}

std::tuple<int,int,int> GradientOptimizer::get_posterized_color(float x, float y) {
    if (!posterized_small_.empty() && !palette_.empty()) {
        int yi = std::clamp(static_cast<int>(y), 0, H_opt_ - 1);
        int xi = std::clamp(static_cast<int>(x), 0, W_opt_ - 1);
        int label = posterized_small_.at<uint8_t>(yi, xi);
        if (label < static_cast<int>(palette_.size())) {
            return palette_[label];
        }
    }
    return get_local_color(x, y);
}

// ================================================================
// Single iteration
// ================================================================

float GradientOptimizer::run_single_iter(
    OptParams& params,
    std::vector<float>& grads, int iter, int total_iters,
    float* d_grad_history, int write_idx,
    bool local_phase, const std::vector<int>& frozen_indices)
{
    int N = params.N;

    // Forward + backward on GPU (optionally writes gradient norms to history)
    float mse = renderer_.render_forward_backward(N, d_grad_history, write_idx);

    // Update frozen mask on GPU
    if (!frozen_indices.empty()) {
        std::vector<int> frozen_mask(N, 0);
        for (int i : frozen_indices) frozen_mask[i] = 1;
        renderer_.set_frozen_mask(frozen_mask.data(), N);
    }

    // Learning rate schedule (cosine)
    float progress = static_cast<float>(iter) / std::max(total_iters - 1, 1);
    float lr = config_.lr_end + (config_.lr_start - config_.lr_end) *
               0.5f * (1.0f + std::cos(static_cast<float>(M_PI) * progress));

    // Fused GPU Adam: gradient clip + Adam update + param clamp + STE quantize
    adam_step_++;
    renderer_.fused_adam_update(
        N, adam_step_, lr, 0.9f, 0.999f, 1e-8f,
        100.0f,   // max_grad_norm
        W_opt_, H_opt_);

    return mse;
}

// ================================================================
// Main optimization loop
// ================================================================

void GradientOptimizer::gradient_optimize() {
    int N = params_.N;
    int cycle_global = config_.reloc_cycle_global;
    int cycle_local = config_.reloc_cycle_local;
    int cycle_total = cycle_global + cycle_local;
    int num_cycles = config_.total_iters / cycle_total;
    int total_global_iters = num_cycles * cycle_global;

    // Allocate gradient history ring buffer on GPU
    int window = config_.reloc_stability_window;
    {
        cudaError_t e = cudaMalloc(&d_grad_history_, window * N * sizeof(float));
        if (e != cudaSuccess) {
            VIN_ERROR("cudaMalloc failed in gradient_optimize: %s", cudaGetErrorString(e));
            return;
        }
        cudaMemset(d_grad_history_, 0, window * N * sizeof(float));
    }
    history_write_idx_ = 0;
    history_count_ = 0;
    reloc_total_ = 0;
    reloc_converged_ = false;

    std::vector<float> grads(N * 9, 0.0f);

    // Upload initial parameters to GPU-resident buffers
    {
        renderer_.upload_params_gpu(params_.to_shape_params().data(), N);
    }

    // Initialize async visualization pipeline
    int viz_size = W_opt_ * H_opt_ * 3;
    {
        cudaError_t e = cudaMallocHost(&viz_pinned_, viz_size * sizeof(uint8_t));
        if (e != cudaSuccess) {
            VIN_WARN("cudaMallocHost failed for visualization: %s", cudaGetErrorString(e));
            viz_pinned_ = nullptr;
        }
    }
    cudaEventCreate(&viz_event_);
    viz_first_frame_ = true;

    // Allocate pinned memory and event for async param snapshots
    {
        cudaError_t e = cudaMallocHost(&snap_pinned_, N * 9 * sizeof(float));
        if (e != cudaSuccess) {
            VIN_WARN("cudaMallocHost failed for snapshots: %s", cudaGetErrorString(e));
            snap_pinned_ = nullptr;
        }
    }
    cudaEventCreate(&snap_event_);
    best_mse_ = 1e20f;
    has_best_ = false;
    snap_pending_ = false;

    VIN_INFO("Starting optimization: %d cycles (%d global + %d local), lr=%.3f->%.3f",
             num_cycles, cycle_global, cycle_local, config_.lr_start, config_.lr_end);

    int global_iter = 0;

    for (int cycle_idx = 0; cycle_idx < num_cycles; ++cycle_idx) {
        if (stop_requested_) break;

        // Consume previous cycle's async snapshot
        if (snap_pending_ && snap_pinned_ && N > 0) {
            cudaEventSynchronize(snap_event_);
            if (cycle_mse_ < best_mse_) {
                best_mse_ = cycle_mse_;
                best_params_flat_.resize(N * 9);
                memcpy(&best_params_flat_[0 * N], snap_pinned_ + 0 * N, N * sizeof(float));
                memcpy(&best_params_flat_[1 * N], snap_pinned_ + 1 * N, N * sizeof(float));
                memcpy(&best_params_flat_[2 * N], snap_pinned_ + 2 * N, N * sizeof(float));
                memcpy(&best_params_flat_[3 * N], snap_pinned_ + 3 * N, N * sizeof(float));
                memcpy(&best_params_flat_[4 * N], snap_pinned_ + 4 * N, N * sizeof(float));
                for (int i = 0; i < N; ++i) {
                    best_params_flat_[5 * N + i] = snap_pinned_[5 * N + i * 3 + 0];
                    best_params_flat_[6 * N + i] = snap_pinned_[5 * N + i * 3 + 1];
                    best_params_flat_[7 * N + i] = snap_pinned_[5 * N + i * 3 + 2];
                }
                memcpy(&best_params_flat_[8 * N], snap_pinned_ + 8 * N, N * sizeof(float));
                has_best_ = true;
            }
            snap_pending_ = false;
        }

        // Phase A: Global optimization
        for (int gi = 0; gi < cycle_global; ++gi) {
            if (stop_requested_) break;

            // Synchronize previous async canvas copy and update canvas_rgb_
            if (!viz_first_frame_ && viz_pinned_) {
                cudaEventSynchronize(viz_event_);
                cv::Mat rgb_mat(H_opt_, W_opt_, CV_8UC3, viz_pinned_);
                int write_idx = viz_write_idx_.load();
                {
                    std::lock_guard<std::mutex> lock(viz_mutex_);
                    canvas_rgb_[write_idx] = rgb_mat;
                }
                // Swap write buffer for next frame
                viz_write_idx_.store(1 - write_idx);
            }

            float mse = run_single_iter(params_, grads,
                                        global_iter, total_global_iters,
                                        d_grad_history_, history_write_idx_);
            cycle_mse_ = mse;

            // Async copy canvas to pinned memory for visualization
            if (viz_pinned_) {
                renderer_.async_copy_canvas(viz_pinned_, viz_event_);
                viz_first_frame_ = false;
            }

            // Update ring buffer write index
            history_write_idx_ = (history_write_idx_ + 1) % window;
            history_count_ = std::min(history_count_ + 1, window);

            if (gi % 50 == 0) {
                VIN_INFO("  cycle %d/%d, iter %d, mse=%.1f", cycle_idx + 1, num_cycles,
                         global_iter, mse);
            }

            if (progress_cb_) {
                progress_cb_(static_cast<float>(global_iter) / total_global_iters,
                            mse, global_iter);
            }

            global_iter++;
        }

        // Queue async param snapshot at end of Phase A
        if (snap_pinned_ && N > 0) {
            renderer_.async_snapshot_params(snap_pinned_, snap_event_, N);
            snap_pending_ = true;
        }

        if (stop_requested_) break;

        // Phase B: Relocation scan
        bool is_last_cycle = (cycle_idx == num_cycles - 1);
        std::vector<int> cand_indices;
        int n_useful = 0, n_useless = 0, n_unstable = 0;

        if (!reloc_converged_ && !is_last_cycle) {
            // Download grads from GPU for relocation analysis
            renderer_.download_grads_packed(grads.data(), N);
            // Compute and download per-shape visibility (blend weight sum)
            renderer_.compute_visibility(N);
            std::vector<float> h_visibility(N);
            renderer_.download_visibility(h_visibility.data(), N);
            // Sync GPU params to CPU for relocation
            {
                auto sp = params_.to_shape_params();
                renderer_.download_params_gpu(sp.data(), N);
                // Copy back to CPU params
                for (int i = 0; i < N; ++i) params_.set_shape(i, sp[i]);
            }
            find_relocation_candidates(grads, h_visibility, cand_indices, n_useful, n_useless, n_unstable);
        }

        // Phase C: Local optimization
        if (!cand_indices.empty()) {
            auto new_indices = relocate_shapes(params_, cand_indices);
            // Upload relocated params to GPU
            {
                renderer_.upload_params_gpu(params_.to_shape_params().data(), N);
            }
            // Reset ALL Adam state (z-order compact invalidates old momentum)
            renderer_.reset_adam_state_all_gpu(N);
            adam_step_ = 0;
            history_write_idx_ = 0;
            history_count_ = 0;
            run_local_phase(params_, new_indices, cycle_local);
        } else if (!is_last_cycle && n_useless == 0 && n_unstable == 0) {
            reloc_converged_ = true;
        }

        VIN_INFO("Cycle %d/%d: reloc=%d, useful=%d, useless=%d, unstable=%d, converged=%d",
                 cycle_idx + 1, num_cycles, static_cast<int>(cand_indices.size()),
                 n_useful, n_useless, n_unstable, reloc_converged_);
    }

    // Consume last cycle's snapshot
    if (snap_pending_ && snap_pinned_ && N > 0) {
        cudaEventSynchronize(snap_event_);
        if (cycle_mse_ < best_mse_) {
            best_mse_ = cycle_mse_;
            best_params_flat_.resize(N * 9);
            memcpy(&best_params_flat_[0 * N], snap_pinned_ + 0 * N, N * sizeof(float));
            memcpy(&best_params_flat_[1 * N], snap_pinned_ + 1 * N, N * sizeof(float));
            memcpy(&best_params_flat_[2 * N], snap_pinned_ + 2 * N, N * sizeof(float));
            memcpy(&best_params_flat_[3 * N], snap_pinned_ + 3 * N, N * sizeof(float));
            memcpy(&best_params_flat_[4 * N], snap_pinned_ + 4 * N, N * sizeof(float));
            for (int i = 0; i < N; ++i) {
                best_params_flat_[5 * N + i] = snap_pinned_[5 * N + i * 3 + 0];
                best_params_flat_[6 * N + i] = snap_pinned_[5 * N + i * 3 + 1];
                best_params_flat_[7 * N + i] = snap_pinned_[5 * N + i * 3 + 2];
            }
            memcpy(&best_params_flat_[8 * N], snap_pinned_ + 8 * N, N * sizeof(float));
            has_best_ = true;
        }
        snap_pending_ = false;
    }

    // Restore best params to CPU + GPU if a better snapshot was found
    if (has_best_ && N > 0) {
        for (int i = 0; i < N; ++i) {
            params_.cx[i] = best_params_flat_[0 * N + i];
            params_.cy[i] = best_params_flat_[1 * N + i];
            params_.rx[i] = best_params_flat_[2 * N + i];
            params_.ry[i] = best_params_flat_[3 * N + i];
            params_.angle[i] = best_params_flat_[4 * N + i];
            params_.color_r[i] = best_params_flat_[5 * N + i];
            params_.color_g[i] = best_params_flat_[6 * N + i];
            params_.color_b[i] = best_params_flat_[7 * N + i];
            params_.color_a[i] = best_params_flat_[8 * N + i] * 255.0f;
        }
        renderer_.upload_params_gpu(params_.to_shape_params().data(), N);
        VIN_INFO("Restored best cycle params: MSE=%.1f", best_mse_);
    }

    // Free snapshot resources
    if (snap_pinned_) {
        cudaFreeHost(snap_pinned_);
        snap_pinned_ = nullptr;
    }
    if (snap_event_) {
        cudaEventDestroy(snap_event_);
        snap_event_ = nullptr;
    }

    // Cleanup — sync render stream only (avoid OpenGL-blocking device sync)
    renderer_.sync_stream();
    if (d_grad_history_) {
        cudaFree(d_grad_history_);
        d_grad_history_ = nullptr;
    }
    if (viz_pinned_) {
        cudaFreeHost(viz_pinned_);
        viz_pinned_ = nullptr;
    }
    if (viz_event_) {
        cudaEventDestroy(viz_event_);
        viz_event_ = nullptr;
    }
}

// ================================================================
// Cyclic Relocation
// ================================================================

void GradientOptimizer::find_relocation_candidates(
    const std::vector<float>& grads,
    const std::vector<float>& visibility,
    std::vector<int>& candidates,
    int& n_useful, int& n_useless, int& n_unstable)
{
    int N = params_.N;
    int window = config_.reloc_stability_window;

    if (history_count_ < window) {
        n_unstable = N;
        return;
    }

    // Download gradient history from GPU
    std::vector<float> h_history(window * N);
    cudaMemcpy(h_history.data(), d_grad_history_, window * N * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Reconstruct chronological order from ring buffer
    std::vector<float> chronological(window * N);
    int idx = history_write_idx_;
    if (idx > 0) {
        // [idx..end] + [0..idx)
        memcpy(chronological.data(), h_history.data() + idx * N,
               (window - idx) * N * sizeof(float));
        memcpy(chronological.data() + (window - idx) * N, h_history.data(),
               idx * N * sizeof(float));
    } else {
        chronological = h_history;
    }

    // Mean gradient norm per shape
    std::vector<float> mean_gn(N, 0.0f);
    for (int w = 0; w < window; ++w) {
        for (int i = 0; i < N; ++i) {
            mean_gn[i] += chronological[w * N + i];
        }
    }
    for (int i = 0; i < N; ++i) mean_gn[i] /= window;

    // Classify shapes: use blend weight sum (actual per-shape visibility)
    // Low visibility = too small OR buried under other shapes
    std::vector<std::pair<float, int>> cand_list;
    n_useful = n_useless = n_unstable = 0;

    for (int i = 0; i < N; ++i) {
        bool is_stable = mean_gn[i] < config_.reloc_stability_threshold;
        if (!is_stable) {
            n_unstable++;
            continue;
        }
        float vis = (i < static_cast<int>(visibility.size())) ? visibility[i] : 0.0f;
        bool is_useless = vis < 2.0f;
        if (is_useless) {
            n_useless++;
            cand_list.push_back({vis, i});
        } else {
            n_useful++;
        }
    }

    // Sort by visibility (lowest first)
    std::sort(cand_list.begin(), cand_list.end());
    candidates.clear();
    for (auto& [_, i] : cand_list) candidates.push_back(i);
}

std::vector<int> GradientOptimizer::relocate_shapes(OptParams& params,
                                                    const std::vector<int>& indices) {
    int n = static_cast<int>(indices.size());
    if (n == 0) return {};

    // Download pre-rendered d_output from GPU (avoids re-rendering)
    int total_pixels = H_opt_ * W_opt_;
    std::vector<float> rendered(total_pixels * 3);
    renderer_.download_output(rendered.data());

    // Compute error map on CPU
    cv::Mat error_map = cv::Mat::zeros(H_opt_, W_opt_, CV_32FC1);
    for (int y = 0; y < H_opt_; ++y) {
        for (int x = 0; x < W_opt_; ++x) {
            int idx = y * W_opt_ + x;
            if (!mask_small_.at<uint8_t>(y, x)) continue;
            auto& tc = target_small_.at<cv::Vec3b>(y, x);
            float dr = rendered[idx * 3 + 0] - tc[0];
            float dg = rendered[idx * 3 + 1] - tc[1];
            float db = rendered[idx * 3 + 2] - tc[2];
            error_map.at<float>(y, x) = dr * dr + dg * dg + db * db;
        }
    }

    // Top-K sampling
    cv::Mat flat_error = error_map.reshape(1, 1);
    int K = std::max(n * 5, std::min(50, static_cast<int>(flat_error.cols * 0.05)));
    cv::Mat sorted_indices;
    cv::sortIdx(flat_error, sorted_indices, cv::SORT_DESCENDING);

    std::uniform_int_distribution<int> dist_k(0, K - 1);
    int long_edge = std::max(W_opt_, H_opt_);
    float size_lo = long_edge * (config_.reloc_radius_lo / 100.0f);
    float size_hi = long_edge * (config_.reloc_radius_hi / 100.0f);
    std::uniform_real_distribution<float> dist_size(size_lo, size_hi);
    std::uniform_real_distribution<float> dist_angle(0.0f, 360.0f);

    // Compute new params for relocated shapes
    std::vector<float> nc_x(n), nc_y(n), nr_x(n), nr_y(n), na_ng(n);
    std::vector<float> nc_r(n), nc_g(n), nc_b(n), nc_a(n);
    for (int j = 0; j < n; ++j) {
        int k_idx = dist_k(rng_);
        int flat_idx = sorted_indices.at<int>(0, std::min(k_idx, flat_error.cols - 1));
        int new_y = flat_idx / W_opt_, new_x = flat_idx % W_opt_;

        nc_x[j] = static_cast<float>(new_x);
        nc_y[j] = static_cast<float>(new_y);
        nr_x[j] = dist_size(rng_);
        nr_y[j] = dist_size(rng_);
        na_ng[j] = dist_angle(rng_);
        nc_a[j] = 255.0f;

        auto [r, g, b] = get_posterized_color(new_x, new_y);
        nc_r[j] = r; nc_g[j] = g; nc_b[j] = b;
    }

    // Compact params_: remove relocated shapes, shift others forward
    int N = params.N;
    std::vector<bool> is_reloc(N, false);
    for (int i : indices) is_reloc[i] = true;

    auto compact = [&](auto& vec) {
        int w = 0;
        for (int i = 0; i < N; ++i) {
            if (!is_reloc[i]) {
                if (w != i) vec[w] = vec[i];
                ++w;
            }
        }
    };

    compact(params.cx);  compact(params.cy);
    compact(params.rx);  compact(params.ry);
    compact(params.angle);
    compact(params.color_r); compact(params.color_g);
    compact(params.color_b); compact(params.color_a);

    // Append relocated shapes at the end of params_ (new top layer)
    int base = N - n;
    std::vector<int> new_indices;
    for (int j = 0; j < n; ++j) {
        int dst = base + j;
        new_indices.push_back(dst);
        params.cx[dst] = nc_x[j]; params.cy[dst] = nc_y[j];
        params.rx[dst] = nr_x[j]; params.ry[dst] = nr_y[j];
        params.angle[dst] = na_ng[j];
        params.color_r[dst] = nc_r[j]; params.color_g[dst] = nc_g[j];
        params.color_b[dst] = nc_b[j]; params.color_a[dst] = nc_a[j];
    }

    reloc_total_ += n;
    VIN_INFO("Relocated %d shapes to top layer (total: %d)", n, reloc_total_);
    return new_indices;
}

float GradientOptimizer::run_local_phase(OptParams& params,
                                          const std::vector<int>& reloc_indices,
                                          int num_iters) {
    int N = params.N;

    // Build frozen mask (frozen shapes won't be updated by GPU Adam)
    std::vector<bool> is_frozen(N, true);
    for (int i : reloc_indices) is_frozen[i] = false;
    std::vector<int> frozen_indices;
    for (int i = 0; i < N; ++i) {
        if (is_frozen[i]) frozen_indices.push_back(i);
    }

    // Set frozen mask on GPU
    std::vector<int> frozen_mask(N, 0);
    for (int i : frozen_indices) frozen_mask[i] = 1;
    renderer_.set_frozen_mask(frozen_mask.data(), N);

    std::vector<float> grads(N * 9, 0.0f);
    float final_mse = 0.0f;

    for (int li = 0; li < num_iters; ++li) {
        if (stop_requested_) break;

        final_mse = run_single_iter(params, grads, li, num_iters,
                                     nullptr, 0, true, frozen_indices);
    }

    // Clear frozen mask after local phase
    std::vector<int> clear_mask(N, 0);
    renderer_.set_frozen_mask(clear_mask.data(), N);

    return final_mse;
}

// ================================================================
// Parameter conversion
// ================================================================

std::vector<Shape> GradientOptimizer::params_to_shapes() {
    // Download params from GPU before converting
    auto sp = params_.to_shape_params();
    renderer_.download_params_gpu(sp.data(), params_.N);
    for (int i = 0; i < params_.N; ++i) params_.set_shape(i, sp[i]);

    std::vector<Shape> shapes;
    float inv_scale = 1.0f / scale_;

    for (int i = 0; i < params_.N; ++i) {
        int a_val = static_cast<int>(std::round(std::clamp(params_.color_a[i], 0.0f, 255.0f)));
        float opacity = a_val / 255.0f;
        int r = static_cast<int>(std::round(std::clamp(params_.color_r[i], 0.0f, 255.0f)));
        int g = static_cast<int>(std::round(std::clamp(params_.color_g[i], 0.0f, 255.0f)));
        int b = static_cast<int>(std::round(std::clamp(params_.color_b[i], 0.0f, 255.0f)));

        shapes.push_back(Shape::make_soft_ellipse_228(
            params_.cx[i] * inv_scale, params_.cy[i] * inv_scale,
            params_.rx[i] * inv_scale, params_.ry[i] * inv_scale,
            fmod(params_.angle[i], 360.0f),
            Color(r, g, b, 255), opacity));
    }

    return shapes;
}

cv::Mat GradientOptimizer::get_canvas() const {
    std::lock_guard<std::mutex> lock(viz_mutex_);
    int write_idx = viz_write_idx_.load();
    int read_idx = 1 - write_idx;
    return canvas_rgb_[read_idx].clone();
}

} // namespace vinylizer
