#pragma once

#include <cuda_runtime.h>
#include <vector>

namespace vinylizer {

// Shape parameters packed for GPU transfer
// Each shape: cx, cy, rx, ry, angle, color[3], opacity = 8 floats
struct ShapeParams {
    float cx, cy, rx, ry, angle;
    float color_r, color_g, color_b;
    float opacity;
};

struct RenderScratch {
    // Pixel coordinates [total_pixels]
    float* d_px = nullptr;
    float* d_py = nullptr;

    // Shape parameters [N]
    float* d_cx = nullptr;
    float* d_cy = nullptr;
    float* d_rx = nullptr;
    float* d_ry = nullptr;
    float* d_angle = nullptr;
    float* d_colors = nullptr;    // [N*3]
    float* d_opacity = nullptr;
    float* d_sina = nullptr;
    float* d_cosa = nullptr;

    // Background color [3]
    float* d_bg_color = nullptr;

    // Render output [total_pixels * 3]
    float* d_output = nullptr;

    // Gradient output [N] each + [N*3] for colors
    float* d_grad_cx = nullptr;
    float* d_grad_cy = nullptr;
    float* d_grad_rx = nullptr;
    float* d_grad_ry = nullptr;
    float* d_grad_angle = nullptr;
    float* d_grad_colors = nullptr;  // [N*3]
    float* d_grad_opacity = nullptr;

    // Packed gradient buffer [N*8] for single-download
    float* d_grads_packed = nullptr;

    // GPU-resident Adam state
    float* d_m_cx = nullptr; float* d_m_cy = nullptr;
    float* d_m_rx = nullptr; float* d_m_ry = nullptr;
    float* d_m_angle = nullptr;
    float* d_m_cr = nullptr; float* d_m_cg = nullptr; float* d_m_cb = nullptr; float* d_m_ca = nullptr;
    float* d_v_cx = nullptr; float* d_v_cy = nullptr;
    float* d_v_rx = nullptr; float* d_v_ry = nullptr;
    float* d_v_angle = nullptr;
    float* d_v_cr = nullptr; float* d_v_cg = nullptr; float* d_v_cb = nullptr; float* d_v_ca = nullptr;

    // STE quantized output (for rendering, separate from params)
    float* d_ste_cx = nullptr; float* d_ste_cy = nullptr;
    float* d_ste_rx = nullptr; float* d_ste_ry = nullptr;
    float* d_ste_angle = nullptr;
    float* d_ste_colors = nullptr;  // [N*3]
    float* d_ste_opacity = nullptr;

    // Frozen mask [N] (1=frozen, 0=active)
    int* d_frozen_mask = nullptr;

    // Gradient clip norm accumulator
    float* d_grad_norm_sq = nullptr;

    // Gradient from loss [total_pixels * 3]
    float* d_grad_output = nullptr;

    // Scratch for tile forward [N * tile_pixels]
    float* d_alpha_scratch = nullptr;
    float* d_r_norm_scratch = nullptr;

    // Active indices buffer [N]
    int* d_active_indices = nullptr;

    // CPU-side pinned memory for shape params (for AABB filtering)
    float* h_cx = nullptr;
    float* h_cy = nullptr;
    float* h_rx = nullptr;
    float* h_ry = nullptr;
    float* h_px = nullptr;
    float* h_py = nullptr;

    // Target image [total_pixels * 3]
    float* d_target = nullptr;

    // Mask [total_pixels] (1.0f for opaque, 0.0f for transparent)
    float* d_mask = nullptr;

    // Target alpha [total_pixels] (255.0f for opaque, 0.0f for transparent)
    float* d_target_alpha = nullptr;

    // Rendered alpha [total_pixels] (output from forward pass, 0-255)
    float* d_rendered_alpha = nullptr;

    // Alpha gradient [total_pixels] (from MSE loss)
    float* d_grad_alpha = nullptr;

    // Per-shape visibility [N] (blend weight sum, for relocation)
    float* d_visibility = nullptr;

    // MSE sum accumulator (pre-allocated, avoids per-iter malloc)
    float* d_mse_sum = nullptr;

    // Alpha LUT texture for alpha_228 approximation
    cudaTextureObject_t alpha_tex = 0;
    cudaArray* alpha_array = nullptr;

    // Canvas uint8 buffer for async visualization [total_pixels * 3]
    uint8_t* d_canvas_u8 = nullptr;

    int total_pixels = 0;
    int max_shapes = 0;
    int tile_pixels = 0;
    int num_tiles = 0;
    cudaStream_t stream = nullptr;

    // Precomputed tile AABBs (constant after init)
    std::vector<float> tile_tx1, tile_tx2, tile_ty1, tile_ty2;

    // GPU buffer for active indices per tile (pre-allocated)
    int* d_active_indices_per_tile = nullptr;  // [num_tiles * max_shapes]
    int* d_active_count_per_tile = nullptr;    // [num_tiles]
};

class DiffRenderer {
public:
    DiffRenderer() = default;
    ~DiffRenderer();

    // Non-copyable
    DiffRenderer(const DiffRenderer&) = delete;
    DiffRenderer& operator=(const DiffRenderer&) = delete;

    void init(int canvas_w, int canvas_h, int max_shapes);
    void set_target(const float* h_target, const float* h_mask, const float* h_target_alpha);
    void set_bg_color(float r, float g, float b);

    // Compute per-shape visibility (blend weight sum on GPU)
    void compute_visibility(int N);
    void download_visibility(float* h_visibility, int N);

    // Forward render: returns rendered image in h_output [H*W*3]
    void render_forward(const ShapeParams* shapes, int N,
                        float* h_output);

    // Download current d_output (already rendered by render_forward_backward)
    // Avoids re-rendering for relocation error map computation
    void download_output(float* h_output);

    // Forward + backward: compute MSE and gradients (GPU-resident params)
    // d_grad_history: optional ring buffer for per-shape gradient norms (relocation)
    // write_idx: write slot in d_grad_history ring buffer
    float render_forward_backward(int N,
        float* d_grad_history = nullptr, int write_idx = 0);

    // Async copy current d_output to host pinned memory
    void async_copy_canvas(uint8_t* h_pinned_canvas, cudaEvent_t& event);

    // Async snapshot of all GPU-resident params to pinned memory (SoA layout)
    void async_snapshot_params(float* h_pinned, cudaEvent_t event, int N);

    // Upload initial shape parameters to GPU-resident buffers
    void upload_params_gpu(const ShapeParams* shapes, int N);

    // Fused GPU Adam: gradient clip + Adam update + param clamp + STE quantize
    void fused_adam_update(
        int N, int adam_step,
        float lr, float beta1, float beta2, float eps,
        float max_grad_norm,
        int W, int H);

    // Download GPU-resident params to CPU ShapeParams
    void download_params_gpu(ShapeParams* shapes, int N);

    // Update frozen mask on GPU
    void set_frozen_mask(const int* frozen, int N);

    // Download packed gradients from GPU to CPU
    void download_grads_packed(float* h_grads, int N);

    // Reset GPU Adam state for specified shape indices
    void reset_adam_state_gpu(const std::vector<int>& indices);

    // Zero all GPU Adam state (used after shape reordering)
    void reset_adam_state_all_gpu(int N);

    // Synchronize the renderer's CUDA stream (call before freeing resources)
    void sync_stream();

    int canvas_w() const { return canvas_w_; }
    int canvas_h() const { return canvas_h_; }

private:
    float compute_mse_and_grad(int N);

    int canvas_w_ = 0;
    int canvas_h_ = 0;
    int max_shapes_ = 0;
    RenderScratch scratch_;
    bool initialized_ = false;
    bool full_scratch_ = false;  // true = single tile covers whole image

    // CPU-side copies for AABB filtering
    std::vector<float> h_cx_, h_cy_, h_rx_, h_ry_;
    std::vector<float> h_px_, h_py_;

    // Pre-allocated upload buffers (avoid per-iter heap allocation)
    std::vector<float> h_angle_buf_, h_colors_buf_, h_opacity_buf_;
};

} // namespace vinylizer
