#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <string>

namespace vinylizer {

constexpr int TYPE_228 = 0;
constexpr int TYPE_HARD_ELLIPSE = 1;
constexpr int TYPE_RECT = 2;
constexpr int TYPE_TRIANGLE = 3;
constexpr int NUM_TYPES = 4;
constexpr int DARTS_MODE = -1;  // Sentinel: shape uses softmax blending (not locked to a type)

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
    float* d_colors = nullptr;    // [N*3] in logit space (sigmoid parameterization)
    float* d_opacity = nullptr;   // [N] in logit space
    float* d_sina = nullptr;
    float* d_cosa = nullptr;

    // Background color [3]
    float* d_bg_color = nullptr;

    // Render output [total_pixels * 3]
    float* d_output = nullptr;

    // Gradient output [N] each + [N*3] for colors
    // d_grad_unpacked: 合并的 SoA buffer [N*9] = cx,cy,rx,ry,angle,colors[3],opacity
    // d_grad_cx 等为偏移指针（指向 d_grad_unpacked 各段），不再独立分配
    float* d_grad_unpacked = nullptr;  // single cudaMalloc, single memset
    float* d_grad_cx = nullptr;        // = d_grad_unpacked + 0
    float* d_grad_cy = nullptr;        // = d_grad_unpacked + max_shapes
    float* d_grad_rx = nullptr;        // = d_grad_unpacked + max_shapes*2
    float* d_grad_ry = nullptr;        // = d_grad_unpacked + max_shapes*3
    float* d_grad_angle = nullptr;     // = d_grad_unpacked + max_shapes*4
    float* d_grad_colors = nullptr;    // = d_grad_unpacked + max_shapes*5  (3N floats)
    float* d_grad_opacity = nullptr;   // = d_grad_unpacked + max_shapes*8

    // Packed gradient buffer [N*13] for single-download (AoS layout for Adam)
    float* d_grads_packed = nullptr;

    // GPU-resident Adam state (合并为 2 个 buffer，偏移指针指向各段)
    // 布局: [13 * max_shapes]，每段 max_shapes 个 float
    // 段顺序: cx(0), cy(1), rx(2), ry(3), angle(4), cr(5), cg(6), cb(7), ca(8), l0(9), l1(10), l2(11), l3(12)
    float* d_adam_m_buffer = nullptr;  // single cudaMalloc, single memset
    float* d_adam_v_buffer = nullptr;
    // 以下为偏移指针（指向 d_adam_m_buffer / d_adam_v_buffer 各段），不再独立分配
    float* d_m_cx = nullptr; float* d_m_cy = nullptr;
    float* d_m_rx = nullptr; float* d_m_ry = nullptr;
    float* d_m_angle = nullptr;
    float* d_m_cr = nullptr; float* d_m_cg = nullptr; float* d_m_cb = nullptr; float* d_m_ca = nullptr;
    float* d_v_cx = nullptr; float* d_v_cy = nullptr;
    float* d_v_rx = nullptr; float* d_v_ry = nullptr;
    float* d_v_angle = nullptr;
    float* d_v_cr = nullptr; float* d_v_cg = nullptr; float* d_v_cb = nullptr; float* d_v_ca = nullptr;

    float* d_type_logits = nullptr;
    float* d_ste_type_logits = nullptr;
    // 以下 logits Adam state 指针已合并为 d_m_logits / d_v_logits
    float* d_m_logits = nullptr;
    float* d_v_logits = nullptr;
    float* d_softmax_w = nullptr;
    float* d_grad_logits = nullptr;

    int* d_shape_types = nullptr;
    int render_mode = 0;

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

    // === Gradient Loss（残差拉普拉斯）===
    // L_total = L_pixel + λ·L_grad，L_grad = Σ_p ||∇R(p)-∇T(p)||²。
    // 由 DᵀD=-Δ 得 ∂L_total/∂R = 2(R-T) - 2λ·Δ(R-T)，实现上只需把 backward 的
    // per-pixel 像素梯度 g(p) 替换为 g(p) - λ·Δg(p)（Δ 为 3×3 拉普拉斯，RGB 独立）。
    // 目标图保持锐利原图（与 baseline 一致），仅改 backward 梯度，不改 forward 与评估。
    // d_grad_lap_tmp: laplacian kernel 的临时输出缓冲 [total_pixels * 3]（init 时分配）。
    float* d_grad_lap_tmp = nullptr;
    float grad_lambda = 0.0f;              // --grad-lambda: 梯度损失权重（0=禁用=baseline）

    // === Tile-based rendering acceleration ===
    int* d_tile_shape_lists = nullptr;     // [num_tiles * max_shapes_per_tile]
    int* d_tile_shape_counts = nullptr;    // [num_tiles]
    int* d_tile_overflow = nullptr;        // [1] overflow flag
    int num_tiles_x = 0;
    int num_tiles_y = 0;
    int num_tiles = 0;
    int max_shapes_per_tile = 2048;
    int tile_size = 16;

    // Compact batch scratch (replaces full-size per-shape alpha cache)
    __half* d_batch_scratch = nullptr;     // [batch_max_shapes * batch_pixels] fp16
    size_t batch_scratch_capacity = 0;

    // Template sampling buffers (blur-template + affine sampling approach)
    float* d_templates = nullptr;          // [num_types * TEMPLATE_SIZE^2] pre-rendered shape templates
    float* d_blurred_templates = nullptr;  // [num_types * TEMPLATE_SIZE^2] blurred templates (updated per iter)
    float* d_template_tmp = nullptr;       // [num_types * TEMPLATE_SIZE^2] temp buffer for separable blur
    float* d_blur_kernel = nullptr;        // pre-allocated Gaussian kernel buffer for template blur

    // All-active indices for template fused path [max_shapes] (0,1,2,...,N-1, uploaded once)
    int* d_all_active_indices = nullptr;

    // Target image [total_pixels * 3]
    float* d_target = nullptr;

    // 双模糊退火：目标图模糊缓冲 [total_pixels * 3]（仅 target_blur_enabled_ 时分配）
    float* d_target_blurred = nullptr;
    float* d_target_blur_tmp = nullptr;

    // Mask [total_pixels] (1.0f for opaque, 0.0f for transparent)
    float* d_mask = nullptr;

    // Target alpha [total_pixels] (255.0f for opaque, 0.0f for transparent)
    float* d_target_alpha = nullptr;

    // Rendered alpha [total_pixels] (output from forward pass, 0-255)
    float* d_rendered_alpha = nullptr;

    // Alpha gradient [total_pixels] (from MSE loss)
    float* d_grad_alpha = nullptr;

    // === Soft depth (learnable render order) ===
    // d_z: [N] learnable depth scalar per shape (z 大 = 底层, z 小 = 顶层)
    // d_grad_z: [N] gradient buffer, atomic-accumulated in backward kernel
    // d_m_z / d_v_z: [N] Adam moments (independent from d_adam_m_buffer layout)
    // d_ste_z: [N] STE copy passed to render kernels (与 d_z 同值，预留接口)
    float* d_z = nullptr;
    float* d_grad_z = nullptr;
    float* d_m_z = nullptr;
    float* d_v_z = nullptr;
    float* d_ste_z = nullptr;
    float current_tau_z = 0.0f;   // 当前 soft-depth 温度（0=禁用）
    int soft_depth_max_m = 64;    // tile 内参与 pairwise 软遮挡的最大形状数
    bool soft_depth_enabled = false;

    // Per-shape visibility [N] (blend weight sum, for relocation)
    float* d_visibility = nullptr;

    // Per-shape theoretical max visibility [N] (Σ(ae) without T attenuation, for utilization)
    float* d_visibility_full = nullptr;

    // Per-shape per-quadrant RGB gradient [N*4*3] for split analysis
    // Layout: shape i, quadrant q (0=tl,1=tr,2=bl,3=br), channel c (0=r,1=g,2=b)
    // Index: ((i*4 + q)*3 + c)
    float* d_grad_rgb_quadrant = nullptr;

    // MSE sum accumulator (pre-allocated, avoids per-iter malloc)
    float* d_mse_sum = nullptr;

    // 双向渲染反向 buffer（仅在 bidir_lambda_ > 0 时分配）
    float* d_output_rev = nullptr;           // [total_pixels * 3] 反向渲染输出
    float* d_rendered_alpha_rev = nullptr;   // [total_pixels] 反向渲染 alpha
    float* d_grad_output_rev = nullptr;       // [total_pixels * 3] 反向渲染像素梯度
    float* d_grad_alpha_rev = nullptr;        // [total_pixels] 反向 alpha 梯度
    float* d_mse_sum_rev = nullptr;           // [1] 反向 MSE 累加器

    // Alpha LUT texture for alpha_228 approximation
    cudaTextureObject_t alpha_tex = 0;
    cudaArray* alpha_array = nullptr;

    // Canvas uint8 buffer for async visualization [total_pixels * 3]
    uint8_t* d_canvas_u8 = nullptr;

    // GPU-resident per-pixel error map scratch [total_pixels] (for relocate & EM-YMP)
    float* d_error_map_scratch = nullptr;

    // EM-YMP: 误差归约与均值标量（device-resident，避免每步 D2H sync）
    float* d_error_sum = nullptr;   // [1] sum(error_map) 归约结果
    float* d_mean_err = nullptr;    // [1] mean_err = error_sum / valid_pixel_count

    // Per-pixel coverage count [total_pixels] (diagnosis only: 每个像素被多少个形状覆盖)
    int* d_coverage_count = nullptr;

    // === Stacking Penalty Loss (SP-Loss) ===
    // L_sp = (λ/N_pixels)·Σ_p max(0, W(p)-K_allowed(p))², W=S-C, S=Σae, C=1-Π(1-ae)
    // S 在 forward kernel 累加（d_sp_S），C=rendered_alpha/255；K_allowed 在 set_target 预计算。
    // backward 注入 ∂L_sp/∂ae_i = (2λ/N_pixels)·max(0,W-K)·(1-Π_{j≠i}(1-ae_j)) 到 dlae。
    float* d_sp_S = nullptr;              // [total_pixels] S(p)=Σae_i（forward 累加输出）
    float* d_sp_G = nullptr;        // [total_pixels] SP-GEO-Loss: G(p)=Σar_i（几何覆盖层数）
    float* d_k_allowed = nullptr;         // [total_pixels] K_allowed(p)（预计算，复杂度自适应）
    float sp_lambda = 0.0f;               // SP-Loss 权重 (0=禁用)
    float sp_k_base = 1.5f;              // 最小允许堆叠量（防留洞）
    float sp_beta = 3.0f;                // 边缘区允许额外层数系数
    float sp_mean_var = 1e-6f;           // 全局均值方差（set_target 预计算，K_allowed 归一化用）

    // Error potential field gradient [total_pixels] each (normalized, uploaded per refresh)
    float* d_pf_grad_x = nullptr;
    float* d_pf_grad_y = nullptr;

    // FH-PF: per-shape vis gate coefficient [N] (1=strong force for low-vis, 0=no force).
    // Recomputed every fh_pf_update_interval iters from d_visibility.
    float* d_fh_vis_gate = nullptr;

    // === SP-GEO (Stacking Penalty Geometric) ===
    // 拥挤度势场 d(p)=max(0, W(p)-K_allowed(p))，W=S-C，复用 d_sp_S + rendered_alpha + d_k_allowed。
    // 势场 d 与其归一化 Sobel 梯度在 host 算后上传；kernel 采样 d(center) 作 gate、∇d(center) 作方向。
    // 位置 pos -= force·gate·∇d，半径 rx/ry -= shrink·d(center)（直接改 d_*/d_ste_*，绕过 Adam）。
    float* d_sp_geo_field = nullptr;        // [total_pixels] d(p)（gate 采样用）
    float* d_sp_geo_grad_x = nullptr;       // [total_pixels] 归一化拥挤度梯度 x
    float* d_sp_geo_grad_y = nullptr;       // [total_pixels] 归一化拥挤度梯度 y
    bool sp_geo_active = false;             // SP-GEO 总开关（影响 forward 是否写 d_sp_S、set_target 是否算 K_allowed）
    bool sp_geo_loss_active = false;  // SP-GEO-Loss 模式开关（true=几何覆盖损失，禁用旧 ae-based SP-Loss）
    float sp_geo_d_ref = 1.0f;              // gate 归一化参考（正 d 的 90 分位）

    // === Auxiliary Visibility Gradient (Aux-Vis) ===
    // g_i = 1 - sigmoid((vis_i - thresh)/temp)：唤醒权重（低 vis→大，高 vis→≈0）。
    // 在 backward kernel 中注入 ∂L_aux/∂ae_i = -λ·g_i·T_i，通过现有 ae→参数链传回。
    // g_i 视为常数权重（stop-gradient）：预计算上传，kernel 只读取，不对 vis_i 求导。
    // d_aux_gate 初始为 0（memset），aux_vis_lambda=0 或 gate=0 时注入量为 0，天然 no-op。
    float* d_aux_gate = nullptr;          // [N] g_i 缓冲
    float aux_vis_lambda = 0.0f;          // 0 = 禁用
    float aux_vis_thresh = 10.0f;         // vis 门控中点（与 reloc/FH-PF 阈值一致）
    float aux_vis_temp = 10.0f;           // vis 门控温度（vis 像素数尺度，需放大）
    bool aux_vis_only = false;            // 自检用：清零 MSE 梯度，只留 aux-vis 注入（机制纯净验证）

    // Multi-scale pyramid buffers for perceptual loss
    float* d_pyr_rendered_1 = nullptr;    // [H/2 * W/2 * 3]
    float* d_pyr_rendered_2 = nullptr;    // [H/4 * W/4 * 3]
    float* d_pyr_target_1 = nullptr;      // [H/2 * W/2 * 3]
    float* d_pyr_target_2 = nullptr;      // [H/4 * W/4 * 3]
    float* d_pyr_grad_tmp = nullptr;      // temp for gradient upsampling chain

    int total_pixels = 0;
    int max_shapes = 0;
    int num_template_types = 4;    // 运行期类型数
    int num_builtins = 4;          // 内置图元数 (0=纯外部图章, 4=含内置)
    cudaStream_t stream = nullptr;

    // 独立 viz stream：D2H copy 不阻塞 compute stream（借鉴 3A 游戏渲染线程分离策略）
    cudaStream_t viz_stream = nullptr;
    cudaEvent_t compute_done_event = nullptr;  // float_to_uint8_kernel 完成后 record

    // Loss config (set via DiffRenderer::set_loss_config)
    float perceptual_weight = 0.0f;   // 0 = disabled
    float alpha_loss_weight = 0.0f;   // extra alpha loss weight (on top of existing)
    float alpha_sparsity_weight = 0.0f; // opacity = 0 or 1 regularization
    int pyramid_levels = 0;           // number of pyramid levels (0 = disabled)
    bool alpha_loss_enabled = false;  // --alpha-loss: gate alpha channel gradient in MSE
    float yield_lambda = 0.0f;        // --yield-loss: occlusion cost weight (0 = disabled)
    bool yield_em_modulate = true;    // --yield-em: true=EM-YMP(误差调制), false=原 YMP

    // === Frequency-Adaptation Loss (FA-Loss) ===
    // L_fa = λ · Σ_i (A_i/ΣA) · (σ²_i/σ²_global)
    //   A_i   = Σ_p ar_i(p)·mask(p)            （形状 i 的几何覆盖面积，不含 opacity）
    //   μ_i   = Σ_p ar_i(p)·mask(p)·I(p) / A_i （覆盖区域的目标加权均值颜色）
    //   σ²_i  = Σ_p ar_i(p)·mask(p)·||I(p)-μ_i||² / A_i （覆盖区域的目标加权方差）
    // 梯度只通过 ar 流回 rx/ry/cx/cy/angle（不流回 color/opacity）。
    // VW-FA 推导结论：dL_fa/d_ar_k(p) = (λ/v) · (||I(p)-μ_k||² - T/s) · mask(p) · T_k(p)
    //   其中 s=ΣA, T=ΣA·σ², v=σ²_global, T_k(p)=遮挡 k 的上层形状残余透过率
    //   量级修正：fa_scale=λ/v（去 s 归一化）；可见性修正：权重乘 T_k(p)
    float fa_lambda = 0.0f;             // --fa-lambda: FA-Loss 权重 (0 = disabled)
    float fa_sigma2_global = 0.0f;      // 预计算的全图目标方差（set_target 时一次性计算）
    float fa_vis_weight = 1.0f;         // --fa-vis-weight: 可见性权重系数（1=VW-FA，0=原 FA 无 VW）
    float* d_fa_stats = nullptr;         // [N*5]: sum_w, sum_wI_r, sum_wI_g, sum_wI_b, sum_wI2
    float* d_fa_mu = nullptr;           // [N*3]: 每形状 μ_i（backward 用）
    float* d_fa_globals = nullptr;      // [3]: [0]=fa_scale=λ/v, [1]=fa_mean_t=T/s, [2]=fa_vis_weight

    // Pre-allocated buffer for batch Adam reset indices
    int* d_reset_indices = nullptr;

    // === R23: Deterministic reduction buffers ===
    // 开启 deterministic_reduce 时分配，替代 atomicAdd 归约路径。
    // d_sq_err_buffer: MSE 逐像素平方误差缓冲 [total_pixels]（替代 atomicAdd(mse_sum)）
    // d_gg_buffer: 梯度裁剪逐元素 g*g 缓冲 [N*13]（替代 atomicAdd(norm_sq)）
    // d_scalar_partials: 标量归约（MSE/norm_sq）的 block partials，[max_blocks_scalar]
    // d_grad_partials: per-shape 梯度归约 partials，[max_blocks_grad * N * 9]
    // d_grad_logits_partials: per-shape logits 梯度 partials，[max_blocks_grad * N * num_types]
    // deterministic_reduce: 0=off（默认，严格等价 baseline），1=开启两遍 kernel 确定性归约。
    float* d_sq_err_buffer = nullptr;            // [total_pixels] MSE 确定性归约用
    float* d_gg_buffer = nullptr;                 // [N*13] norm_sq 确定性归约用
    float* d_scalar_partials = nullptr;          // [max_blocks_scalar]
    float* d_grad_partials = nullptr;            // [max_blocks_grad * N * 9]
    float* d_grad_logits_partials = nullptr;     // [max_blocks_grad * N * num_template_types]
    int max_blocks_scalar = 0;
    int max_blocks_grad = 0;
    int deterministic_reduce = 0;                // 0=off, 1=on
};

struct TileBatchDesc {
    int tile_start;
    int tile_count;
    int total_pixels;
    int max_shapes;
};

class DiffRenderer {
public:
    DiffRenderer() = default;
    ~DiffRenderer();

    // Non-copyable
    DiffRenderer(const DiffRenderer&) = delete;
    DiffRenderer& operator=(const DiffRenderer&) = delete;

    // R23: 确定性归约开关。必须在 init() 之前调用（影响缓冲区分配）。
    // en=true 时 init() 分配 partials 缓冲，render/adam 走确定性两遍 kernel 路径。
    void set_deterministic_reduce(bool en) { deterministic_reduce_ = en ? 1 : 0; }
    bool deterministic_reduce_enabled() const { return deterministic_reduce_ != 0; }

    void init(int canvas_w, int canvas_h, int max_shapes, int num_template_types = 4, int num_builtins = 4);
    void load_external_stamps(const std::vector<std::string>& paths);
    int num_template_types() const { return scratch_.num_template_types; }
    void set_target(const float* h_target, const float* h_mask, const float* h_target_alpha);
    void set_bg_color(float r, float g, float b);
    void set_blur_sigma(float sigma);
    // 双模糊退火：开启后训练损失对「按当前 sigma 模糊的目标图」计算 MSE。
    // 必须在 init() 之后、set_target() 之前调用。原始 d_target 保持不变，
    // 诊断/最终 PSNR 仍用原始目标评估。
    void set_target_blur_enabled(bool enabled);
    bool target_blur_enabled() const { return target_blur_enabled_; }
    void set_loss_config(float perceptual_weight, int pyramid_levels,
                         float alpha_loss_weight, float alpha_sparsity_weight);
    void set_alpha_loss_enabled(bool enabled) { scratch_.alpha_loss_enabled = enabled; }

    // Yield Gradient / Occlusion Cost：>0 启用让位梯度损失项。
    // backward kernel 会维护 vis_below_acc 并把 ∂L_yield/∂ae 累加到 dlae，
    // 让顶层形状的梯度携带"你遮挡了多少下层形状"的信息。
    void set_yield_lambda(float lambda) { scratch_.yield_lambda = lambda; }

    // EM-YMP 误差调制开关：true 时 vis_below_acc 累加量乘以 err(p)/mean_err，
    // 上层只为高误差区的下层让位；false 退化为原 YMP（err_norm=1）。
    void set_yield_em_modulate(bool en) { scratch_.yield_em_modulate = en; }

    // EM-YMP: 重算逐像素误差图 + 全局平均误差（写入 d_error_map_scratch / d_mean_err）。
    // 在 backward 前调用，供 tile_backward_kernel 调制 vis_below_acc。
    void compute_error_map_and_mean();

    // === Frequency-Adaptation Loss (FA-Loss) API ===
    // 启用 FA-Loss：lambda > 0 时在 forward 后、backward 前调用 compute_fa_loss_stats，
    // 计算每形状 A_i/μ_i/σ²_i 与全局 scale/mean_t，供 tile_backward_kernel 注入 fa 梯度。
    // 梯度只流回 rx/ry/cx/cy/angle（通过 ar 链），不影响 color/opacity。
    void set_fa_lambda(float lambda) { scratch_.fa_lambda = lambda; }
    void set_fa_vis_weight(float w) { scratch_.fa_vis_weight = w; }
    void compute_fa_loss_stats(int N);

    // 双向渲染 λ 配置：>0 启用反向渲染（N-1→0 compositing 顺序），
    // 最终梯度 = 正向梯度 + λ × 反向梯度。0=禁用（默认）。
    void set_bidir_lambda(float lambda) { bidir_lambda_ = lambda; }

    // === Gradient Loss（残差拉普拉斯）===
    // λ>0 时 compute_mse_and_grad 在 mse_grad_kernel 之后把 d_grad_output 替换为
    // g(p) - λ·Δg(p)。λ=0（默认）完全等价 baseline。
    void set_grad_lambda(float lambda) { scratch_.grad_lambda = lambda; }

    // === Soft depth (learnable render order) API ===
    // 启用/禁用 soft depth。启用后 render_forward_backward 会把 d_z / tau_z 传给 kernel。
    void set_soft_depth_enabled(bool enabled) { scratch_.soft_depth_enabled = enabled; }
    bool soft_depth_enabled() const { return scratch_.soft_depth_enabled; }
    void set_tau_z(float tau_z) { scratch_.current_tau_z = tau_z; }
    float current_tau_z() const { return scratch_.current_tau_z; }
    void set_soft_depth_max_m(int m) { scratch_.soft_depth_max_m = m; }
    int soft_depth_max_m() const { return scratch_.soft_depth_max_m; }
    // 上传/下载 z 参数
    void upload_z(const float* h_z, int N);
    void download_z(float* h_z, int N);
    // 重置 z 的 Adam 状态（relocation 后调用）
    void reset_z_adam_state(int N);
    // 独立的 z Adam 更新（与 fused_adam_update 分离，避免破坏 grads_packed 布局）
    void fused_adam_update_z(int N, int adam_step, float lr, float beta1, float beta2,
                             float eps, float lr_factor_z);

    // Apply uniform scale constraint: copy d_rx → d_ry (rx=ry)
    void apply_uniform_scale(int N);

    // Compute per-shape visibility (blend weight sum on GPU)
    void compute_visibility(int N);
    void download_visibility(float* h_visibility, int N);
    void download_visibility_full(float* h_visibility_full, int N);

    // Forward render: returns rendered image in h_output [H*W*3]
    void render_forward(const ShapeParams* shapes, int N,
                        float* h_output);

    // Download current d_output (already rendered by render_forward_backward)
    // Avoids re-rendering for relocation error map computation
    void download_output(float* h_output);

    void upload_type_logits(const float* logits, int N);
    void download_type_logits(float* h_logits, int N);
    // Async download type logits to pinned buffer, returns event to wait on
    void download_type_logits_async(int N);
    // Try to process the latest async download (non-blocking), returns true if data was processed
    bool try_process_type_logits(int N, float* h_logits_out);
    void set_render_mode(int mode);
    int get_render_mode() const { return scratch_.render_mode; }
    void lock_types(const int* h_types, int N);
    void upload_types(const int* h_types, int N);
    void download_types(int* h_types, int N);

    // Forward + backward: compute MSE and gradients (GPU-resident params)
    // d_grad_history: optional ring buffer for per-shape gradient norms (relocation)
    // write_idx: write slot in d_grad_history ring buffer
    float render_forward_backward(int N,
        float* d_grad_history = nullptr, int write_idx = 0,
        float temperature = 1.0f,
        bool output_quadrant = false);

    // Download the per-quadrant RGB gradient buffer to host [N*4*3]
    void download_grad_rgb_quadrant(float* h_grad, int N);

    // Async copy current d_output to host pinned memory
    void async_copy_canvas(uint8_t* h_pinned_canvas, cudaEvent_t& event);

    // Async snapshot of all GPU-resident params to pinned memory (SoA layout)
    void async_snapshot_params(float* h_pinned, cudaEvent_t event, int N);

    // Upload initial shape parameters to GPU-resident buffers
    void upload_params_gpu(const ShapeParams* shapes, int N);

    // Fused GPU Adam: gradient clip + Adam update + param clamp + STE quantize
    // update_mask: 参数组屏蔽（解耦优化用）。bit0(1)=位置组(cx/cy/angle),
    //              bit1(2)=RGBA, bit2(4)=type_logits, bit3(8)=尺度组(rx/ry)。
    //              默认 15=全量更新（与 baseline 严格等价）。
    //              位置相位传 5（位置+logits），颜色相位传 10（RGBA+尺度）。
    //              被屏蔽组跳过 Adam 更新（m/v 保持不变），clamp/STE 全量执行（幂等）。
    // max_area_ratio: 面积硬上限（0=关，当前固定传 0）。
    void fused_adam_update(
        int N, int adam_step,
        float lr, float beta1, float beta2, float eps,
        float max_grad_norm,
        int W, int H,
        float lr_factor_xy, float lr_factor_r, float lr_factor_angle,
        float lr_factor_rgb, float lr_factor_a,
        float logits_lr_factor = 1.0f,
        int update_mask = 15,
        float max_area_ratio = 0.0f,
        bool uniform_scale = false);

    // === 数值校验辅助（低速单元素 cudaMemcpy）：读写 STE 缓冲中的单个参数。
    // param_offset: 0=cx,1=cy,2=rx,3=ry,4=angle, 8=opacity logit, 9+t=logit_t
    float get_ste_param(int param_offset, int idx);
    void set_ste_param(int param_offset, int idx, float value);

    // Download GPU-resident params to CPU ShapeParams
    void download_params_gpu(ShapeParams* shapes, int N);

    // Download GPU-resident params directly to SoA buffers (no ShapeParams intermediate)
    // Logit parameterization: h_colors and h_opacity are LOGITS (direct copy from GPU).
    // Caller is responsible for any linear conversion if needed.
    // h_colors: [N*3] interleaved (r,g,b per shape, logit space)
    // h_opacity: [N] in logit space (matches GPU d_opacity)
    // h_type_logits: [N*4] (may be nullptr to skip)
    void download_params_soa(
        float* h_cx, float* h_cy, float* h_rx, float* h_ry, float* h_angle,
        float* h_colors, float* h_opacity,
        float* h_type_logits, int N);

    // Upload SoA params directly to GPU-resident buffers (no ShapeParams intermediate)
    // Logit parameterization: h_colors and h_opacity are LOGITS (direct copy to GPU).
    // Caller is responsible for any linear conversion before calling.
    // h_colors: [N*3] interleaved (r,g,b per shape, logit space)
    // h_opacity: [N] in logit space
    // h_type_logits: [N*4] (may be nullptr to skip)
    // Also D2D copies to STE buffers (same as upload_params_gpu)
    void upload_params_soa(
        const float* h_cx, const float* h_cy, const float* h_rx, const float* h_ry,
        const float* h_angle, const float* h_colors, const float* h_opacity,
        const float* h_type_logits, int N);

    // Update frozen mask on GPU
    void set_frozen_mask(const int* frozen, int N);

    // Clamp opacity logit to [logit_min, +inf) on both d_opacity and d_ste_opacity
    // so rendered opacity sigmoid(logit/tau) >= opacity_min. Call after Adam update.
    // logit_min must be precomputed on host using the current tau.
    void clamp_opacity(int N, float logit_min);

    // Download packed gradients from GPU to CPU
    void download_grads_packed(float* h_grads, int N);

    // Reset GPU Adam state for specified shape indices
    void reset_adam_state_gpu(const std::vector<int>& indices);

    // Zero all GPU Adam state (used after shape reordering)
    void reset_adam_state_all_gpu(int N);

    // Compact GPU Adam state to match shape reordering:
    // new_to_old[new_i] = old_i (for surviving shapes), or -1 (for new/reloc shapes).
    // Surviving shapes keep their momentum; new shapes start with zeroed state.
    // Replaces the brute-force reset_adam_state_all_gpu after relocate_shapes.
    void compact_adam_state_gpu(const std::vector<int>& new_to_old, int N);

    // Synchronize the renderer's CUDA stream (call before freeing resources)
    void sync_stream();

    // Compute per-pixel error map on GPU (for relocate): (rendered - target)^2 summed over RGB
    // Writes to internal d_error_map_scratch; caller reads via download_error_map()
    void compute_error_map_gpu();
    // Download the GPU error map to host (synchronizes stream)
    void download_error_map(float* h_error_map);

    // === Diagnosis: per-pixel coverage count ===
    // 统计每个像素被多少个形状覆盖（ar > 0），独立 kernel，不影响渲染/优化。
    void compute_coverage_count(int N);
    void download_coverage_count(int* h_count, int total_pixels);

    // === Error Potential Field ===
    // Download current rendered alpha coverage [total_pixels] (0-255, synchronizes stream)
    void download_rendered_alpha(float* h_alpha);
    // Upload normalized potential gradient field [total_pixels] each (host -> device)
    void upload_potential_field(const float* h_grad_x, const float* h_grad_y);
    // Apply external force from potential gradient field to shape positions (GPU).
    // Modifies d_cx/d_cy and d_ste_cx/d_ste_cy directly (bypasses Adam, by design).
    // lambda: base strength (px/iter) already including annealing; per-shape soft gate
    // (1 - sigmoid((cov - gate_k)/gate_temp)) applied inside the kernel, where cov is
    // the rendered alpha coverage [0,1] sampled at the shape center.
    void apply_potential_force(int N, float lambda, float gate_k, float gate_temp);

    // === Frequency-Hole Potential Field (FH-PF) ===
    // Compute per-shape vis gate from d_visibility: gate_i = 1 - sigmoid((vis_i-thresh)/temp).
    // Must be called after compute_visibility(N). Writes to d_fh_vis_gate [N].
    void compute_fh_vis_gate(int N, float vis_thresh, float vis_temp);
    // Apply FH-PF external force: pos += lambda * gate_i * grad_U(center).
    // Reads d_pf_grad_x/y (uploaded via upload_potential_field) and d_fh_vis_gate.
    // Modifies d_cx/d_cy and d_ste_cx/d_ste_cy directly (bypasses Adam, by design).
    void apply_fh_potential_force(int N, float lambda);

    // === Auxiliary Visibility Gradient (Aux-Vis) ===
    // 配置：λ>0 启用。backward kernel 注入 ∂L_aux/∂ae_i = -λ·g_i·T_i，
    // 通过现有 ae→参数链传回 cx/cy/rx/ry/angle/opacity/logits（走 Adam）。
    // 与 FH-PF 的本质区别：FH-PF 是外力直接改位置（bypass Adam），Aux-Vis 是内生梯度（走 Adam）。
    void set_aux_vis_lambda(float lambda) { scratch_.aux_vis_lambda = lambda; }
    void set_aux_vis_thresh(float t) { scratch_.aux_vis_thresh = t; }
    void set_aux_vis_temp(float t) { scratch_.aux_vis_temp = t; }
    void set_aux_vis_only(bool v) { scratch_.aux_vis_only = v; }
    // 计算 g_i = 1 - sigmoid((vis_i - thresh)/temp)，写入 d_aux_gate [N]。
    // 必须在 compute_visibility(N) 之后调用（读取 d_visibility）。
    void compute_aux_vis_gate(int N);

    // === Stacking Penalty Loss (SP-Loss) ===
    // 配置：λ>0 启用。backward kernel 注入 ∂L_sp/∂ae_i = (2λ/N)·max(0,W-K)·(1-Π_{j≠i}(1-ae_j))。
    // K_allowed(p) = k_base + β·(局部方差(p)/全局均值方差)，在 set_target 时预计算。
    void set_sp_lambda(float lambda) { scratch_.sp_lambda = lambda; }
    void set_sp_k_base(float k) { scratch_.sp_k_base = k; }
    void set_sp_beta(float b) { scratch_.sp_beta = b; }

    // === Stacking Penalty Geometric (SP-GEO) ===
    // 配置：enable 时 forward 写 d_sp_S（供 host 算 d(p)），set_target 预计算 K_allowed。
    // 外力直接改 d_cx/d_cy/d_rx/d_ry 及 STE（绕过 Adam）。与 FH-PF 顺序叠加（先推后拉）。
    void set_sp_geo_enabled(bool v) { scratch_.sp_geo_active = v; }
    void set_sp_geo_loss_enabled(bool v) { scratch_.sp_geo_loss_active = v; }
    void set_sp_geo_d_ref(float r) { scratch_.sp_geo_d_ref = r; }
    // 下载 K_allowed[p] 到 host（SP-GEO host 端算 d(p) 用，复用已预计算的 d_k_allowed）
    void download_k_allowed(float* h_k, int total_pixels);
    // 下载 S(p)=Σae [total_pixels] 到 host（SP-GEO host 端算 d(p)=max(0,S-C-K) 用）
    void download_sp_S(float* h_S, int total_pixels);
    void download_sp_G(float* h_G, int total_pixels);
    // 上传 d(p) 势场 [total_pixels] + 归一化梯度 [total_pixels]×2 到设备
    void upload_sp_geo_field(const float* h_field, const float* h_gx, const float* h_gy, int total_pixels);
    // 施加 SP-GEO 外力：pos -= force·gate·∇d(center)，rx/ry -= shrink·d(center)。
    // gate=min(1, d(center)/d_ref)。读 d_sp_geo_field / d_sp_geo_grad_x/y。
    void apply_sp_geo_force(int N, float force, float shrink);

    int canvas_w() const { return canvas_w_; }
    int canvas_h() const { return canvas_h_; }

    // 只读访问 GPU 缓冲布局（本方法不改变渲染器状态）。
    const RenderScratch& scratch() const { return scratch_; }

private:
    float compute_mse_and_grad(int N);
    // 反向 MSE + grad 计算（双向渲染用）。仅在 bidir_lambda_ > 0 时调用。
    // 复用 mse_grad_kernel，但写入反向 buffer (d_grad_output_rev / d_mse_sum_rev)。
    // 简化版：不做 perceptual（pyramid）部分，仅做基本 MSE（含 alpha_loss_enabled 时的 alpha 通道）。
    void compute_mse_and_grad_rev(int N);
    void build_batch_descriptors(int N);

    int canvas_w_ = 0;
    int canvas_h_ = 0;
    int max_shapes_ = 0;
    float blur_sigma_ = 2.0f;
    float last_blur_sigma_ = -1.0f;    // cached sigma to skip redundant template blur
    bool target_blur_enabled_ = false; // 双模糊退火开关（默认关，baseline 不变）
    float last_target_blur_sigma_ = -1.0f;  // cached sigma to skip redundant target blur
    float bidir_lambda_ = 0.0f;        // 双向渲染 λ，0=禁用
    int valid_pixel_count_ = 0;        // EM-YMP: mask 内有效像素数（set_target 时统计，用于 mean_err 归一化）
    int deterministic_reduce_ = 0;     // R23: 0=off（默认），1=开启确定性归约
    RenderScratch scratch_;
    bool initialized_ = false;

    // CPU-side upload buffer for device-to-host param downloads
    std::vector<float> h_cx_, h_cy_, h_rx_, h_ry_;

    // Pre-allocated upload buffers (avoid per-iter heap allocation)
    std::vector<float> h_angle_buf_, h_colors_buf_, h_opacity_buf_;

    // Async debug statistics (double-buffered pinned memory + CUDA events)
    float* h_debug_logits_pinned_[2] = {nullptr, nullptr};  // [2][N*4]
    float* h_debug_grads_pinned_[2] = {nullptr, nullptr};   // [2][N*13]
    cudaEvent_t debug_events_[2] = {nullptr, nullptr};
    int debug_buf_idx_ = 0;
    bool debug_first_[2] = {true, true};  // whether the buffer has valid data

    // Async MSE readback (pinned memory + event, lag 1 iter to avoid CPU sync)
    float* h_mse_pinned_ = nullptr;
    cudaEvent_t mse_event_ = nullptr;
    float last_mse_ = 0.0f;
    bool mse_first_iter_ = true;

    // GPU-resident scale factor for gradient clipping (avoids CPU sync in fused_adam_update)
    float* d_clip_scale_ = nullptr;

    // Tile-batch scheduling state
    std::vector<int> h_tile_counts_;
    std::vector<TileBatchDesc> batch_descs_;
};

} // namespace vinylizer
