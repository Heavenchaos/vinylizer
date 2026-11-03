// KMedoidsInitializer 实现 —— 见 kmedoids_initializer.h 顶部的规格对照与 RNG 声明。
// 参考实现（只读）：exp\scripts\py\construct_common.py / construct_kmedoids.py

#include "core/initializer/kmedoids_initializer.h"
#include "common/logger.h"
#include "cuda/kmedoids_kernels.cuh"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace vinylizer {

namespace {

// —— 与 Python 版一一对应的常量（construct_common.py）——
constexpr double kBaseDensity = 0.10;   // weight_field base
constexpr int    kVarWindow   = 7;      // var_field win
constexpr float  kMinR        = 1.0f;   // MIN_R
constexpr float  kMaxR        = 400.0f; // MAX_R
constexpr double kEps         = 1e-12;

double now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

// np.percentile(a, pct)（linear 插值；会就地排序输入副本）
float percentile_linear(std::vector<float>& v, double pct) {
    std::sort(v.begin(), v.end());
    const double pos = (pct / 100.0) * static_cast<double>(v.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = static_cast<size_t>(std::ceil(pos));
    const double frac = pos - static_cast<double>(lo);
    return static_cast<float>(static_cast<double>(v[lo]) * (1.0 - frac) +
                              static_cast<double>(v[hi]) * frac);
}

// norm01（construct_common.py:62-68）：减 min、除 99.5 分位、clip[0,1]
cv::Mat norm01(const cv::Mat& a) {
    std::vector<float> tmp(a.ptr<float>(0), a.ptr<float>(0) + a.total());
    const float lo = *std::min_element(tmp.begin(), tmp.end());
    const float hi = percentile_linear(tmp, 99.5);
    cv::Mat out = a.clone();
    if (static_cast<double>(hi) - static_cast<double>(lo) < kEps) {
        out.setTo(0.0f);
        return out;
    }
    const float inv = 1.0f / (hi - lo);
    float* p = out.ptr<float>(0);
    for (size_t i = 0; i < out.total(); ++i) {
        p[i] = std::min(1.0f, std::max(0.0f, (p[i] - lo) * inv));
    }
    return out;
}

// 最近 site 分配（construct_common.py:161-177）：欧氏平方距离，argmin 取最小索引。
// Python 为 float32 运算，这里保持 float32 以对齐数值行为。
void assign_labels(const std::vector<float>& gx, const std::vector<float>& gy,
                   const std::vector<float>& sx, const std::vector<float>& sy,
                   std::vector<int>& lab) {
    const int P = static_cast<int>(gx.size());
    const int N = static_cast<int>(sx.size());
    lab.resize(P);
    for (int p = 0; p < P; ++p) {
        const float x = gx[p], y = gy[p];
        float best = std::numeric_limits<float>::infinity();
        int best_k = 0;
        for (int k = 0; k < N; ++k) {
            const float dx = x - sx[k];
            const float dy = y - sy[k];
            const float d = dx * dx + dy * dy;
            if (d < best) { best = d; best_k = k; }
        }
        lab[p] = best_k;
    }
}

// 累积分布采样（等价 numpy Generator.choice(P, p=cdf 归一化) 的 searchsorted）：
// 取第一个 cdf[i] >= u 的 i。
int cdf_sample(const std::vector<double>& cdf, double u) {
    auto it = std::lower_bound(cdf.begin(), cdf.end(), u);
    int i = static_cast<int>(it - cdf.begin());
    if (i >= static_cast<int>(cdf.size())) i = static_cast<int>(cdf.size()) - 1;
    return i;
}

} // namespace

KMedoidsInitializer::KMedoidsInitializer(const cv::Mat& target_rgb,
                                        const cv::Mat& mask_opt,
                                        int W_opt, int H_opt,
                                        int grid, float alpha, float radius_scale,
                                        int iters, float opacity,
                                        const cv::Rect& content_rect)
    : target_rgb_(target_rgb.clone()), mask_opt_(mask_opt.clone()),
      W_opt_(W_opt), H_opt_(H_opt),
      content_rect_(content_rect),
      grid_(grid), alpha_(alpha), radius_scale_(radius_scale),
      iters_(iters), opacity_(opacity) {}

void KMedoidsInitializer::initialize(OptParams& params, int /*num_template_types*/,
                                     const OptimizerConfig& config,
                                     std::mt19937& /*rng*/)
{
    const int n = params.N;

    // ── backend 选择（--km-backend auto|cpu|cuda）──
    // auto: 有可用 CUDA 设备 → cuda，否则 cpu。cpu 永远可用（对拍基准 + fallback）。
    // 放在计时起点之前，避免设备查询本身计入 profile。
    bool use_cuda = false;
    {
        const std::string& want = config.km_backend;
        const int ndev = (want == "cpu") ? 0 : km_cuda_device_count();
        use_cuda = (ndev > 0) && (want == "cuda" || want == "auto");
        if (want == "cuda" && ndev <= 0) {
            VIN_WARN("kmedoids: --km-backend cuda requested but no CUDA device available; "
                     "falling back to cpu");
        }
    }

    // ── 0. 目标缩放到 opt 分辨率（opt_resolution >= 图像边长时 scale=1，为恒等拷贝）──
    // 随后裁到内容区 ROI：网格/权重/统计只在内容区内进行，形状中心因此不会落进
    // 透明 padding 区（详见头文件说明）。写回 params 时再加回 ROI 原点偏移。
    const double t_start = now_ms();
    double t_mark = t_start;
    cv::Mat target_small;
    cv::resize(target_rgb_, target_small, {W_opt_, H_opt_}, 0, 0, cv::INTER_AREA);
    cv::Rect roi(0, 0, W_opt_, H_opt_);
    if (!content_rect_.empty()) {
        roi = roi & (content_rect_ & cv::Rect(0, 0, W_opt_, H_opt_));
        if (roi.width <= 0 || roi.height <= 0) roi = cv::Rect(0, 0, W_opt_, H_opt_);
    }
    if (roi != cv::Rect(0, 0, W_opt_, H_opt_)) target_small = target_small(roi).clone();
    const int H = target_small.rows, W = target_small.cols;

    // 逐像素不透明 mask（与 target_small 同 ROI 同尺寸）：>0 = 不透明。
    // opaque_px = ROI 内实际不透明像素数（cell_area 的分子，见第 3 步）。
    // mask_absent = ROI 全透明（退化输入，如整图透明的 PNG）：无从过滤，
    // 全部退化为旧行为（P = 网格点数、cell_area = ROI 面积 / 网格点数）。
    const cv::Mat mask_roi = mask_opt_(roi);
    const int opaque_px = cv::countNonZero(mask_roi);
    const bool mask_absent = (opaque_px == 0);

    // C: KM 读到的 RGB 与 GPU target 口径一致（透明像素 RGB 视为 0）。
    // 全不透明图 mask 全 1 → setTo 无像素命中 → 逐位不变。
    cv::Mat target_masked = target_small.clone();
    if (!mask_absent) target_masked.setTo(cv::Scalar(0, 0, 0), mask_roi == 0);
    cv::Mat target_f;
    target_masked.convertTo(target_f, CV_32FC3, 1.0 / 255.0);
    const double t_prep = now_ms();

    // ── 1. var_field：7x7 box filter（REPLICATE）三通道局部方差和 + norm01 ──
    cv::Mat acc = cv::Mat::zeros(H, W, CV_32FC1);
    for (int c = 0; c < 3; ++c) {
        cv::Mat ch, m, m2;
        cv::extractChannel(target_f, ch, c);
        cv::boxFilter(ch, m, -1, {kVarWindow, kVarWindow}, {-1, -1}, true,
                      cv::BORDER_REPLICATE);
        cv::boxFilter(ch.mul(ch), m2, -1, {kVarWindow, kVarWindow}, {-1, -1}, true,
                      cv::BORDER_REPLICATE);
        cv::Mat d = m2 - m.mul(m);
        cv::max(d, 0.0, d);          // max(E[x^2]-E[x]^2, 0) 逐通道
        acc += d;
    }
    cv::Mat var01 = norm01(acc);
    const double t_var = now_ms();

    // ── 2. weight_field：w = (base + (1-base)*f)^alpha，再 w /= mean(w) ──
    cv::Mat w_full(H, W, CV_32FC1);
    {
        const float* src = var01.ptr<float>(0);
        float* dst = w_full.ptr<float>(0);
        for (size_t i = 0; i < w_full.total(); ++i) {
            dst[i] = static_cast<float>(
                std::pow(kBaseDensity + (1.0 - kBaseDensity) * static_cast<double>(src[i]),
                         static_cast<double>(alpha_)));
        }
    }
    // B: 透明像素权重置 0 → 透明区（含 INTER_AREA 降采样与 kmeans++/Lloyd 加权）
    // 采样概率为 0。全不透明图 mask 全 1 → 一个像素都不改（逐位 no-op）。
    if (!mask_absent) {
        for (int y = 0; y < H; ++y) {
            const uint8_t* mr = mask_roi.ptr<uint8_t>(y);
            float* wr = w_full.ptr<float>(y);
            for (int x = 0; x < W; ++x) {
                if (mr[x] == 0) wr[x] = 0.0f;
            }
        }
    }
    {
        double sum = 0.0;
        const float* p = w_full.ptr<float>(0);
        for (size_t i = 0; i < w_full.total(); ++i) sum += p[i];
        // 均值口径 = 参与统计的不透明像素（全不透明图 = H*W，与旧口径逐位一致）
        const double den = mask_absent ? static_cast<double>(W) * static_cast<double>(H)
                                       : static_cast<double>(opaque_px);
        double mean = sum / den;
        if (mean < kEps) mean = kEps;
        float* p2 = w_full.ptr<float>(0);
        for (size_t i = 0; i < w_full.total(); ++i)
            p2[i] = static_cast<float>(static_cast<double>(p2[i]) / mean);
    }
    const double t_w = now_ms();

    // ── 3. make_grid + 逐像素 mask 过滤 + w 降采样到网格（INTER_AREA）──
    const int g = std::max(1, grid_);
    const int gh = static_cast<int>(std::ceil(static_cast<double>(H) / g));
    const int gw = static_cast<int>(std::ceil(static_cast<double>(W) / g));
    const int P_grid = gw * gh;
    std::vector<float> gx_all(P_grid), gy_all(P_grid);
    for (int y = 0; y < gh; ++y) {
        for (int x = 0; x < gw; ++x) {
            gx_all[y * gw + x] = static_cast<float>((x + 0.5) * (static_cast<double>(W) / gw));
            gy_all[y * gw + x] = static_cast<float>((y + 0.5) * (static_cast<double>(H) / gh));
        }
    }
    cv::Mat w_small;
    cv::resize(w_full, w_small, {gw, gh}, 0, 0, cv::INTER_AREA);
    const float* w_small_p = w_small.ptr<float>(0);

    // 候选点 = 网格中心；用像素 mask 过滤掉落在透明像素上的点（逐像素口径，
    // 不再是包围盒）。代表像素 = 网格中心所在像素（floor），与旧 Python 复刻一致。
    auto build_keep = [&](bool use_mask) {
        std::vector<int> k;
        k.reserve(P_grid);
        for (int y = 0; y < gh; ++y) {
            for (int x = 0; x < gw; ++x) {
                const int idx = y * gw + x;
                if (use_mask) {
                    const int xi = std::min(W - 1, static_cast<int>(gx_all[idx]));
                    const int yi = std::min(H - 1, static_cast<int>(gy_all[idx]));
                    if (mask_roi.at<uint8_t>(yi, xi) == 0) continue;
                }
                k.push_back(idx);
            }
        }
        return k;
    };
    bool use_mask = !mask_absent;
    std::vector<int> keep = build_keep(use_mask);
    if (keep.empty()) { use_mask = false; keep = build_keep(false); }  // 退化保护（P 不可为 0）
    const int P = static_cast<int>(keep.size());
    std::vector<float> gx(P), gy(P);
    std::vector<double> w(P);
    for (int i = 0; i < P; ++i) {
        gx[i] = gx_all[keep[i]];
        gy[i] = gy_all[keep[i]];
        w[i] = static_cast<double>(w_small_p[keep[i]]);
    }
    // cell_area 口径：参与统计的不透明像素数 / 候选点数
    // （旧口径 = ROI 像素数 / 网格点数；ROI 全不透明时两者恒等 → 逐位不变）
    const double cell_area = use_mask
        ? static_cast<double>(opaque_px) / P
        : static_cast<double>(W) * static_cast<double>(H) / P;
    const double t_grid = now_ms();

    // ── GPU 上下文（独立分配，init 结束即释放；失败则回落 CPU）──
    std::unique_ptr<KmGpuAssigner> gpu;
    if (use_cuda) {
        gpu.reset(new KmGpuAssigner(P, n));
        if (!gpu->ok()) {
            VIN_WARN("kmedoids: CUDA buffer allocation failed (%s); falling back to cpu",
                     gpu->error());
            gpu.reset();
            use_cuda = false;
        } else {
            gpu->upload_coords(gx.data(), gy.data(), P);
        }
    }
    const char* backend_name = use_cuda ? "cuda" : "cpu";

    // ── 4. kmeanspp：加权 D^2 采样 ──
    std::mt19937_64 rng64(static_cast<uint64_t>(config.random_seed));
    std::uniform_real_distribution<double> uni01(0.0, 1.0);
    std::uniform_int_distribution<int> uni_idx(0, P - 1);

    std::vector<float> sx(n, 0.0f), sy(n, 0.0f);
    std::vector<double> cdf(P);
    {
        // p = w / sum(w) → 第 0 个 site
        double tot = 0.0;
        for (int i = 0; i < P; ++i) tot += w[i];
        if (tot <= 0.0) tot = 1.0;
        double c = 0.0;
        for (int i = 0; i < P; ++i) { c += w[i] / tot; cdf[i] = c; }
        const int i0 = cdf_sample(cdf, uni01(rng64));
        sx[0] = gx[i0];
        sy[0] = gy[i0];
    }
    // d2：CPU 路径是本函数的 host vector；GPU 路径是 KmGpuAssigner 持有的 pinned
    // host 缓冲（设备端算完直接拷回该缓冲，避免 pageable 中转拷贝）。
    std::vector<float> d2_cpu;
    float* d2 = nullptr;
    if (use_cuda) {
        gpu->d2_init(sx[0], sy[0]);
        d2 = gpu->d2_host();
    } else {
        d2_cpu.resize(P);
        d2 = d2_cpu.data();
        for (int p = 0; p < P; ++p) {
            const float dx = gx[p] - sx[0], dy = gy[p] - sy[0];
            d2[p] = dx * dx + dy * dy;
        }
    }
    for (int k = 1; k < n; ++k) {
        // q = w * d2, 采样 p = q / sum(q)；sum(q)<=0 时退化为均匀随机索引
        double tot = 0.0;
        for (int i = 0; i < P; ++i) { cdf[i] = w[i] * static_cast<double>(d2[i]); tot += cdf[i]; }
        int ik;
        if (tot <= 0.0) {
            ik = uni_idx(rng64);
        } else {
            double c = 0.0;
            for (int i = 0; i < P; ++i) { c += cdf[i] / tot; cdf[i] = c; }
            ik = cdf_sample(cdf, uni01(rng64));
        }
        sx[k] = gx[ik];
        sy[k] = gy[ik];
        // d2 = min(d2, ||coords - site_k||^2)（float32，对齐 numpy）
        const float ax = sx[k], ay = sy[k];
        if (use_cuda) {
            // GPU 原地更新 + 拷回 pinned host 缓冲（后续 cdf 累积采样仍走 CPU 原逻辑）
            gpu->d2_update_download(ax, ay);
        } else {
            for (int p = 0; p < P; ++p) {
                const float dx = gx[p] - ax, dy = gy[p] - ay;
                const float dk = dx * dx + dy * dy;
                if (dk < d2[p]) d2[p] = dk;
            }
        }
    }
    const double t_pp = now_ms();

    // ── 5. lloyd：iters 轮 assign → 加权质心（空簇保留原值）──
    std::vector<int> lab(P);
    std::vector<double> mass(n), mx(n), my(n);
    int empty_last = 0;
    for (int it = 0; it < iters_; ++it) {
        if (use_cuda) gpu->assign(sx.data(), sy.data(), n, lab.data());
        else assign_labels(gx, gy, sx, sy, lab);
        std::fill(mass.begin(), mass.end(), 0.0);
        std::fill(mx.begin(), mx.end(), 0.0);
        std::fill(my.begin(), my.end(), 0.0);
        for (int p = 0; p < P; ++p) {
            const int k = lab[p];
            const double wk = w[p];
            mass[k] += wk;
            mx[k] += wk * static_cast<double>(gx[p]);
            my[k] += wk * static_cast<double>(gy[p]);
        }
        int empty = 0;
        for (int k = 0; k < n; ++k) {
            if (mass[k] > kEps) {
                sx[k] = static_cast<float>(mx[k] / mass[k]);
                sy[k] = static_cast<float>(my[k] / mass[k]);
            } else {
                ++empty;  // 死簇保留原值（Python: new = sites，仅 alive 更新）
            }
        }
        empty_last = empty;
    }
    const double t_lloyd = now_ms();

    // ── 6. cell_stats：cnt / mass / cen（按 cnt 不加权）/ 相对 cen 的二阶矩 ──
    std::vector<long long> cnt(n, 0);
    std::vector<double> sxs(n, 0.0), sys(n, 0.0);
    std::vector<double> cenx(n, 0.0), ceny(n, 0.0);
    {
        if (use_cuda) gpu->assign(sx.data(), sy.data(), n, lab.data());
        else assign_labels(gx, gy, sx, sy, lab);   // lloyd 后的最终标签
        for (int p = 0; p < P; ++p) {
            const int k = lab[p];
            cnt[k]++;
            sxs[k] += static_cast<double>(gx[p]);
            sys[k] += static_cast<double>(gy[p]);
        }
        std::vector<char> ok(n);
        for (int k = 0; k < n; ++k) {
            ok[k] = (cnt[k] > 1) ? 1 : 0;
            // fallback = 最终 site 坐标（cell_stats 的 fallback=sites）
            cenx[k] = ok[k] ? (sxs[k] / static_cast<double>(cnt[k])) : sx[k];
            ceny[k] = ok[k] ? (sys[k] / static_cast<double>(cnt[k])) : sy[k];
        }
        std::vector<double> mxx(n, 0.0), myy(n, 0.0), mxy(n, 0.0);
        for (int p = 0; p < P; ++p) {
            const int k = lab[p];
            const double dx = static_cast<double>(gx[p]) - cenx[k];
            const double dy = static_cast<double>(gy[p]) - ceny[k];
            mxx[k] += dx * dx;
            myy[k] += dy * dy;
            mxy[k] += dx * dy;
        }
        for (int k = 0; k < n; ++k) {
            if (ok[k]) {
                const double inv = 1.0 / static_cast<double>(cnt[k]);
                mxx[k] *= inv; myy[k] *= inv; mxy[k] *= inv;
            }
        }

        // ── 7. cell_axes + cellpix 半径 + 颜色 + z-order ──
        std::vector<float> rx(n, 0.0f), ry(n, 0.0f), ang(n, 0.0f);
        std::vector<float> cr(n, 0.0f), cg(n, 0.0f), cb(n, 0.0f);
        const uint8_t* pix = target_masked.ptr<uint8_t>(0);
        const int W1 = W - 1, H1 = H - 1;
        for (int k = 0; k < n; ++k) {
            // 结构张量主轴 theta（construct_common.cell_axes）；radius-mode=cellpix
            // 不使用 lam1/lam2，故只算 theta
            const double theta = 0.5 * std::atan2(2.0 * mxy[k], mxx[k] - myy[k] + kEps);

            // radius-mode=cellpix：area = cnt * cell_area, r_eq = sqrt(area/pi)
            const double area = static_cast<double>(cnt[k]) * cell_area;
            const double r_eq = std::sqrt(std::max(area, 0.0) / M_PI);
            const float r_major = static_cast<float>(r_eq);
            const float r_minor = static_cast<float>(r_eq);   // aniso = 1.0
            float rx0 = std::min(kMaxR, std::max(kMinR, r_major * radius_scale_));
            float ry0 = std::min(kMaxR, std::max(kMinR, r_minor * radius_scale_));
            float a = static_cast<float>(theta);
            if (ry0 > rx0) { a += static_cast<float>(M_PI / 2.0); }
            rx[k] = std::max(rx0, ry0);
            ry[k] = std::min(rx0, ry0);
            // degrees + 归一到 [0,360)
            float deg = static_cast<float>(a * (180.0 / M_PI) + 360.0);
            deg = std::fmod(deg, 360.0f);
            if (deg < 0.0f) deg += 360.0f;
            ang[k] = deg;

            // 颜色：质心处最近像素（np.round = round-half-to-even → std::nearbyint）
            const int xi = std::min(W1, std::max(0, static_cast<int>(std::nearbyint(cenx[k]))));
            const int yi = std::min(H1, std::max(0, static_cast<int>(std::nearbyint(ceny[k]))));
            const uint8_t* px = pix + (static_cast<size_t>(yi) * W + xi) * 3;
            cr[k] = static_cast<float>(px[0]);
            cg[k] = static_cast<float>(px[1]);
            cb[k] = static_cast<float>(px[2]);
        }

        // zorder radius_desc：(-rx, -rx*ry) 的 lexsort ⇒ (rx*ry 降序, rx 降序)
        std::vector<int> order(n);
        for (int i = 0; i < n; ++i) order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            const float aa = rx[a] * ry[a], ab = rx[b] * ry[b];
            if (aa != ab) return aa > ab;
            return rx[a] > rx[b];
        });

        // ── 8. 写入 OptParams（logit 域；index 0 = 底层，与 zorder 数组同序）──
        // 坐标加回内容区原点偏移 → 参数回到优化画布（含 padding）坐标系。
        const float op_logit = vin_logit(opacity_);
        for (int j = 0; j < n; ++j) {
            const int k = order[j];
            params.cx[j] = static_cast<float>(cenx[k]) + roi.x;
            params.cy[j] = static_cast<float>(ceny[k]) + roi.y;
            params.rx[j] = rx[k];
            params.ry[j] = ry[k];
            params.angle[j] = ang[k];
            params.color_r[j] = vin_logit(cr[k] / 255.0f);
            params.color_g[j] = vin_logit(cg[k] / 255.0f);
            params.color_b[j] = vin_logit(cb[k] / 255.0f);
            params.color_a[j] = op_logit;
        }
        VIN_INFO("KMedoidsInitializer: cells empty(cnt<=1)=%d, lloyd empty-mass=%d",
                 (int)std::count(ok.begin(), ok.end(), (char)0), empty_last);
    }
    const double t_done = now_ms();

    // ── 阶段耗时 profile（阶段 2b：CPU/GPU 后端对照）──
    VIN_INFO("KMedoidsInitializer: N=%d grid=%d(%dx%d P=%d) alpha=%.2f rs=%.2f iters=%d backend=%s",
             n, g, gw, gh, P, alpha_, radius_scale_, iters_, backend_name);
    VIN_INFO("KMedoidsInitializer: mask-aware candidates P=%d (grid %dx%d=%d, opaque_px=%d, "
             "cell_area=%.4f, filtering=%s)",
             P, gw, gh, P_grid, (int)opaque_px, cell_area, use_mask ? "on" : "off");
    VIN_INFO("KMedoidsInitializer: canvas=%dx%d content ROI=[%d,%d)+%dx%d "
             "(grid confined to content; radii may still exceed it)",
             W_opt_, H_opt_, roi.x, roi.y, roi.width, roi.height);
    VIN_INFO("[km-profile] backend=%s resize+var_field_prep=%.1fms var_field=%.1fms "
             "weight=%.1fms grid+resize=%.1fms kmeanspp=%.1fms lloyd(%d)=%.1fms "
             "stats+axes+zorder+write=%.1fms  TOTAL=%.1fms",
             backend_name, t_prep - t_start, t_var - t_prep, t_w - t_var, t_grid - t_w,
             t_pp - t_grid, iters_, t_lloyd - t_pp, t_done - t_lloyd, t_done - t_start);
}

} // namespace vinylizer
