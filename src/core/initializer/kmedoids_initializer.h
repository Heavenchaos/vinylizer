#pragma once

#include "core/initializer/base_initializer.h"
#include <opencv2/core.hpp>

namespace vinylizer {

// ============================================================================
// KMedoidsInitializer —— 误差加权 k-medoids 构造式几何初始化
//
// 热核（assign / kmeans++ 的 D² 更新）自阶段 2b 起可跑在 GPU 上
// （src/cuda/kmedoids_kernels.{cu,cuh}），由 --km-backend auto|cpu|cuda 选择，
// 默认 auto（有可用 CUDA 设备→cuda，否则 cpu）。
// GPU 只负责"选标签 / 取 min"这一步确定性运算，labels / d2 拷回 host 后仍由
// 本文件原有的归约与累积采样逻辑处理，因此 GPU 与 CPU 路径逐位一致
// （见 tests/kmedoids_gpu/kmedoids_gpu_manifest.txt）。
// CPU 路径原样保留：它既是对拍基准，也是无 CUDA 设备时的 fallback。
//
// 这是 exp\scripts\py\construct_kmedoids.py + construct_common.py 的 C++ 移植
// （规格来源只读参考，逐函数对应）：
//   var_field      : 7x7 box filter（BORDER_REPLICATE）三通道局部方差和
//                    sum_c max(E[x^2]-E[x]^2, 0)，再 norm01（减 min、除 99.5 分位、clip[0,1]）
//                    （透明像素 RGB 按 GPU target 口径视为 0，见构造器 mask_opt 说明 C）
//   weight_field   : w = (base + (1-base)*f)^alpha（base=0.10），再 w /= mean(w)
//                    （透明像素权重置 0，见说明 B）
//   make_grid      : gh=ceil(H/g), gw=ceil(W/g)；坐标 ((x+0.5)*W/gw, (y+0.5)*H/gh)
//                    （候选点再按逐像素 mask 过滤：只留落点像素不透明的点，见说明 A）
//   kmeanspp/lloyd : 加权 D^2 采样 + iters 轮加权 Lloyd（质心 = 加权质心）
//   cell_stats     : 每 Voronoi 胞的 cnt / mass / cen（按 cnt 不加权）/ 二阶矩
//   cell_axes      : 结构张量主轴 lam1/lam2/theta
//   cellpix 半径   : area = cnt*cell_area, r_eq = sqrt(area/pi)，r = clip(r_eq*radius_scale, 1, 400)
//                    cell_area = 参与统计的不透明像素数 / 候选点数
//                    （全不透明图 = ROI 像素数 / 网格点数，与旧口径恒等）
//   颜色           : 胞质心处最近像素 RGB（np.round = round-half-to-even）
//   zorder radius_desc : 按 (rx*ry desc, rx desc) 排序，index 0 = 底层
//
// 研究定版参数（release CLI 默认）：grid=4, alpha=5, radius_scale=2.5, iters=15,
//   opacity=0.9, aniso=1.0（rx==ry）, orient=cell, weight=var。
//
// ★ RNG 差异声明（重要）：本实现自带 RNG（std::mt19937_64 + 手写累积分布查找），
//   **不复现 numpy 的 PCG64 / Generator.choice**。因此几何与 Python 版**不逐点一致**，
//   只保证：(a) 同 seed 同输出（单线程、固定浮点顺序，严格确定性）；
//   (b) 统计质量与最终 MSE 同量级（见 release\tests\kmedoids\ 验收记录）。
//
// 挂载点：BaseInitializer 接口足够承载本策略（几何 + 颜色全部写入 OptParams 的
//   logit 域参数），无需改动接口。type 不在本策略内（对齐 Python 的
//   `--init-shapes-mode reset`：type_logits 全 0，由 DARTS 重搜）。
// ============================================================================
class KMedoidsInitializer : public BaseInitializer {
public:
    // target_rgb: 目标 RGB 图像 (H, W, 3), CV_8UC3, [0,255]
    // mask_opt: 优化分辨率的"不透明 mask"（CV_8U，>0 = 不透明，尺寸 W_opt x H_opt，
    //   与 optimizer.cpp 里构造 GPU target 用的 mask_small_ 同一张）。三项作用：
    //     A) 候选网格点只保留落在不透明像素上的点（逐像素口径，不再用包围盒）；
    //        同时 cell_area 口径同步改为"不透明像素数 / 候选点数"，否则半径会缩水。
    //     B) 权重场（var_field → w）中透明像素权重置 0 → 透明区采样概率为 0。
    //     C) var_field/取色都用"透明像素 RGB=0"的图（与 GPU target 一致，
    //        见 optimizer.cpp:146-163 的 h_target 构造）。
    //   全不透明图（mask 全 1）下这三项全部退化为恒等 → 输出逐位不变。
    // W_opt/H_opt: 优化分辨率（opt_resolution >= 图像边长时即原生尺寸）
    // content_rect: 内容区 ROI（优化分辨率坐标系）。k-medoids 的候选网格只铺在
    //   内容区内，形状中心因此不会落进透明 padding 区（那里的目标 alpha=0，
    //   形状训练后必然全透明 → 白占 FH6 图层额度）。空矩形 = 整张画布。
    //   注意：形状半径不受此限制（越界部分由 padding 保护并在输出时裁剪）。
    KMedoidsInitializer(const cv::Mat& target_rgb, const cv::Mat& mask_opt,
                        int W_opt, int H_opt,
                        int grid, float alpha, float radius_scale, int iters,
                        float opacity, const cv::Rect& content_rect = cv::Rect());

    void initialize(OptParams& params, int num_template_types,
                    const OptimizerConfig& config,
                    std::mt19937& rng) override;

private:
    cv::Mat target_rgb_;
    cv::Mat mask_opt_;      // CV_8U，>0 = 不透明（优化分辨率，含 padding 画布）
    int W_opt_ = 0, H_opt_ = 0;
    cv::Rect content_rect_;
    int grid_ = 4;
    float alpha_ = 5.0f;
    float radius_scale_ = 2.5f;
    int iters_ = 15;
    float opacity_ = 0.9f;
};

} // namespace vinylizer
