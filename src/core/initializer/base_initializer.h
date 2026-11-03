#pragma once

#include "core/optimizer.h"  // for OptParams, OptimizerConfig
#include <random>

namespace vinylizer {

// 初始化器为单个形状返回的参数包
struct ShapeInitParams {
    float rx = 0, ry = 0;   // 恒等（rx == ry）
    float angle = 0;
    // r/g/b/a are LOGIT-space values (sigmoid parameterization, aligned with
    // diffbmp). They are written directly to OptParams.color_r/g/b/a.
    float r = 0, g = 0, b = 0, a = 0;
};

// 抽象基类：形状初始化策略
class BaseInitializer {
public:
    virtual ~BaseInitializer() = default;

    // 完整初始化：采样 N 个形状位置并填充所有参数到 OptParams
    // 类型分配由 GradientOptimizer 在 initialize() 之后单独处理
    virtual void initialize(OptParams& params, int num_template_types,
                            const OptimizerConfig& config,
                            std::mt19937& rng) = 0;

    // 在给定位置生成参数，用于 relocation
    // 默认返回占位值（ImportanceInitializer 不覆盖）
    virtual ShapeInitParams sample_at(float cx, float cy,
                                      std::mt19937& rng) const {
        (void)cx; (void)cy; (void)rng;
        return {0, 0, 0, 0, 0, 0, 0};
    }

    // 是否支持 sample_at（VarianceInitializer 返回 true）
    virtual bool supports_sample_at() const { return false; }
};

} // namespace vinylizer
