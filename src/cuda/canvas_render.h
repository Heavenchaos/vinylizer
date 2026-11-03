#pragma once

#include "common/types.h"
#include <opencv2/core.hpp>
#include <vector>

namespace vinylizer {

cv::Mat canvas_render_gpu(
    const std::vector<Shape>& shapes, int w, int h,
    const std::vector<std::string>& stamp_names = {},
    const std::vector<cv::Mat>& stamp_images = {});

} // namespace vinylizer
