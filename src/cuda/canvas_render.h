#pragma once

#include "common/types.h"
#include <opencv2/core.hpp>
#include <vector>

namespace vinylizer {

cv::Mat canvas_render_gpu(const std::vector<Shape>& shapes, int w, int h);

} // namespace vinylizer
