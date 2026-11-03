#pragma once

#include "common/types.h"
#include <string>
#include <vector>

namespace vinylizer {

void write_json(const std::string& path, const std::vector<Shape>& shapes,
                int canvas_w, int canvas_h,
                const std::vector<std::string>& stamp_names = {});

} // namespace vinylizer
