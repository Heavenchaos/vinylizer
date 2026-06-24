#pragma once

#include "common/types.h"
#include <string>

namespace vinylizer {

struct PipelineConfig {
    std::string input_path;
    std::string output_path;
    int num_shapes = 500;
    int canvas_w = 512;
    int canvas_h = 512;
    int num_cycles = 3;
    float lr_start = 0.1f;
    float lr_end = 0.01f;
};

void run_pipeline(const PipelineConfig& config);

} // namespace vinylizer
