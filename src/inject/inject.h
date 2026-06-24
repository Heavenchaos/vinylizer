#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#pragma once

#include "common/types.h"
#include <json.hpp>
#include <string>
#include <vector>
#include <cstdint>
#include <tuple>

#ifdef _WIN32
#include <windows.h>
#endif

namespace vinylizer {

struct InjectConfig {
    std::string json_path;
    int layer_count = 0;     // Number of placeholder shapes in FH6
    int process_id = 0;      // FH6 process ID, 0 = auto-detect
};

struct LayerInfo {
    int index = 0;
    uint64_t ptr = 0;
    float pos_x = 0, pos_y = 0;
    float scale_x = 0, scale_y = 0;
    float rotation = 0;
    uint8_t color[4] = {0, 0, 0, 255};
    uint8_t shape_id = 0;
    uint8_t mask = 0;
};

// Find FH6 process
std::pair<int, std::string> find_fh6_process();

// Validate JSON file
std::vector<nlohmann::json> validate_json(const std::string& path);

// Main inject function
bool inject_shapes(const InjectConfig& config);

// Lower-level API
uint64_t locate_layer_table(int pid, int layer_count);

// Layer statistics
struct LayerStats {
    int total_valid = 0;
    int count_101 = 0;   // rectangle
    int count_102 = 0;   // ellipse
};
LayerStats count_valid_layers(int pid, uint64_t table_address, int layer_count);

// Sample layers for user confirmation
std::vector<LayerInfo> sample_layers(int pid, uint64_t table_address, int layer_count, int n = 6);

// With PID (opens/closes handle each call — for simple CLI usage)
bool write_layer(int pid, uint64_t table_address, int index, const nlohmann::json& shape_data);
bool write_clear_layer(int pid, uint64_t table_address, int index);
LayerInfo read_layer_info(int pid, uint64_t layer_ptr);

// With pre-opened HANDLE (reuse across calls — for inject worker)
bool write_layer_h(HANDLE h, uint64_t table_address, int index, const nlohmann::json& shape_data);
bool write_clear_layer_h(HANDLE h, uint64_t table_address, int index);

} // namespace vinylizer
