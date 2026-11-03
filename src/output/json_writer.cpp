#include "output/json_writer.h"
#include "common/logger.h"
#include "common/path_utils.h"
#include <json.hpp>
#include <fstream>

namespace vinylizer {

void write_json(const std::string& path, const std::vector<Shape>& shapes,
                int canvas_w, int canvas_h,
                const std::vector<std::string>& stamp_names) {
    nlohmann::ordered_json payload;

    payload["canvas_w"] = canvas_w;
    payload["canvas_h"] = canvas_h;
    if (!stamp_names.empty()) {
        nlohmann::ordered_json stamps_arr = nlohmann::ordered_json::array();
        for (const auto& name : stamp_names) {
            stamps_arr.push_back(name);
        }
        payload["stamps"] = stamps_arr;
    }

    nlohmann::ordered_json shapes_arr = nlohmann::ordered_json::array();

    for (const auto& shape : shapes) {
        nlohmann::ordered_json s;
        if (!shape.stamp_name.empty()) {
            s["type"] = shape.stamp_name;
        } else {
            s["type"] = shape.type;
        }

        s["data"] = {shape.cx, shape.cy, shape.rx, shape.ry,
                     fmod(shape.angle, 360.0f)};

        s["color"] = {
            shape.color.r, shape.color.g, shape.color.b,
            shape.opacity * 255.0f
        };
        s["score"] = shape.score;
        // 输出 z（soft depth），用于 preview 渲染时按 z 排序。
        // z 大 = 底层, z 小 = 顶层；默认 0 表示未启用。
        s["z"] = shape.z;

        shapes_arr.push_back(s);
    }

    payload["shapes"] = shapes_arr;

    std::ofstream file = open_ofstream(path);
    if (!file.is_open()) {
        VIN_ERROR("Failed to open JSON file for writing: %s", path.c_str());
        return;
    }
    file << payload.dump(2);
    VIN_INFO("Wrote %d shapes to %s", static_cast<int>(shapes_arr.size()), path.c_str());
}

} // namespace vinylizer
