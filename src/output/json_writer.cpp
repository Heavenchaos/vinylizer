#include "output/json_writer.h"
#include "common/logger.h"
#include "common/path_utils.h"
#include <json.hpp>
#include <fstream>

namespace vinylizer {

void write_json(const std::string& path, const std::vector<Shape>& shapes,
                int canvas_w, int canvas_h) {
    nlohmann::ordered_json shapes_arr = nlohmann::ordered_json::array();

    for (const auto& shape : shapes) {
        nlohmann::ordered_json s;
        s["type"] = shape.type;

        if (shape.is_rect()) {
            s["data"] = {shape.cx, shape.cy, shape.rx, shape.ry};
        } else {
            s["data"] = {shape.cx, shape.cy, shape.rx, shape.ry,
                         fmod(shape.angle, 360.0f)};
        }

        s["color"] = {
            shape.color.r, shape.color.g, shape.color.b,
            static_cast<int>(std::lround(shape.opacity * 255))
        };
        s["score"] = shape.score;

        shapes_arr.push_back(s);
    }

    nlohmann::ordered_json payload;
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
