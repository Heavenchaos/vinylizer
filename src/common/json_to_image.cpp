#include "common/json_to_image.h"
#include "common/canvas.h"
#include "common/types.h"
#include "common/logger.h"
#include "common/path_utils.h"
#include "common/stamp_loader.h"
#include "inject/inject.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <json.hpp>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>

namespace vinylizer {

// 从 inject.cpp 迁入（纯 JSON 解析，无平台依赖；Linux 构建不含 inject 模块）
std::vector<nlohmann::json> validate_json(const std::string& path) {
    std::ifstream file = open_ifstream(path);
    if (!file.is_open()) {
        VIN_ERROR("Cannot open JSON file: %s", path.c_str());
        return {};
    }

    nlohmann::json data;
    try {
        data = nlohmann::json::parse(file);
    } catch (const nlohmann::json::parse_error& e) {
        VIN_ERROR("JSON parse error: %s", e.what());
        return {};
    }

    if (!data.is_object() || !data.contains("shapes") || !data["shapes"].is_array()) {
        VIN_ERROR("JSON must contain 'shapes' array");
        return {};
    }

    std::vector<nlohmann::json> valid_shapes;
    for (const auto& shape : data["shapes"]) {
        // type 可能为数字或字符串
        if (shape["type"].is_number_integer()) {
            int type = shape["type"].get<int>();
            if (type != 1 && type != 16 && type != 103 && type != 228) {
                VIN_ERROR("Unsupported shape type: %d", type);
                return {};
            }
        } else if (shape["type"].is_string()) {
            // 字符串 type（外部图章），始终合法
        } else {
            VIN_ERROR("Shape type must be int or string");
            return {};
        }
        valid_shapes.push_back(shape);
    }

    VIN_INFO("JSON validated: %d shapes", static_cast<int>(valid_shapes.size()));
    return valid_shapes;
}

cv::Mat json_to_image(const std::string& json_path) {
    VIN_INFO("json_to_image: start, path=%s", json_path.c_str());
    cudaSetDevice(0);

    try {
        std::ifstream file = open_ifstream(json_path);
        nlohmann::json data = nlohmann::json::parse(file);

        // 读取 canvas 尺寸
        int cw = data.value("canvas_w", 0);
        int ch = data.value("canvas_h", 0);

        // 读取 stamps
        std::vector<std::string> stamp_names;
        std::vector<cv::Mat> stamp_images;
        if (data.contains("stamps") && data["stamps"].is_array()) {
            // 获取 JSON 文件所在目录
            std::string json_dir = json_path;
            size_t last_slash = json_dir.find_last_of("/\\");
            if (last_slash != std::string::npos) json_dir = json_dir.substr(0, last_slash + 1);
            else json_dir = "";

            for (const auto& s : data["stamps"]) {
                std::string name = s.get<std::string>();
                stamp_names.push_back(name);
                // 处理绝对路径（如 F:\...）和相对路径
                std::string stamp_path;
                if (name.size() >= 2 && name[1] == ':')
                    stamp_path = name;  // Windows 绝对路径
                else if (!name.empty() && (name[0] == '/' || name[0] == '\\'))
                    stamp_path = name;  // Unix 绝对路径
                else
                    stamp_path = json_dir + name;
                cv::Mat img = imread_utf8(stamp_path, cv::IMREAD_UNCHANGED);
                if (!img.empty()) {
                    img = resize_stamp_keep_aspect(img, TEMPLATE_SIZE);
                }
                stamp_images.push_back(img);
            }
        }

        // 如果没有 canvas 尺寸，fallback 到 bbox 计算（无 pad）
        std::vector<Shape> shape_vec;
        float ox = 0.0f, oy = 0.0f;
        if (cw <= 0 || ch <= 0) {
            // 旧版 JSON 没有 canvas_w/h，用 bbox fallback
            auto shapes = validate_json(json_path);
            if (shapes.empty()) return cv::Mat();
            float min_x = 1e9f, min_y = 1e9f, max_x = -1e9f, max_y = -1e9f;
            for (auto& s : shapes) {
                auto& d = s["data"];
                float cx = d[0], cy = d[1], rx = d[2], ry = d[3];
                float angle = d.size() >= 5 ? d[4].get<float>() : 0;
                float cos_a = std::cos(angle * M_PI / 180.0f);
                float sin_a = std::sin(angle * M_PI / 180.0f);
                float hx = std::abs(rx * cos_a) + std::abs(ry * sin_a);
                float hy = std::abs(rx * sin_a) + std::abs(ry * cos_a);
                min_x = std::min(min_x, cx - hx);
                min_y = std::min(min_y, cy - hy);
                max_x = std::max(max_x, cx + hx);
                max_y = std::max(max_y, cy + hy);
            }
            cw = std::max(1, static_cast<int>(std::ceil(max_x - min_x)));
            ch = std::max(1, static_cast<int>(std::ceil(max_y - min_y)));
            ox = min_x;
            oy = min_y;
        }

        // 读取 shapes
        for (auto& s : data["shapes"]) {
            Shape shape;
            if (s["type"].is_string()) {
                shape.type = 0;
                shape.stamp_name = s["type"].get<std::string>();
            } else {
                shape.type = s["type"].get<int>();
            }
            auto& d = s["data"];
            auto& c = s["color"];
            float cr = c[0].get<float>();
            float cg = c[1].get<float>();
            float cb = c[2].get<float>();
            float ca = c[3].get<float>();
            shape.cx = d[0].get<float>() - ox;
            shape.cy = d[1].get<float>() - oy;
            shape.rx = d[2];
            shape.ry = d[3];
            shape.angle = d.size() >= 5 ? d[4].get<float>() : 0;
            shape.color = Color(cr, cg, cb, ca);
            shape.opacity = ca / 255.0f;
            // 读取 z（soft depth），用于 preview 渲染时按 z 排序。
            // 兼容旧 JSON：若没有 z 字段，默认 0（保持 index 倒序的等价行为）。
            shape.z = s.value("z", 0.0f);
            shape_vec.push_back(shape);
        }

        Canvas canvas(cw, ch);
        canvas.add_shapes(shape_vec, stamp_names, stamp_images);
        return canvas.clone_image();
    } catch (const std::exception& e) {
        VIN_ERROR("json_to_image exception: %s", e.what());
        return cv::Mat();
    } catch (...) {
        VIN_ERROR("json_to_image unknown exception");
        return cv::Mat();
    }
}

} // namespace vinylizer
