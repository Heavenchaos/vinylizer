#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <array>
#include <algorithm>

// Template sampling — global scope (used by .cuh device code without namespace)
constexpr int TEMPLATE_SIZE = 128;
constexpr int MAX_TEMPLATE_TYPES = 64;

namespace vinylizer {

// Shape type constants (matching FH6)
constexpr int SHAPE_RECTANGLE        = 1;
constexpr int SHAPE_ROTATED_ELLIPSE  = 16;
constexpr int SHAPE_TRIANGLE        = 103;
constexpr int SHAPE_SOFT_ELLIPSE_228 = 228;

struct Color {
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
    float a = 255.0f;

    Color() = default;
    Color(float r, float g, float b, float a = 255.0f)
        : r(r), g(g), b(b), a(a) {}

    bool is_transparent() const { return a == 0.0f; }

    // BGR order (for OpenCV interop) — 保持 uint8_t 返回用于 OpenCV 互操作
    std::array<uint8_t, 3> to_bgr() const { return {static_cast<uint8_t>(std::clamp(b, 0.0f, 255.0f)), static_cast<uint8_t>(std::clamp(g, 0.0f, 255.0f)), static_cast<uint8_t>(std::clamp(r, 0.0f, 255.0f))}; }
    std::array<uint8_t, 4> to_bgra() const { return {static_cast<uint8_t>(std::clamp(b, 0.0f, 255.0f)), static_cast<uint8_t>(std::clamp(g, 0.0f, 255.0f)), static_cast<uint8_t>(std::clamp(r, 0.0f, 255.0f)), static_cast<uint8_t>(std::clamp(a, 0.0f, 255.0f))}; }
};

struct Shape {
    int   type   = SHAPE_ROTATED_ELLIPSE;
    float cx     = 0.0f;
    float cy     = 0.0f;
    float rx     = 0.0f;
    float ry     = 0.0f;
    float angle  = 0.0f;
    Color color;
    float score   = 0.0f;
    float opacity = 1.0f;
    std::string stamp_name;
    // Soft depth: 可学习深度标量。z 大 = 底层, z 小 = 顶层。
    // 默认 0 表示未启用 soft depth（保持向后兼容）。
    float z = 0.0f;

    bool is_ellipse()          const { return type == SHAPE_ROTATED_ELLIPSE; }
    bool is_soft_ellipse_228() const { return type == SHAPE_SOFT_ELLIPSE_228; }
    bool is_rect()             const { return type == SHAPE_RECTANGLE; }
    bool is_tri()              const { return type == SHAPE_TRIANGLE; }

    static Shape make_ellipse(float cx, float cy, float rx, float ry,
                              float angle, Color color, float opacity = 1.0f) {
        return {SHAPE_ROTATED_ELLIPSE, cx, cy, rx, ry, angle, color, 0.0f, opacity};
    }

    static Shape make_rect(float cx, float cy, float w, float h, Color color) {
        return {SHAPE_RECTANGLE, cx, cy, w, h, 0.0f, color, 0.0f, 1.0f};
    }

    static Shape make_soft_ellipse_228(float cx, float cy, float rx, float ry,
                                       float angle, Color color, float opacity = 1.0f) {
        return {SHAPE_SOFT_ELLIPSE_228, cx, cy, rx, ry, angle, color, 0.0f, opacity};
    }

    static Shape make_triangle(float cx, float cy, float half_base, float height,
                               float angle, Color color, float opacity = 1.0f) {
        return {SHAPE_TRIANGLE, cx, cy, half_base, height, angle, color, 0.0f, opacity};
    }
};

struct Layer {
    std::string name;
    std::vector<Shape> shapes;
    int z_start = 0;
};

} // namespace vinylizer
