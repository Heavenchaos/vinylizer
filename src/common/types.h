#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <array>

namespace vinylizer {

// Shape type constants (matching FH6)
constexpr int SHAPE_RECTANGLE        = 1;
constexpr int SHAPE_ROTATED_ELLIPSE  = 16;
constexpr int SHAPE_SOFT_ELLIPSE_228 = 228;

struct Color {
    uint8_t r = 0;
    uint8_t g = 0;
    uint8_t b = 0;
    uint8_t a = 255;

    Color() = default;
    Color(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255)
        : r(r), g(g), b(b), a(a) {}

    bool is_transparent() const { return a == 0; }

    // BGR order (for OpenCV interop)
    std::array<uint8_t, 3> to_bgr() const { return {b, g, r}; }
    std::array<uint8_t, 4> to_bgra() const { return {b, g, r, a}; }
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

    bool is_ellipse()          const { return type == SHAPE_ROTATED_ELLIPSE; }
    bool is_soft_ellipse_228() const { return type == SHAPE_SOFT_ELLIPSE_228; }
    bool is_rect()             const { return type == SHAPE_RECTANGLE; }

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
};

struct Layer {
    std::string name;
    std::vector<Shape> shapes;
    int z_start = 0;
};

} // namespace vinylizer
