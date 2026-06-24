#pragma once

#include <cmath>
#include <algorithm>
#include <array>

namespace vinylizer {

constexpr float C1_228 = 0.9132f;
constexpr float C2_228 = 0.1875f;
constexpr float C3_228 = -0.1007f;
constexpr int ALPHA_LUT_SIZE = 4096;

inline float alpha_228_newton(float rn) {
    if (rn >= 1.0f) return 0.0f;
    float a = 1.0f - rn;
    for (int k = 0; k < 5; k++) {
        float fa = 1.0f - C1_228 * a - C2_228 * a * a - C3_228 * a * a * a;
        float fpa = -(C1_228 + 2.0f * C2_228 * a + 3.0f * C3_228 * a * a);
        a = a - (fa - rn) / fpa;
    }
    return std::clamp(a, 0.0f, 1.0f);
}

inline const std::array<float, ALPHA_LUT_SIZE>& alpha_228_lut() {
    static const std::array<float, ALPHA_LUT_SIZE> lut = []() {
        std::array<float, ALPHA_LUT_SIZE> arr;
        for (int i = 0; i < ALPHA_LUT_SIZE; i++) {
            float rn = static_cast<float>(i) / (ALPHA_LUT_SIZE - 1);
            arr[i] = alpha_228_newton(rn);
        }
        return arr;
    }();
    return lut;
}

inline float alpha_228_cpu(float rn) {
    if (rn >= 1.0f) return 0.0f;
    int idx = static_cast<int>(rn * (ALPHA_LUT_SIZE - 1));
    if (idx >= ALPHA_LUT_SIZE) idx = ALPHA_LUT_SIZE - 1;
    return alpha_228_lut()[idx];
}

} // namespace vinylizer
