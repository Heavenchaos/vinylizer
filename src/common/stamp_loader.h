#pragma once

// ============================================================
// stamp_loader.h — 外部图章图片加载工具
//
// 提供两个纯函数，实现 pydiffbmp 的图章录入管线：
//   1. resize_stamp_keep_aspect — 强制 RGBA、保持宽高比、
//      LANCZOS4 插值、补零到正方形
//   2. extract_stamp_alpha      — 背景去除 + 灰度反相，
//      输出 float alpha [0,1]
//
// 所有调用点已包含 OpenCV，无额外依赖。
// ============================================================

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vector>
#include <algorithm>
#include <cstdint>

namespace vinylizer {

// ============================================================
// resize_stamp_keep_aspect
//
// 将输入图片强制转为 RGBA，保持宽高比缩放到 output_size
// 的最大边，居中补零到 output_size × output_size 正方形。
//
// 返回 CV_8UC4 的正方形 RGBA cv::Mat。
// ============================================================
inline cv::Mat resize_stamp_keep_aspect(const cv::Mat& src, int output_size = 256) {
    // 步骤1: 强制 RGBA
    cv::Mat rgba;
    if (src.channels() == 4) {
        rgba = src.clone();
    } else if (src.channels() == 3) {
        cv::cvtColor(src, rgba, cv::COLOR_BGR2BGRA);
    } else if (src.channels() == 1) {
        cv::cvtColor(src, rgba, cv::COLOR_GRAY2BGRA);
    } else {
        return cv::Mat();  // 不支持的通道数
    }

    // 步骤2: 保持宽高比，LANCZOS4 缩放
    int w = rgba.cols;
    int h = rgba.rows;
    float scale = (float)output_size / (float)std::max(w, h);
    int new_w = (int)(w * scale);
    int new_h = (int)(h * scale);
    if (new_w < 1) new_w = 1;
    if (new_h < 1) new_h = 1;

    cv::Mat resized;
    cv::resize(rgba, resized, {new_w, new_h}, 0, 0, cv::INTER_LANCZOS4);

    // 步骤3: 居中补零到正方形
    cv::Mat padded(output_size, output_size, CV_8UC4, cv::Scalar(0, 0, 0, 0));
    int ox = (output_size - new_w) / 2;
    int oy = (output_size - new_h) / 2;
    resized.copyTo(padded(cv::Rect(ox, oy, new_w, new_h)));

    return padded;
}

// ============================================================
// extract_stamp_alpha
//
// 从已缩放的正方形 RGBA cv::Mat 中提取 float alpha [0,1]，
// 使用 pydiffbmp 的逻辑：
//
//   1. 从 RGB 通道计算灰度
//   2. 背景去除：gray > bg_threshold 的像素视为背景
//   3. 与原始 alpha 合并：alpha_channel = min(alpha_from_bg, orig_alpha)
//   4. 最终 alpha = (1 - gray/255) * (alpha_channel > 0)
//
// 输入必须为 CV_8UC4、正方形。
// 输出为行主序的 float 向量，长度 output_size^2。
// ============================================================
inline std::vector<float> extract_stamp_alpha(const cv::Mat& rgba, int bg_threshold = 250) {
    const int S = rgba.rows;  // 必须为正方形
    const int S2 = S * S;

    std::vector<float> result(S2, 0.0f);

    if (rgba.empty() || rgba.channels() != 4 || rgba.rows != rgba.cols) {
        return result;
    }

    // 步骤1: 从 RGB 提取灰度
    cv::Mat gray;
    cv::cvtColor(rgba, gray, cv::COLOR_BGRA2GRAY);  // 丢弃 alpha，仅用 RGB→灰度

    // 步骤2-4: 逐像素 alpha 提取
    for (int y = 0; y < S; y++) {
        for (int x = 0; x < S; x++) {
            uint8_t g = gray.at<uint8_t>(y, x);
            uint8_t orig_a = rgba.at<cv::Vec4b>(y, x)[3];

            // 背景去除：灰度 > 阈值 → 背景
            uint8_t alpha_from_bg = (g <= bg_threshold) ? 255 : 0;

            // 合并：取背景去除和原始 alpha 的交集
            uint8_t alpha_channel = (std::min)(alpha_from_bg, orig_a);

            // 最终 alpha = (1 - gray/255) * (前景区域)
            if (alpha_channel > 0) {
                result[y * S + x] = 1.0f - (float)g / 255.0f;
            } else {
                result[y * S + x] = 0.0f;
            }
        }
    }

    return result;
}

} // namespace vinylizer
