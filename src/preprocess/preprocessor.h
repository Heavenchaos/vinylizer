#pragma once

#include "common/types.h"
#include <opencv2/core.hpp>
#include <vector>
#include <tuple>

namespace vinylizer {

// Compute dominant background color from preprocessed result
std::tuple<int, int, int> compute_bg_color_from_result(const cv::Mat& posterized,
    const cv::Mat& opaque_mask,
    const std::vector<std::tuple<int, int, int>>& palette,
    const cv::Mat& rgb_image);

struct PreprocessResult {
    cv::Mat rgb_image;
    cv::Mat alpha_mask;
    cv::Mat opaque_mask;
    cv::Mat posterized;       // label map (uint8)
    cv::Mat edge_map;
    cv::Mat grad_dir;         // gradient direction in degrees (float32)
    cv::Mat saliency_map;     // float32
    cv::Mat dark_mask;
    std::vector<std::tuple<int, int, int>> palette;  // RGB colors
    int kmeans_k = 0;
};

class Preprocessor {
public:
    void set_kmeans_k(int k) { kmeans_k_ = k; }

    PreprocessResult process(const cv::Mat& input_img);

private:
    int kmeans_k_ = 0;

    void extract_alpha(const cv::Mat& input, cv::Mat& alpha,
                       cv::Mat& rgb, cv::Mat& opaque);
    int determine_k(const cv::Mat& rgb, const cv::Mat& mask);
    void quantize_colors(const cv::Mat& rgb, const cv::Mat& mask, int K,
                         cv::Mat& posterized,
                         std::vector<std::tuple<int, int, int>>& palette);
    void merge_similar_hues(cv::Mat& posterized,
                            std::vector<std::tuple<int, int, int>>& palette,
                            const cv::Mat& rgb, const cv::Mat& mask);
    cv::Mat detect_edges(const cv::Mat& rgb);
    cv::Mat compute_saliency(const cv::Mat& rgb);
    cv::Mat extract_dark_pixels(const cv::Mat& rgb);
};

} // namespace vinylizer
