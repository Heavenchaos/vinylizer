#include "preprocess/preprocessor.h"
#include "common/logger.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/core/utility.hpp>

#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <set>
#include <functional>

namespace vinylizer {

std::tuple<int, int, int> compute_bg_color_from_result(
    const cv::Mat& posterized, const cv::Mat& opaque_mask,
    const std::vector<std::tuple<int, int, int>>& palette,
    const cv::Mat& rgb_image)
{
    if (!posterized.empty() && !palette.empty()) {
        std::unordered_map<int, int> counts;
        for (int y = 0; y < posterized.rows; ++y) {
            const auto* post_row = posterized.ptr<uint8_t>(y);
            const auto* mask_row = opaque_mask.ptr<uint8_t>(y);
            for (int x = 0; x < posterized.cols; ++x) {
                if (mask_row[x] > 0) {
                    counts[post_row[x]]++;
                }
            }
        }

        int dominant_label = -1, max_count = 0;
        for (auto& [label, cnt] : counts) {
            if (cnt > max_count) {
                max_count = cnt;
                dominant_label = label;
            }
        }
        if (dominant_label >= 0 && dominant_label < static_cast<int>(palette.size())) {
            return palette[dominant_label];
        }
    }

    cv::Scalar mean = cv::mean(rgb_image, opaque_mask);
    return {
        static_cast<int>(mean[0]), static_cast<int>(mean[1]), static_cast<int>(mean[2])
    };
}

// ── extract_alpha ────────────────────────────────────────────
void Preprocessor::extract_alpha(const cv::Mat& input, cv::Mat& alpha,
                                  cv::Mat& rgb, cv::Mat& opaque) {
    if (input.channels() == 4) {
        cv::Mat bgr;
        cv::extractChannel(input, alpha, 3);
        cv::cvtColor(input, bgr, cv::COLOR_BGRA2BGR);  // drop alpha
        cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
        cv::threshold(alpha, opaque, 127, 255, cv::THRESH_BINARY);
    } else {
        cv::cvtColor(input, rgb, cv::COLOR_BGR2RGB);
        opaque = cv::Mat(input.rows, input.cols, CV_8UC1, cv::Scalar(255));
        alpha  = cv::Mat(input.rows, input.cols, CV_8UC1, cv::Scalar(255));
    }
}

// ── determine_k ──────────────────────────────────────────────
int Preprocessor::determine_k(const cv::Mat& rgb, const cv::Mat& mask) {
    // Count unique colors in masked region
    std::unordered_set<int> unique_colors;
    for (int y = 0; y < rgb.rows; ++y) {
        const auto* rgb_row = rgb.ptr<cv::Vec3b>(y);
        const auto* mask_row = mask.ptr<uint8_t>(y);
        for (int x = 0; x < rgb.cols; ++x) {
            if (mask_row[x] > 0) {
                const auto& c = rgb_row[x];
                unique_colors.insert((c[0] << 16) | (c[1] << 8) | c[2]);
            }
        }
    }
    int count = static_cast<int>(unique_colors.size());
    int K = 16 + count / 2000;
    return std::min(K, 64);
}

// ── quantize_colors ──────────────────────────────────────────
void Preprocessor::quantize_colors(const cv::Mat& rgb, const cv::Mat& mask,
                                    int K,
                                    cv::Mat& posterized,
                                    std::vector<std::tuple<int, int, int>>& palette) {
    int h = rgb.rows, w = rgb.cols;
    posterized = cv::Mat::zeros(h, w, CV_8UC1);
    palette.clear();

    // Collect masked pixels
    cv::Mat pixels;
    for (int y = 0; y < h; ++y) {
        const auto* rgb_row = rgb.ptr<cv::Vec3b>(y);
        const auto* mask_row = mask.ptr<uint8_t>(y);
        for (int x = 0; x < w; ++x) {
            if (mask_row[x] > 0) {
                cv::Mat row = (cv::Mat_<float>(1, 3) <<
                    rgb_row[x][0], rgb_row[x][1], rgb_row[x][2]);
                pixels.push_back(row);
            }
        }
    }

    if (pixels.empty()) return;

    K = std::min(K, pixels.rows);

    cv::Mat labels, centers;
    cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 10, 1.0);
    cv::kmeans(pixels, K, labels, criteria, 3, cv::KMEANS_PP_CENTERS, centers);

    // Build palette
    for (int i = 0; i < centers.rows; ++i) {
        palette.emplace_back(
            static_cast<int>(centers.at<float>(i, 0)),
            static_cast<int>(centers.at<float>(i, 1)),
            static_cast<int>(centers.at<float>(i, 2))
        );
    }

    // Fill posterized label map
    int idx = 0;
    for (int y = 0; y < h; ++y) {
        auto* post_row = posterized.ptr<uint8_t>(y);
        const auto* mask_row = mask.ptr<uint8_t>(y);
        for (int x = 0; x < w; ++x) {
            if (mask_row[x] > 0) {
                post_row[x] = static_cast<uint8_t>(labels.at<int>(idx));
                ++idx;
            }
        }
    }
}

// ── merge_similar_hues ───────────────────────────────────────
void Preprocessor::merge_similar_hues(
    cv::Mat& posterized,
    std::vector<std::tuple<int, int, int>>& palette,
    const cv::Mat& rgb, const cv::Mat& mask) {

    int n_colors = static_cast<int>(palette.size());
    if (n_colors <= 4) return;

    // Convert palette to HSV
    struct HSVEntry { int h, s, v; };
    std::vector<HSVEntry> hsv_palette(n_colors);
    for (int i = 0; i < n_colors; ++i) {
        auto [r, g, b] = palette[i];
        cv::Mat rgb_pixel(1, 1, CV_8UC3, cv::Scalar(r, g, b));
        cv::Mat hsv_pixel;
        cv::cvtColor(rgb_pixel, hsv_pixel, cv::COLOR_RGB2HSV);
        const auto& hsv = hsv_pixel.at<cv::Vec3b>(0, 0);
        hsv_palette[i] = {hsv[0], hsv[1], hsv[2]};
    }

    // Count pixels per label
    std::vector<int64_t> pixel_counts(n_colors, 0);
    for (int y = 0; y < posterized.rows; ++y) {
        const auto* post_row = posterized.ptr<uint8_t>(y);
        const auto* mask_row = mask.ptr<uint8_t>(y);
        for (int x = 0; x < posterized.cols; ++x) {
            if (mask_row[x] > 0) {
                pixel_counts[post_row[x]]++;
            }
        }
    }

    // Union-Find
    std::vector<int> parent(n_colors);
    for (int i = 0; i < n_colors; ++i) parent[i] = i;

    std::function<int(int)> find = [&](int x) -> int {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };

    auto unite = [&](int a, int b) {
        int ra = find(a), rb = find(b);
        if (ra != rb) {
            if (pixel_counts[ra] < pixel_counts[rb]) std::swap(ra, rb);
            parent[rb] = ra;
        }
    };

    const int hue_threshold = 20;
    const int sat_threshold = 60;
    const int val_threshold = 60;

    for (int i = 0; i < n_colors; ++i) {
        if (pixel_counts[i] == 0) continue;
        auto [hi, si, vi] = hsv_palette[i];
        if (si < 30) continue;

        for (int j = i + 1; j < n_colors; ++j) {
            if (pixel_counts[j] == 0) continue;
            auto [hj, sj, vj] = hsv_palette[j];
            if (sj < 30) continue;

            int dh = std::abs(hi - hj);
            if (dh > 90) dh = 180 - dh;
            int ds = std::abs(si - sj);
            int dv = std::abs(vi - vj);

            if (dh < hue_threshold && ds < sat_threshold && dv < val_threshold) {
                unite(i, j);
            }
        }
    }

    // Group and merge
    std::unordered_map<int, std::vector<int>> groups;
    for (int i = 0; i < n_colors; ++i) {
        groups[find(i)].push_back(i);
    }

    auto new_palette = palette;
    int n_merged = 0, n_total_merged = 0;

    for (auto& [root, members] : groups) {
        if (members.size() <= 1) continue;
        n_merged++;
        n_total_merged += static_cast<int>(members.size());

        int64_t total_weight = 0;
        double avg_r = 0, avg_g = 0, avg_b = 0;
        for (int m : members) {
            total_weight += pixel_counts[m];
        }
        if (total_weight == 0) continue;

        for (int m : members) {
            auto [r, g, b] = palette[m];
            double w = static_cast<double>(pixel_counts[m]);
            avg_r += r * w;
            avg_g += g * w;
            avg_b += b * w;
        }
        avg_r /= total_weight;
        avg_g /= total_weight;
        avg_b /= total_weight;

        auto merged_color = std::make_tuple(
            std::clamp(static_cast<int>(std::lround(avg_r)), 0, 255),
            std::clamp(static_cast<int>(std::lround(avg_g)), 0, 255),
            std::clamp(static_cast<int>(std::lround(avg_b)), 0, 255)
        );

        for (int m : members) {
            new_palette[m] = merged_color;
        }
    }

    // Count unique colors after merge
    std::set<std::tuple<int,int,int>> unique_set(new_palette.begin(), new_palette.end());
    int n_unique = static_cast<int>(unique_set.size());

    VIN_INFO("Hue merge: %d groups, %d colors merged, unique: %d -> %d",
             n_merged, n_total_merged, n_colors, n_unique);

    palette = new_palette;
}

// ── detect_edges ─────────────────────────────────────────────
cv::Mat Preprocessor::detect_edges(const cv::Mat& rgb) {
    cv::Mat gray, blur, edges;
    cv::cvtColor(rgb, gray, cv::COLOR_RGB2GRAY);
    cv::GaussianBlur(gray, blur, cv::Size(3, 3), 0);
    cv::Canny(blur, edges, 50, 150);
    return edges;
}

// ── compute_saliency ─────────────────────────────────────────
cv::Mat Preprocessor::compute_saliency(const cv::Mat& rgb) {
    cv::Mat saliency(rgb.rows, rgb.cols, CV_32FC1, cv::Scalar(1.0f));
    cv::Mat gray;
    cv::cvtColor(rgb, gray, cv::COLOR_RGB2GRAY);

    cv::CascadeClassifier face_cascade, eye_cascade;
    std::string face_path = cv::samples::findFile("haarcascade_frontalface_default.xml", false, false);
    std::string eye_path  = cv::samples::findFile("haarcascade_eye.xml", false, false);

    if (!face_path.empty() && face_cascade.load(face_path)) {
        std::vector<cv::Rect> faces;
        face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(30, 30));
        for (const auto& f : faces) {
            saliency(f).setTo(3.0f);
        }
    }

    if (!eye_path.empty() && eye_cascade.load(eye_path)) {
        std::vector<cv::Rect> eyes;
        eye_cascade.detectMultiScale(gray, eyes, 1.1, 3, 0, cv::Size(20, 20));
        for (const auto& e : eyes) {
            saliency(e).setTo(10.0f);
        }
    }

    return saliency;
}

// ── extract_dark_pixels ──────────────────────────────────────
cv::Mat Preprocessor::extract_dark_pixels(const cv::Mat& rgb) {
    cv::Mat gray, dark;
    cv::cvtColor(rgb, gray, cv::COLOR_RGB2GRAY);
    cv::threshold(gray, dark, 50, 255, cv::THRESH_BINARY_INV);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(dark, dark, cv::MORPH_OPEN, kernel);
    return dark;
}

// ── process (main entry) ─────────────────────────────────────
PreprocessResult Preprocessor::process(const cv::Mat& input_img) {
    VIN_INFO("Preprocessing...");
    PreprocessResult result;

    extract_alpha(input_img, result.alpha_mask, result.rgb_image, result.opaque_mask);

    int K = kmeans_k_;
    if (K == 0) {
        K = determine_k(result.rgb_image, result.opaque_mask);
    }
    result.kmeans_k = K;
    VIN_INFO("K-Means K = %d", K);

    quantize_colors(result.rgb_image, result.opaque_mask, K,
                    result.posterized, result.palette);
    VIN_INFO("Palette has %d colors", static_cast<int>(result.palette.size()));

    merge_similar_hues(result.posterized, result.palette,
                       result.rgb_image, result.opaque_mask);

    result.edge_map = detect_edges(result.rgb_image);

    cv::Mat gray;
    cv::cvtColor(result.rgb_image, gray, cv::COLOR_RGB2GRAY);
    cv::Mat grad_x, grad_y;
    cv::Sobel(gray, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(gray, grad_y, CV_32F, 0, 1, 3);
    cv::phase(grad_x, grad_y, result.grad_dir, true);

    result.saliency_map = compute_saliency(result.rgb_image);
    result.dark_mask = extract_dark_pixels(result.rgb_image);

    VIN_INFO("Preprocessing done.");
    return result;
}

} // namespace vinylizer
