#pragma once

#include "common/types.h"
#include <opencv2/core.hpp>
#include <vector>

namespace vinylizer {

class Canvas {
public:
    Canvas() = default;
    Canvas(int w, int h);

    void init(int w, int h, const cv::Mat& mask = {});

    int width()  const { return w_; }
    int height() const { return h_; }
    cv::Mat clone_image() const { return image_.clone(); }

    void add_shapes(const std::vector<Shape>& shapes);
    void clear();

private:
    int w_ = 0;
    int h_ = 0;
    cv::Mat image_;
    cv::Mat mask_;
};

} // namespace vinylizer
