#include "common/canvas.h"
#include "cuda/canvas_render.h"

namespace vinylizer {

Canvas::Canvas(int w, int h) {
    init(w, h);
}

void Canvas::init(int w, int h, const cv::Mat& mask) {
    w_ = w;
    h_ = h;
    image_ = cv::Mat::zeros(h, w, CV_8UC3);
    mask_ = mask;
}

void Canvas::add_shapes(const std::vector<Shape>& shapes) {
    if (shapes.empty()) {
        clear();
        return;
    }
    image_ = canvas_render_gpu(shapes, w_, h_);
}

void Canvas::clear() {
    image_ = cv::Mat::zeros(h_, w_, CV_8UC3);
}

} // namespace vinylizer
