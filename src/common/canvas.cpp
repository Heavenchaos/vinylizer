#include "common/canvas.h"
#include "cuda/canvas_render.h"

namespace vinylizer {

Canvas::Canvas(int w, int h) {
    init(w, h);
}

void Canvas::init(int w, int h, const cv::Mat& mask) {
    w_ = w;
    h_ = h;
    image_ = cv::Mat(h, w, CV_8UC3, cv::Scalar(255, 255, 255));
    mask_ = mask;
}

void Canvas::add_shapes(const std::vector<Shape>& shapes,
                        const std::vector<std::string>& stamp_names,
                        const std::vector<cv::Mat>& stamp_images) {
    if (shapes.empty()) {
        clear();
        return;
    }
    image_ = canvas_render_gpu(shapes, w_, h_, stamp_names, stamp_images);
}

void Canvas::clear() {
    image_ = cv::Mat(h_, w_, CV_8UC3, cv::Scalar(255, 255, 255));
}

} // namespace vinylizer
