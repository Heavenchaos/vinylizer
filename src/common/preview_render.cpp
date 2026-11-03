#include "common/preview_render.h"
#include "common/json_to_image.h"

namespace vinylizer {

QImage render_preview_from_json(const std::string& json_path) {
    cv::Mat rgb = json_to_image(json_path);
    if (rgb.empty()) return QImage();

    // json_to_image 返回的 cv::Mat 字节布局是 [R,G,B]，
    // QImage::Format_RGB888 按 [R,G,B] 读字节 → 天然正确
    QImage img(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
    return img.copy();
}

} // namespace vinylizer
