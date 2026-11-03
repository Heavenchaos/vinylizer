#pragma once

#include <opencv2/core.hpp>
#include <string>

namespace vinylizer {

// 从 JSON 文件渲染预览图，返回字节布局为 RGB 的 cv::Mat
// 内部使用 GPU 加速（canvas_render_gpu），性能等同于原地渲染
//
// 返回的 cv::Mat：
//   - 字节顺序: [R,G,B]（与 Shape::Color 的语义一致）
//   - 不可直接 cv::imwrite（OpenCV 要求 BGR），需用 cvtColor(RGB2BGR)
//   - 可直接用于 QImage(Format_RGB888)
cv::Mat json_to_image(const std::string& json_path);

} // namespace vinylizer
