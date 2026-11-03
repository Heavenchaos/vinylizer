#pragma once

#include <QImage>
#include <string>

namespace vinylizer {

QImage render_preview_from_json(const std::string& json_path);

} // namespace vinylizer
