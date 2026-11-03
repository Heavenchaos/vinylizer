#include <QApplication>
#include <QPalette>
#include <QPainter>
#include <QPixmap>
#include <QIcon>
#include <QImage>
#include <cuda_runtime.h>
#include <cstdlib>
#include <string>

#include "gui/main_window.h"
#include "gui/optimize_page.h"
#include "gui/optimize_params.h"
#include "common/logger.h"
#include "core/pipeline.h"

namespace {

// 窗口图标（无外部资源文件，运行时画一个）
QIcon make_icon() {
    QPixmap pm(64, 64);
    pm.fill(Qt::transparent);
    QPainter p(&pm);
    p.setRenderHint(QPainter::Antialiasing);
    p.setPen(Qt::NoPen);
    p.setBrush(QColor("#4a9eff"));
    p.drawEllipse(2, 2, 60, 60);
    p.setBrush(QColor("#0f0f23"));
    p.drawEllipse(21, 21, 22, 22);
    p.end();
    return QIcon(pm);
}

void apply_dark_theme(QApplication& app) {
    app.setStyle("Fusion");
    QPalette dark;
    dark.setColor(QPalette::Window, QColor("#0f0f23"));
    dark.setColor(QPalette::WindowText, QColor("#ffffff"));
    dark.setColor(QPalette::Base, QColor("#16213e"));
    dark.setColor(QPalette::AlternateBase, QColor("#1a1a2e"));
    dark.setColor(QPalette::Text, QColor("#ffffff"));
    dark.setColor(QPalette::Button, QColor("#16213e"));
    dark.setColor(QPalette::ButtonText, QColor("#ffffff"));
    dark.setColor(QPalette::Highlight, QColor("#4a9eff"));
    dark.setColor(QPalette::HighlightedText, QColor("#ffffff"));
    dark.setColor(QPalette::ToolTipBase, QColor("#16213e"));
    dark.setColor(QPalette::ToolTipText, QColor("#ffffff"));
    dark.setColor(QPalette::Disabled, QPalette::Text, QColor("#555555"));
    dark.setColor(QPalette::Disabled, QPalette::ButtonText, QColor("#555555"));
    app.setPalette(dark);
}

// 隐藏自测入口：--autotest <img> <N> <out.json> [total_iters]
// 走与界面完全相同的 PipelineConfig 构造路径（OptimizeParams + build_pipeline_config），
// 用于验证「界面参数 → 引擎」的传递与 CLI 同参数输出逐位一致。
// [total_iters] 省略时用界面默认值（= 界面上「高级选项 → 总迭代数」的初值）。
int run_autotest(const QString& img, int n, const QString& out, int total_iters) {
    QImage image(img);
    if (image.isNull()) {
        VIN_ERROR("autotest: cannot load image %s", img.toUtf8().constData());
        return 2;
    }
    vinylizer::OptimizeParams params = vinylizer::OptimizePage::default_params();
    params.input_path = img.toStdString();
    params.num_shapes = n;
    params.canvas_w = image.width();
    params.canvas_h = image.height();
    if (total_iters > 0) params.total_iters = total_iters;

    vinylizer::PipelineConfig cfg =
        vinylizer::build_pipeline_config(params, out.toStdString());
    VIN_INFO("autotest: img=%s N=%d out=%s total_iters=%d init_mode=%s darts=%d",
             params.input_path.c_str(), n, out.toUtf8().constData(),
             params.total_iters, cfg.init_mode.c_str(), cfg.enable_darts ? 1 : 0);
    vinylizer::run_pipeline(cfg);
    return 0;
}

} // namespace

int main(int argc, char* argv[]) {
    vinylizer::AsyncLogger::instance().set_level(vinylizer::LogLevel::INFO);

    // 显式初始化 CUDA primary context（主线程持有引用，整个进程生命周期有效）
    cudaSetDevice(0);
    cudaFree(0);

    if (argc >= 5 && std::string(argv[1]) == "--autotest") {
        QApplication app(argc, argv);
        const int iters = (argc >= 6) ? std::atoi(argv[5]) : 0;
        return run_autotest(QString::fromLocal8Bit(argv[2]),
                            std::atoi(argv[3]),
                            QString::fromLocal8Bit(argv[4]),
                            iters);
    }

    QApplication app(argc, argv);
    apply_dark_theme(app);
    app.setWindowIcon(make_icon());

    vinylizer::MainWindow window;
    window.show();

    return app.exec();
}
