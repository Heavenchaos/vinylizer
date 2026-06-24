#include <QApplication>
#include <QMainWindow>
#include <QPalette>
#include <QIcon>
#include "gui/render_canvas.h"
#include "common/logger.h"

int main(int argc, char* argv[]) {
    vinylizer::Logger::instance().set_level(vinylizer::LogLevel::INFO);

    QApplication app(argc, argv);
    app.setWindowIcon(QIcon(":/icon/icon.png"));

    // Dark theme
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
    app.setPalette(dark);

    QMainWindow window;
    window.setWindowTitle("Vinylizer");
    window.setMinimumSize(1000, 700);
    window.resize(1200, 800);

    auto* canvas = new vinylizer::RenderCanvas();
    window.setCentralWidget(canvas);
    window.show();

    return app.exec();
}
