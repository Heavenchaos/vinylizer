#include "gui/main_window.h"
#include "gui/optimize_page.h"
#include "gui/inject_page.h"

#include <QWidget>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QStackedWidget>

namespace vinylizer {

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    setWindowTitle("Vinylizer");
    setMinimumSize(1000, 700);
    resize(1200, 800);

    auto* central = new QWidget;
    auto* root = new QHBoxLayout(central);
    root->setContentsMargins(0, 0, 0, 0);
    root->setSpacing(0);

    // ── 侧边导航 ──
    auto* sidebar = new QWidget;
    sidebar->setFixedWidth(168);
    sidebar->setStyleSheet("background:#1a1a2e;");
    auto* sv = new QVBoxLayout(sidebar);
    sv->setContentsMargins(0, 16, 0, 16);
    sv->setSpacing(4);

    auto* title = new QLabel("Vinylizer");
    title->setAlignment(Qt::AlignCenter);
    title->setStyleSheet("color:#ffffff; font-size:18px; font-weight:bold; padding:8px;");
    sv->addWidget(title);
    sv->addSpacing(16);

    const QString nav_style =
        "QPushButton { text-align:left; padding:10px 16px; border:none; background:transparent;"
        " color:#a0a0b0; font-size:14px; }"
        "QPushButton:hover { color:#ffffff; }"
        "QPushButton:checked { color:#ffffff; background:#16213e; border-left:3px solid #4a9eff; }";

    nav_optimize_ = new QPushButton(QString::fromUtf8("  矢量化"));
    nav_inject_ = new QPushButton(QString::fromUtf8("  FH6 注入"));
    nav_optimize_->setCheckable(true);
    nav_inject_->setCheckable(true);
    nav_optimize_->setChecked(true);
    nav_optimize_->setStyleSheet(nav_style);
    nav_inject_->setStyleSheet(nav_style);
    nav_optimize_->setMinimumHeight(40);
    nav_inject_->setMinimumHeight(40);
    sv->addWidget(nav_optimize_);
    sv->addWidget(nav_inject_);
    sv->addStretch();
    root->addWidget(sidebar);

    // ── 页面 ──
    stack_ = new QStackedWidget;
    optimize_page_ = new OptimizePage;
    inject_page_ = new InjectPage;
    stack_->addWidget(optimize_page_);  // 0
    stack_->addWidget(inject_page_);    // 1
    root->addWidget(stack_, 1);

    setCentralWidget(central);

    connect(nav_optimize_, &QPushButton::clicked, this, [this]() { switch_page(0); });
    connect(nav_inject_, &QPushButton::clicked, this, [this]() { switch_page(1); });
    connect(optimize_page_, &OptimizePage::goto_inject_page,
            this, &MainWindow::goto_inject_with);
}

void MainWindow::switch_page(int index) {
    stack_->setCurrentIndex(index);
    nav_optimize_->setChecked(index == 0);
    nav_inject_->setChecked(index == 1);
}

void MainWindow::goto_inject_with(const QString& json_path) {
    inject_page_->set_json_path(json_path);
    switch_page(1);
}

} // namespace vinylizer
