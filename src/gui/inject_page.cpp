#include "gui/inject_page.h"
#include "gui/inject_worker.h"
#include "common/preview_render.h"
#include "common/logger.h"
#include "inject/inject.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QSpinBox>
#include <QPlainTextEdit>
#include <QScrollBar>
#include <QFileDialog>
#include <QFileInfo>
#include <QMessageBox>
#include <QApplication>
#include <QtConcurrent>

namespace vinylizer {

InjectPage::InjectPage(QWidget* parent) : QWidget(parent) {
    auto* root = new QHBoxLayout(this);
    root->setContentsMargins(16, 16, 16, 16);
    root->setSpacing(16);

    // ── 左：文件 / 图层数 / 预览 / 注入 ──
    auto* left = new QVBoxLayout;
    json_btn_ = new QPushButton(QString::fromUtf8("选择 JSON 文件"));
    json_btn_->setMinimumHeight(34);
    left->addWidget(json_btn_);

    json_label_ = new QLabel(QString::fromUtf8("未选择文件"));
    json_label_->setStyleSheet("color:#a0a0b0;");
    json_label_->setWordWrap(true);
    left->addWidget(json_label_);

    auto* lrow = new QHBoxLayout;
    lrow->addWidget(new QLabel(QString::fromUtf8("占位图层数")));
    layer_spin_ = new QSpinBox;
    layer_spin_->setRange(1, 100000);
    layer_spin_->setValue(2000);
    layer_spin_->setMinimumWidth(120);
    lrow->addWidget(layer_spin_);
    lrow->addStretch();
    left->addLayout(lrow);

    auto* hint1 = new QLabel(QString::fromUtf8(
        "请先在 FH6 涂装编辑器里摆好这么多占位图形，注入时需要 FH6 正在运行。"));
    hint1->setWordWrap(true);
    hint1->setStyleSheet("color:#a0a0b0;");
    left->addWidget(hint1);

    preview_label_ = new QLabel(QString::fromUtf8("选择 JSON 后显示预览"));
    preview_label_->setAlignment(Qt::AlignCenter);
    preview_label_->setMinimumHeight(200);
    preview_label_->setStyleSheet("background:#0d0d1a; color:#a0a0b0; border:1px solid #333355;");
    left->addWidget(preview_label_, 1);

    inject_btn_ = new QPushButton(QString::fromUtf8("注入 FH6"));
    inject_btn_->setMinimumHeight(38);
    left->addWidget(inject_btn_);

    auto* disc = new QLabel(QString::fromUtf8(
        "免责说明：注入依赖的游戏内存偏移常量可能随游戏版本更新失效，"
        "若注入失败请等待适配更新。"));
    disc->setWordWrap(true);
    disc->setStyleSheet("color:#ff9f43;");
    left->addWidget(disc);

    auto* left_wrap = new QWidget;
    left_wrap->setLayout(left);
    left_wrap->setMinimumWidth(320);
    left_wrap->setMaximumWidth(460);
    root->addWidget(left_wrap);

    // ── 右：日志 ──
    auto* right = new QVBoxLayout;
    auto* lhead = new QHBoxLayout;
    lhead->addWidget(new QLabel(QString::fromUtf8("日志")));
    lhead->addStretch();
    clear_btn_ = new QPushButton(QString::fromUtf8("清空日志"));
    lhead->addWidget(clear_btn_);
    right->addLayout(lhead);

    log_view_ = new QPlainTextEdit;
    log_view_->setReadOnly(true);
    log_view_->setStyleSheet("background:#0d0d1a; color:#c8c8d8; border:1px solid #333355;");
    log_view_->setMaximumBlockCount(2000);
    right->addWidget(log_view_, 1);
    root->addLayout(right, 1);

    connect(json_btn_, &QPushButton::clicked, this, &InjectPage::choose_json);
    connect(inject_btn_, &QPushButton::clicked, this, &InjectPage::start_inject);
    connect(clear_btn_, &QPushButton::clicked, this, &InjectPage::clear_log);
    connect(&preview_watcher_, &QFutureWatcher<QImage>::finished,
            this, &InjectPage::on_preview_ready);

    inject_btn_->setEnabled(false);
}

InjectPage::~InjectPage() {
    preview_watcher_.waitForFinished();
    if (worker_) {
        worker_->confirm(false);
        worker_->wait();
    }
}

void InjectPage::set_json_path(const QString& path) {
    if (path.isEmpty()) return;
    json_path_ = path;
    json_label_->setText(QString::fromUtf8("当前文件：") + QFileInfo(path).fileName());
    inject_btn_->setEnabled(!running_);

    preview_image_ = QImage();
    preview_label_->setText(QString::fromUtf8("正在渲染预览…"));
    const std::string jp = path.toStdString();
    preview_watcher_.setFuture(
        QtConcurrent::run([jp]() { return render_preview_from_json(jp); }));
}

void InjectPage::choose_json() {
    const QString path = QFileDialog::getOpenFileName(this, QString::fromUtf8("选择 shapes.json"),
                                                      QString(), "JSON (*.json)");
    if (path.isEmpty()) return;
    set_json_path(path);
}

void InjectPage::clear_log() { log_view_->clear(); }

void InjectPage::start_inject() {
    if (running_ || json_path_.isEmpty()) return;

    running_ = true;
    inject_btn_->setEnabled(false);
    log_view_->clear();
    on_log(QString::fromUtf8("[*] 开始注入"), false);

    InjectConfig cfg;
    cfg.json_path = json_path_.toStdString();
    cfg.layer_count = layer_spin_->value();
    cfg.process_id = 0;  // auto-detect

    worker_ = new InjectWorker(cfg, this);
    connect(worker_, &InjectWorker::log_message, this, &InjectPage::on_log);
    connect(worker_, &InjectWorker::confirm_requested, this, &InjectPage::on_confirm_requested);
    connect(worker_, &InjectWorker::done, this, &InjectPage::on_finished);
    worker_->start();
}

void InjectPage::on_log(QString msg, bool is_error) {
    if (is_error) {
        log_view_->appendHtml(QString("<span style='color:#e74c3c;'>%1</span>")
                                  .arg(msg.toHtmlEscaped()));
    } else {
        log_view_->appendPlainText(msg);
    }
    log_view_->verticalScrollBar()->setValue(log_view_->verticalScrollBar()->maximum());
}

void InjectPage::on_confirm_requested() {
    QApplication::beep();
    const auto ret = QMessageBox::question(
        this, QString::fromUtf8("确认导入"),
        QString::fromUtf8("请确认上面列出的图层信息是否为你刚才在 FH6 中创建的占位图形。\n\n"
                          "确认后开始把 shapes.json 写入这些图层（多余的图层会被清零）。"),
        QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
    const bool yes = (ret == QMessageBox::Yes);
    if (!yes) on_log(QString::fromUtf8("[*] 已取消导入"), false);
    if (worker_) worker_->confirm(yes);
}

void InjectPage::on_finished(bool success) {
    running_ = false;
    inject_btn_->setEnabled(!json_path_.isEmpty());
    worker_ = nullptr;
}

void InjectPage::on_preview_ready() {
    preview_image_ = preview_watcher_.result();
    if (preview_image_.isNull()) {
        preview_label_->setText(QString::fromUtf8("无法渲染预览"));
        return;
    }
    preview_label_->setText(QString());
    QSize avail = preview_label_->size() - QSize(8, 8);
    if (avail.width() < 80 || avail.height() < 80) avail = QSize(360, 260);
    preview_label_->setPixmap(QPixmap::fromImage(
        preview_image_.scaled(avail, Qt::KeepAspectRatio, Qt::SmoothTransformation)));
}

} // namespace vinylizer
