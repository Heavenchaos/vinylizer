#include "gui/render_canvas.h"
#include "gui/optimize_worker.h"
#include "gui/inject_worker.h"
#include "common/canvas.h"
#include "common/logger.h"
#include "output/json_writer.h"

#include <QPainter>
#include <QMouseEvent>
#include <QKeyEvent>
#include <QFileDialog>
#include <QOpenGLFunctions>
#include <QDateTime>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>

namespace vinylizer {

// ── Colors ──────────────────────────────────────────────────
static const QColor COL_BG          = "#0f0f23";
static const QColor COL_SIDEBAR     = "#1a1a2e";
static const QColor COL_CARD        = "#16213e";
static const QColor COL_ACCENT      = "#4a9eff";
static const QColor COL_DANGER      = "#e74c3c";
static const QColor COL_TEXT        = "#ffffff";
static const QColor COL_TEXT2       = "#a0a0b0";
static const QColor COL_TEXT3       = "#555555";
static const QColor COL_PROGRESS_BG = "#2a2a3e";
static const QColor COL_MSE_TEXT    = "#ff9f43";
static const QColor COL_INPUT_BORDER= "#333355";
static const QColor COL_INPUT_BG    = "#0d0d1a";

// ── Constructor / Destructor ────────────────────────────────
RenderCanvas::RenderCanvas(QWidget* parent)
    : QOpenGLWidget(parent) {
    setMouseTracking(true);
    setFocusPolicy(Qt::StrongFocus);
    connect(&anim_timer_, &QTimer::timeout, this, &RenderCanvas::on_anim_tick);
    anim_timer_.start(16);
    frame_timer_.start();
    sidebar_indicator_y_ = 0;
    idle_btn_trans_.pos_x = 1.0f;  // Initial scale = 1.0
    idle_btn_scale_ = 1.0f;

    // Async JSON preview
    connect(&inject_preview_watcher_, &QFutureWatcher<QImage>::finished,
            this, &RenderCanvas::on_inject_preview_ready);
    connect(&opt_preview_watcher_, &QFutureWatcher<QImage>::finished,
            this, &RenderCanvas::on_opt_preview_ready);
}

RenderCanvas::~RenderCanvas() {
    if (opt_worker_) {
        opt_worker_->request_stop();
        opt_worker_->wait();
    }
    if (inject_worker_) {
        inject_worker_->wait();
    }
}

// ── GL ──────────────────────────────────────────────────────
void RenderCanvas::initializeGL() {
    initializeOpenGLFunctions();
    glClearColor(0.059f, 0.059f, 0.137f, 1.0f); // #0f0f23
}

void RenderCanvas::resizeGL(int w, int h) {
    win_w_ = w; win_h_ = h;
    content_x_ = sidebar_w_;
    content_w_ = w - sidebar_w_;
    content_h_ = h;
}

void RenderCanvas::paintGL() {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);
    p.setRenderHint(QPainter::TextAntialiasing);

    // Background
    p.fillRect(rect(), COL_BG);

    // Draw pages FIRST (they may slide over content area)
    draw_optimize_page(p);
    draw_inject_page(p);

    // Sidebar ALWAYS on top (drawn last so it covers sliding pages)
    draw_sidebar(p);
}

// ── Animation tick ──────────────────────────────────────────
void RenderCanvas::on_anim_tick() {
    float dt = static_cast<float>(frame_timer_.elapsed());
    frame_timer_.start();

    // Cap dt to avoid jumps
    if (dt > 50) dt = 50;

    sidebar_indicator_anim_.update(dt);
    sidebar_indicator_y_ = sidebar_indicator_anim_.pos_y;

    page_out_trans_.update(dt);
    page_in_trans_.update(dt);

    if (page_switching_ && !page_out_trans_.active && !page_in_trans_.active) {
        page_switching_ = false;
        current_page_ = pending_page_;
    }

    idle_btn_trans_.update(dt);
    idle_btn_scale_ = idle_btn_trans_.pos_x;

    config_trans_.update(dt);
    running_trans_.update(dt);
    src_image_trans_.update(dt);
    inject_page_trans_.update(dt);

    update();
}

// ── Sidebar ─────────────────────────────────────────────────
void RenderCanvas::draw_sidebar(QPainter& p) {
    p.fillRect(0, 0, sidebar_w_, win_h_, COL_SIDEBAR);

    // Tab buttons
    struct TabInfo { const char* label; Page page; int y_center; };
    TabInfo tabs[] = {
        {"制作", Page::Optimize, win_h_ / 2 - 30},
        {"导入", Page::Inject,   win_h_ / 2 + 30}
    };

    QFont font("Microsoft YaHei", 12);
    p.setFont(font);

    for (auto& tab : tabs) {
        bool active = (current_page_ == tab.page && !page_switching_);
        QRect btn_rect(0, tab.y_center - 20, sidebar_w_, 40);

        if (active) {
            p.fillRect(btn_rect, COL_CARD);
            // Indicator bar
            float iy = sidebar_indicator_y_;
            p.fillRect(QRectF(0, iy - 20, 3, 40), COL_ACCENT);
        }

        // Horizontal text
        p.setPen(active ? COL_TEXT : COL_TEXT2);
        p.drawText(QRectF(btn_rect), Qt::AlignCenter, QString::fromUtf8(tab.label));
    }
}

// ── Optimize page ───────────────────────────────────────────
void RenderCanvas::draw_optimize_page(QPainter& p) {
    float ox = 0, alpha = 1.0f;
    if (page_switching_) {
        if (pending_page_ == Page::Optimize) {
            ox = page_in_trans_.pos_x; alpha = page_in_trans_.opacity;
        } else {
            ox = page_out_trans_.pos_x; alpha = page_out_trans_.opacity;
        }
    } else if (current_page_ != Page::Optimize) {
        return;
    }

    p.save();
    p.setOpacity(alpha);
    p.translate(content_x_ + ox, 0);

    switch (opt_state_) {
    case OptState::Idle:    draw_idle_state(p, 0, 1); break;
    case OptState::Config:  draw_config_state(p, config_trans_.pos_x, config_trans_.opacity); break;
    case OptState::Running: draw_running_state(p, running_trans_.pos_x, running_trans_.opacity); break;
    case OptState::Done:    draw_done_state(p); break;
    }

    p.restore();
}

// ── Done state ───────────────────────────────────────────────
void RenderCanvas::draw_done_state(QPainter& p) {
    // Show final preview image centered
    int pw = content_w_ * 55 / 100;
    int ph = content_h_ - 120;

    if (!preview_image_.isNull()) {
        QImage scaled = preview_image_.scaled(pw, ph, Qt::KeepAspectRatio, Qt::SmoothTransformation);
        int dx = (content_w_ - pw) / 2 + (pw - scaled.width()) / 2;
        int dy = 30 + (ph - scaled.height()) / 2;
        draw_rounded_rect(p, QRectF(dx - 2, dy - 2, scaled.width() + 4, scaled.height() + 4), COL_INPUT_BORDER, 6);
        p.drawImage(dx, dy, scaled);
    } else if (opt_preview_loading_) {
        draw_rounded_rect(p, QRectF((content_w_ - pw) / 2, 30, pw, ph), COL_CARD, 6);
        p.setPen(COL_TEXT);
        p.setFont(QFont("Microsoft YaHei", 14));
        p.drawText(QRectF((content_w_ - pw) / 2, 30, pw, ph),
                   Qt::AlignCenter, QString::fromUtf8("正在处理…"));
    }

    // "生成已结束" text
    p.setPen(COL_TEXT);
    p.setFont(QFont("Microsoft YaHei", 18, QFont::Bold));
    p.drawText(QRectF(0, content_h_ - 100, content_w_, 30), Qt::AlignCenter, QString::fromUtf8("生成已结束"));

    // MSE info
    p.setPen(COL_MSE_TEXT);
    p.setFont(QFont("Consolas", 12));
    p.drawText(QRectF(0, content_h_ - 70, content_w_, 20), Qt::AlignCenter,
               QString("Final MSE: %1").arg(current_mse_, 0, 'f', 1));

    // Return button
    auto dbtn = done_btn_rect();
    QColor dc = done_btn_hovered_ ? COL_ACCENT.lighter(110) : COL_ACCENT;
    draw_rounded_rect(p, QRectF(dbtn), dc, 8);
    p.setPen(COL_TEXT);
    p.setFont(QFont("Microsoft YaHei", 13, QFont::Bold));
    p.drawText(QRectF(dbtn), Qt::AlignCenter, QString::fromUtf8("返回"));
}

QRect RenderCanvas::done_btn_rect() const {
    int bw = 160, bh = 40;
    return QRect(content_w_ / 2 - bw / 2, content_h_ - 45, bw, bh);
}

// ── Idle state ──────────────────────────────────────────────
void RenderCanvas::draw_idle_state(QPainter& p, float, float) {
    auto btn = idle_btn_rect();

    p.save();
    p.translate(btn.center().x(), btn.center().y());
    p.scale(idle_btn_scale_, idle_btn_scale_);
    p.translate(-btn.width() / 2, -btn.height() / 2);

    QColor btn_color = COL_ACCENT;
    if (idle_btn_pressed_) btn_color = btn_color.darker(120);
    else if (idle_btn_hovered_) btn_color = btn_color.lighter(110);

    draw_rounded_rect(p, QRectF(0, 0, btn.width(), btn.height()), btn_color, 12);
    p.setPen(COL_TEXT);
    p.setFont(QFont("Microsoft YaHei", 20, QFont::Bold));
    p.drawText(QRectF(0, 0, btn.width(), btn.height()), Qt::AlignCenter, QString::fromUtf8("点击以开始"));

    p.restore();
}

QRect RenderCanvas::idle_btn_rect() const {
    int bw = 320, bh = 80;
    return QRect(content_w_ / 2 - bw / 2, content_h_ / 2 - bh / 2, bw, bh);
}

// ── Config state ────────────────────────────────────────────
void RenderCanvas::draw_config_state(QPainter& p, float ox, float alpha) {
    p.save();
    p.setOpacity(alpha);
    p.translate(ox, 0);

    int margin = 30;
    int left_w = content_w_ * 45 / 100;
    int right_w = content_w_ - left_w - margin * 3;

    // Source image
    auto img_rect = config_image_rect();
    if (!source_image_.isNull()) {
        QImage scaled = source_image_.scaled(img_rect.size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
        int x = img_rect.x() + (img_rect.width() - scaled.width()) / 2;
        int y = img_rect.y() + (img_rect.height() - scaled.height()) / 2;
        draw_rounded_rect(p, QRectF(img_rect.adjusted(-2,-2,2,2)), COL_INPUT_BORDER, 8);
        p.drawImage(x, y, scaled);
    }

    // Parameter panel
    int px = left_w + margin * 2;
    int py = 40;
    int pw = right_w;
    int row_h = 50;
    int label_w = 120;
    int input_h = 32;

    draw_rounded_rect(p, QRectF(px - 15, 20, pw + 30, content_h_ - 40), COL_CARD, 8);

    QFont label_font("Microsoft YaHei", 11);
    QFont value_font("Consolas", 11);

    // File name (read-only)
    p.setFont(label_font);
    p.setPen(COL_TEXT2);
    p.drawText(px, py + 16, QString::fromUtf8("文件名"));
    p.setFont(value_font);
    p.setPen(COL_TEXT);
    QString fname = QString::fromStdString(source_filename_);
    if (fname.length() > 35) fname = fname.left(32) + "...";
    p.drawText(px + label_w, py + 16, fname);
    py += row_h;

    // Resolution (read-only)
    p.setFont(label_font);
    p.setPen(COL_TEXT2);
    p.drawText(px, py + 16, QString::fromUtf8("分辨率"));
    p.setFont(value_font);
    p.setPen(COL_TEXT);
    p.drawText(px + label_w, py + 16, QString("%1 x %2").arg(source_w_).arg(source_h_));
    py += row_h + 10;

    // Editable fields
    auto inputs = config_input_rects();
    const char* labels[] = {"优化器宽度", "优化器高度", "迭代循环数", "图层数量", "学习率起始", "学习率终止"};
    for (int i = 0; i < 6; i++) {
        p.setFont(label_font);
        p.setPen(COL_TEXT2);
        p.drawText(px, inputs[i].y() + input_h - 8, QString::fromUtf8(labels[i]));
        draw_input_field(p, inputs[i], input_texts_[i], cursor_pos_[i], focused_input_ == i);
    }

    // Start button
    auto sbtn = start_btn_rect();
    bool enabled = validate_config();
    QColor sbtn_color = enabled ? COL_ACCENT : COL_TEXT3;
    if (enabled && start_btn_hovered_) sbtn_color = sbtn_color.lighter(110);
    draw_rounded_rect(p, QRectF(sbtn), sbtn_color, 6);
    p.setPen(enabled ? COL_TEXT : COL_TEXT3);
    p.setFont(QFont("Microsoft YaHei", 12, QFont::Bold));
    p.drawText(QRectF(sbtn), Qt::AlignCenter, QString::fromUtf8("开始生成"));

    p.restore();
}

QRect RenderCanvas::config_image_rect() const {
    int left_w = content_w_ * 45 / 100;
    int margin = 30;
    int img_max_h = content_h_ - 80;
    int img_max_w = left_w - margin;
    int iw = std::min(img_max_w, img_max_h);
    return QRect(margin, 40, iw, iw);
}

std::vector<QRect> RenderCanvas::config_input_rects() const {
    int left_w = content_w_ * 45 / 100;
    int margin = 30;
    int px = left_w + margin * 2 + 120;
    int pw = content_w_ - left_w - margin * 3 - 120 - 30;
    int py = 40 + 50 * 2 + 10 + 8;
    int row_h = 50;
    int input_h = 32;

    std::vector<QRect> rects;
    for (int i = 0; i < 6; i++) {
        rects.push_back(QRect(px, py + i * row_h, pw, input_h));
    }
    return rects;
}

QRect RenderCanvas::start_btn_rect() const {
    int left_w = content_w_ * 45 / 100;
    int margin = 30;
    int px = left_w + margin * 2;
    int pw = content_w_ - left_w - margin * 3;
    return QRect(px + pw / 2 - 100, content_h_ - 70, 200, 40);
}

// ── Running state ───────────────────────────────────────────
void RenderCanvas::draw_running_state(QPainter& p, float ox, float alpha) {
    p.save();
    p.setOpacity(alpha);
    p.translate(ox, 0);

    int margin = 20;

    // Source image (top-left, ~25% width)
    int src_w = content_w_ * 25 / 100;
    int src_img_h = 0;
    if (!source_image_.isNull()) {
        QImage scaled = source_image_.scaledToWidth(src_w, Qt::SmoothTransformation);
        src_img_h = scaled.height();
        draw_rounded_rect(p, QRectF(margin - 2, margin - 2, scaled.width() + 4, scaled.height() + 4), COL_INPUT_BORDER, 6);
        p.drawImage(margin, margin, scaled);
    }

    // Info area: directly below source image
    int info_x = margin;
    int info_y = margin + src_img_h + 16;
    int info_w = src_w;

    // Progress bar
    int bar_h = 10;
    draw_rounded_rect(p, QRectF(info_x, info_y, info_w, bar_h), COL_PROGRESS_BG, 5);
    if (progress_ > 0) {
        int fill_w = static_cast<int>(info_w * std::min(1.0f, progress_));
        draw_rounded_rect(p, QRectF(info_x, info_y, fill_w, bar_h), COL_ACCENT, 5);
    }

    // Cycle info
    info_y += bar_h + 10;
    p.setFont(QFont("Consolas", 10));
    p.setPen(COL_TEXT);
    p.drawText(info_x, info_y + 12, QString("Cycle %1/%2").arg(current_cycle_).arg(total_cycles_));

    // MSE
    info_y += 22;
    p.setPen(COL_MSE_TEXT);
    p.drawText(info_x, info_y + 12, QString("MSE: %1").arg(current_mse_, 0, 'f', 1));

    // MSE chart
    info_y += 24;
    int chart_h = 100;
    draw_mse_chart(p, QRectF(info_x, info_y, info_w, chart_h));

    // Stop button
    info_y += chart_h + 12;
    auto sbtn = stop_btn_rect();
    QColor sc = stop_btn_hovered_ ? COL_DANGER.lighter(110) : COL_DANGER;
    draw_rounded_rect(p, QRectF(sbtn), sc, 6);
    p.setPen(COL_TEXT);
    p.setFont(QFont("Microsoft YaHei", 10));
    p.drawText(QRectF(sbtn), Qt::AlignCenter, QString::fromUtf8("停止"));

    // Preview image (right side, takes remaining space)
    int preview_x = src_w + margin * 2 + 10;
    int preview_w = content_w_ - preview_x - margin;
    int preview_h = content_h_ - margin * 2;

    if (!preview_image_.isNull()) {
        QImage scaled = preview_image_.scaled(preview_w, preview_h, Qt::KeepAspectRatio, Qt::SmoothTransformation);
        int dx = preview_x + (preview_w - scaled.width()) / 2;
        int dy = margin + (preview_h - scaled.height()) / 2;
        draw_rounded_rect(p, QRectF(dx - 2, dy - 2, scaled.width() + 4, scaled.height() + 4), COL_INPUT_BORDER, 6);
        p.drawImage(dx, dy, scaled);
    } else {
        draw_rounded_rect(p, QRectF(preview_x, margin, preview_w, preview_h), COL_CARD, 8);
        p.setPen(COL_TEXT3);
        p.setFont(QFont("Microsoft YaHei", 12));
        if (opt_stopping_) {
            p.drawText(QRectF(preview_x, margin, preview_w, preview_h), Qt::AlignCenter, QString::fromUtf8("正在停止..."));
        } else {
            p.drawText(QRectF(preview_x, margin, preview_w, preview_h), Qt::AlignCenter, QString::fromUtf8("优化中..."));
        }
    }

    p.restore();
}

QRect RenderCanvas::stop_btn_rect() const {
    int margin = 20;
    int src_w = content_w_ * 25 / 100;
    // Approximate: below the chart area
    return QRect(margin, content_h_ - 60, 80, 30);
}

// ── MSE chart ───────────────────────────────────────────────
void RenderCanvas::draw_mse_chart(QPainter& p, const QRectF& rect) {
    draw_rounded_rect(p, rect, QColor("#0d0d1a"), 4);

    if (mse_history_.size() < 2) return;

    float min_mse = 1e30f, max_mse = 0;
    for (auto& pt : mse_history_) {
        min_mse = std::min(min_mse, pt.mse);
        max_mse = std::max(max_mse, pt.mse);
    }
    if (max_mse - min_mse < 1) { min_mse -= 1; max_mse += 1; }

    int margin = 8;
    float w = rect.width() - margin * 2;
    float h = rect.height() - margin * 2;

    QPolygonF poly;
    int n = static_cast<int>(mse_history_.size());
    for (int i = 0; i < n; i++) {
        float x = rect.x() + margin + (w * i / (n - 1));
        float y = rect.y() + margin + h - h * (mse_history_[i].mse - min_mse) / (max_mse - min_mse);
        poly << QPointF(x, y);
    }

    QPen pen(COL_ACCENT, 1.5);
    p.setPen(pen);
    p.drawPolyline(poly);
}

// ── Inject page ─────────────────────────────────────────────
void RenderCanvas::draw_inject_page(QPainter& p) {
    float ox = 0, alpha = 1.0f;
    if (page_switching_) {
        if (pending_page_ == Page::Inject) {
            ox = page_in_trans_.pos_x; alpha = page_in_trans_.opacity;
        } else {
            ox = page_out_trans_.pos_x; alpha = page_out_trans_.opacity;
        }
    } else if (current_page_ != Page::Inject) {
        return;
    }

    p.save();
    p.setOpacity(alpha);
    p.translate(content_x_ + ox, 0);

    int left_w = content_w_ * 35 / 100;
    int right_w = content_w_ - left_w;
    int margin = 20;

    // Left panel
    draw_rounded_rect(p, QRectF(margin, 20, left_w - margin * 2, content_h_ - 40), COL_CARD, 8);

    // JSON file button
    auto jbtn = inject_json_btn_rect();
    QColor jbtn_c = inject_json_btn_hovered_ ? COL_ACCENT.lighter(110) : COL_ACCENT;
    draw_rounded_rect(p, QRectF(jbtn), jbtn_c, 6);
    p.setPen(COL_TEXT);
    p.setFont(QFont("Microsoft YaHei", 10));
    p.drawText(QRectF(jbtn), Qt::AlignCenter,
               inject_json_path_.empty() ? QString::fromUtf8("选择 JSON 文件") : QString::fromUtf8("已选择文件"));

    // File name display
    if (!inject_json_filename_.empty()) {
        p.setPen(COL_TEXT2);
        p.setFont(QFont("Consolas", 9));
        QString fn = QString::fromStdString(inject_json_filename_);
        if (fn.length() > 30) fn = fn.left(27) + "...";
        p.drawText(margin + 15, jbtn.bottom() + 18, fn);
    }

    // Layer count input
    int ly = jbtn.bottom() + 40;
    p.setFont(QFont("Microsoft YaHei", 10));
    p.setPen(COL_TEXT2);
    p.drawText(margin + 15, ly, QString::fromUtf8("Layer 数量"));
    auto lrect = inject_layer_input_rect();
    draw_input_field(p, lrect, inject_layer_text_, inject_cursor_pos_, inject_layer_input_focus_ == 1);

    // Inject button
    auto ibtn = inject_btn_rect();
    bool inject_enabled = !inject_json_path_.empty() && !inject_running_;
    QColor ibtn_c = inject_enabled ? (inject_btn_hovered_ ? COL_ACCENT.lighter(110) : COL_ACCENT) : COL_TEXT3;
    draw_rounded_rect(p, QRectF(ibtn), ibtn_c, 6);
    p.setPen(inject_enabled ? COL_TEXT : COL_TEXT3);
    p.setFont(QFont("Microsoft YaHei", 10, QFont::Bold));
    p.drawText(QRectF(ibtn), Qt::AlignCenter, inject_running_ ? QString::fromUtf8("注入中...") : QString::fromUtf8("注入 FH6"));

    // Right panel: log area
    int log_x = left_w + margin;
    int log_y = 20;
    int log_w = right_w - margin * 2;
    int log_h = content_h_ - 40;

    // Log background
    draw_rounded_rect(p, QRectF(log_x, log_y, log_w, log_h), QColor(0x0d, 0x0d, 0x1a, 200), 8);

    // Preview background (45% opacity, scaled to fit log area, centered)
    if (inject_preview_loaded_ && !inject_preview_.isNull()) {
        p.save();
        p.setOpacity(0.45);
        QImage scaled = inject_preview_.scaled(log_w, log_h, Qt::KeepAspectRatio, Qt::SmoothTransformation);
        int dx = log_x + (log_w - scaled.width()) / 2;
        int dy = log_y + (log_h - scaled.height()) / 2;
        p.drawImage(dx, dy, scaled);
        p.restore();
    } else if (inject_preview_loading_) {
        p.setPen(COL_TEXT3);
        p.setFont(QFont("Microsoft YaHei", 12));
        p.drawText(QRectF(log_x, log_y, log_w, log_h), Qt::AlignCenter, QString::fromUtf8("加载预览中..."));
    }

    // Log text
    p.setClipRect(QRectF(log_x + 8, log_y + 8, log_w - 16, log_h - 16));
    QFont log_font("Consolas", 9);
    p.setFont(log_font);
    QFontMetrics fm(log_font);
    int line_h = fm.height() + 2;
    int max_lines = (log_h - 60) / line_h;

    int start = std::max(0, static_cast<int>(inject_logs_.size()) - max_lines);
    for (int i = start; i < static_cast<int>(inject_logs_.size()); i++) {
        int y = log_y + 12 + (i - start) * line_h;
        p.setPen(inject_logs_[i].is_error ? COL_DANGER : COL_TEXT2);
        p.drawText(log_x + 12, y + fm.ascent(), inject_logs_[i].msg);
    }
    p.setClipping(false);

    // Confirm input (if waiting)
    if (inject_waiting_confirm_) {
        int input_y = log_y + log_h - 44;
        draw_rounded_rect(p, QRectF(log_x + 8, input_y, log_w - 16, 36), COL_INPUT_BG, 4);

        auto irect = inject_input_rect();
        draw_input_field(p, irect, inject_confirm_text_, static_cast<int>(inject_confirm_text_.size()), inject_confirm_focused_);

        auto sbtn = inject_send_btn_rect();
        draw_rounded_rect(p, QRectF(sbtn), inject_send_btn_hovered_ ? COL_ACCENT.lighter(110) : COL_ACCENT, 4);
        p.setPen(COL_TEXT);
        p.setFont(QFont("Microsoft YaHei", 9));
        p.drawText(QRectF(sbtn), Qt::AlignCenter, QString::fromUtf8("发送"));
    }

    p.restore();
}

QRect RenderCanvas::inject_json_btn_rect() const {
    int left_w = content_w_ * 35 / 100;
    int margin = 20;
    return QRect(margin + 15, 50, left_w - margin * 2 - 30, 36);
}

QRect RenderCanvas::inject_layer_input_rect() const {
    int left_w = content_w_ * 35 / 100;
    int margin = 20;
    auto jbtn = inject_json_btn_rect();
    int ly = jbtn.bottom() + 40;
    return QRect(margin + 15 + 100, ly - 14, left_w - margin * 2 - 30 - 110, 28);
}

QRect RenderCanvas::inject_btn_rect() const {
    int left_w = content_w_ * 35 / 100;
    int margin = 20;
    auto jbtn = inject_json_btn_rect();
    int ly = jbtn.bottom() + 90;
    return QRect(margin + 15, ly, 200, 36);
}

QRect RenderCanvas::inject_input_rect() const {
    int left_w = content_w_ * 35 / 100;
    int right_w = content_w_ - left_w;
    int margin = 20;
    int log_x = left_w + margin;
    int log_h = content_h_ - 40;
    int input_y = 20 + log_h - 44 + 4;
    return QRect(log_x + 16, input_y, right_w - margin * 2 - 80, 28);
}

QRect RenderCanvas::inject_send_btn_rect() const {
    auto irect = inject_input_rect();
    return QRect(irect.right() + 8, irect.y(), 50, 28);
}

// ── Drawing helpers ─────────────────────────────────────────
void RenderCanvas::draw_rounded_rect(QPainter& p, const QRectF& rect, const QColor& color, float radius) {
    p.setPen(Qt::NoPen);
    p.setBrush(color);
    p.drawRoundedRect(rect, radius, radius);
}

void RenderCanvas::draw_text_centered(QPainter& p, const QRectF& rect, const QString& text, int size, const QColor& color) {
    p.setPen(color);
    p.setFont(QFont("Microsoft YaHei", size));
    p.drawText(rect, Qt::AlignCenter, text);
}

void RenderCanvas::draw_input_field(QPainter& p, const QRectF& rect, const std::string& text, int cursor, bool focused, bool numeric_only) {
    (void)numeric_only;
    QColor bg = COL_INPUT_BG;
    QColor border = focused ? COL_ACCENT : COL_INPUT_BORDER;
    draw_rounded_rect(p, rect, bg, 4);
    p.setPen(border);
    p.setBrush(Qt::NoBrush);
    p.drawRoundedRect(rect, 4, 4);

    p.setPen(COL_TEXT);
    p.setFont(QFont("Consolas", 10));
    QFontMetrics fm(QFont("Consolas", 10));

    QString display = QString::fromStdString(text);
    p.drawText(QRectF(rect.x() + 8, rect.y(), rect.width() - 16, rect.height()),
               Qt::AlignVCenter | Qt::AlignLeft, display);

    // Cursor
    if (focused) {
        int cx = rect.x() + 8 + fm.horizontalAdvance(display.left(cursor));
        p.setPen(COL_ACCENT);
        p.drawLine(cx, rect.y() + 5, cx, rect.bottom() - 5);
    }
}

// ── Mouse events ────────────────────────────────────────────
void RenderCanvas::mousePressEvent(QMouseEvent* e) {
    QPoint pos = e->pos();
    int cx = pos.x() - content_x_;
    int cy = pos.y();

    if (current_page_ == Page::Optimize) {
        if (opt_state_ == OptState::Idle) {
            auto btn = idle_btn_rect();
            if (btn.contains(cx, cy)) {
                idle_btn_pressed_ = true;
                idle_btn_trans_.to(0.97f, 0, 1, 100, Easing::EaseOut);
            }
        } else if (opt_state_ == OptState::Config) {
            auto inputs = config_input_rects();
            focused_input_ = -1;
            for (int i = 0; i < 6; i++) {
                if (inputs[i].contains(cx, cy)) { focused_input_ = i; break; }
            }
            auto sbtn = start_btn_rect();
            if (sbtn.contains(cx, cy) && validate_config()) {
                transition_to_running();
            }
        } else if (opt_state_ == OptState::Running) {
            auto sbtn = stop_btn_rect();
            if (sbtn.contains(cx, cy)) {
                stop_optimization();
            }
        } else if (opt_state_ == OptState::Done) {
            auto dbtn = done_btn_rect();
            if (dbtn.contains(cx, cy)) {
                transition_to_idle();
            }
        }
    } else if (current_page_ == Page::Inject) {
        // Sidebar clicks
        if (pos.x() < sidebar_w_) {
            int tab_y1 = win_h_ / 2 - 30;
            int tab_y2 = win_h_ / 2 + 30;
            if (cy >= tab_y1 - 20 && cy <= tab_y1 + 20) switch_page(Page::Optimize);
            else if (cy >= tab_y2 - 20 && cy <= tab_y2 + 20) switch_page(Page::Inject);
            return;
        }

        auto jbtn = inject_json_btn_rect();
        if (jbtn.contains(cx, cy)) { start_inject_file_dialog(); return; }

        auto lrect = inject_layer_input_rect();
        inject_layer_input_focus_ = lrect.contains(cx, cy) ? 1 : 0;
        inject_confirm_focused_ = false;

        if (inject_waiting_confirm_) {
            auto irect = inject_input_rect();
            if (irect.contains(cx, cy)) { inject_confirm_focused_ = true; }
        }

        auto ibtn = inject_btn_rect();
        if (ibtn.contains(cx, cy) && !inject_json_path_.empty() && !inject_running_) {
            start_inject();
        }

        if (inject_waiting_confirm_) {
            auto sbtn = inject_send_btn_rect();
            if (sbtn.contains(cx, cy)) {
                send_inject_confirm(QString::fromStdString(inject_confirm_text_));
            }
        }
    }

    // Sidebar tab clicks
    if (pos.x() < sidebar_w_) {
        int tab_y1 = win_h_ / 2 - 30;
        int tab_y2 = win_h_ / 2 + 30;
        if (cy >= tab_y1 - 20 && cy <= tab_y1 + 20) switch_page(Page::Optimize);
        else if (cy >= tab_y2 - 20 && cy <= tab_y2 + 20) switch_page(Page::Inject);
    }
}

void RenderCanvas::mouseReleaseEvent(QMouseEvent* e) {
    (void)e;
    if (opt_state_ == OptState::Idle && idle_btn_pressed_) {
        idle_btn_pressed_ = false;
        idle_btn_trans_.to(1.0f, 0, 1, 150, Easing::Spring);

        // Check if still over button
        QPoint pos = e->pos();
        int cx = pos.x() - content_x_;
        int cy = pos.y();
        auto btn = idle_btn_rect();
        if (btn.contains(cx, cy)) {
            start_file_dialog();
        }
    }
}

void RenderCanvas::mouseMoveEvent(QMouseEvent* e) {
    QPoint pos = e->pos();
    int cx = pos.x() - content_x_;
    int cy = pos.y();

    if (opt_state_ == OptState::Idle) {
        auto btn = idle_btn_rect();
        bool was_hovered = idle_btn_hovered_;
        idle_btn_hovered_ = btn.contains(cx, cy);
        if (idle_btn_hovered_ != was_hovered) {
            if (idle_btn_hovered_) {
                idle_btn_trans_.to(1.03f, 0, 1, 150, Easing::Spring);
            } else {
                idle_btn_trans_.to(1.0f, 0, 1, 150, Easing::Spring);
            }
        }
    } else if (opt_state_ == OptState::Config) {
        auto sbtn = start_btn_rect();
        start_btn_hovered_ = sbtn.contains(cx, cy);
    } else if (opt_state_ == OptState::Running) {
        auto sbtn = stop_btn_rect();
        stop_btn_hovered_ = sbtn.contains(cx, cy);
    } else if (opt_state_ == OptState::Done) {
        auto dbtn = done_btn_rect();
        done_btn_hovered_ = dbtn.contains(cx, cy);
    }

    // Inject page hover
    if (current_page_ == Page::Inject) {
        auto jbtn = inject_json_btn_rect();
        inject_json_btn_hovered_ = jbtn.contains(cx, cy);
        auto ibtn = inject_btn_rect();
        inject_btn_hovered_ = ibtn.contains(cx, cy);
        if (inject_waiting_confirm_) {
            auto sbtn = inject_send_btn_rect();
            inject_send_btn_hovered_ = sbtn.contains(cx, cy);
        }
    }
}

void RenderCanvas::keyPressEvent(QKeyEvent* e) {
    if (e->key() == Qt::Key_Tab) {
        switch_page(current_page_ == Page::Optimize ? Page::Inject : Page::Optimize);
        return;
    }

    if (current_page_ == Page::Optimize && opt_state_ == OptState::Config) {
        if (focused_input_ >= 0 && focused_input_ < 6) {
            auto& txt = input_texts_[focused_input_];
            auto& cur = cursor_pos_[focused_input_];

            if (e->key() == Qt::Key_Backspace && cur > 0) {
                txt.erase(txt.begin() + cur - 1);
                cur--;
            } else if (e->key() == Qt::Key_Left && cur > 0) {
                cur--;
            } else if (e->key() == Qt::Key_Right && cur < static_cast<int>(txt.size())) {
                cur++;
            } else if (e->key() == Qt::Key_Enter || e->key() == Qt::Key_Return) {
                if (validate_config()) transition_to_running();
            } else {
                QString ch = e->text();
                if (!ch.isEmpty() && ch[0].isPrint()) {
                    // Numeric only for all fields
                    if (ch[0].isDigit() || (ch[0] == '.' && txt.find('.') == std::string::npos)) {
                        txt.insert(txt.begin() + cur, ch[0].toLatin1());
                        cur++;
                    }
                }
            }
            apply_config_params();
            return;
        }
    }

    if (current_page_ == Page::Inject) {
        if (inject_layer_input_focus_ == 1) {
            auto& txt = inject_layer_text_;
            if (e->key() == Qt::Key_Backspace && inject_cursor_pos_ > 0) {
                txt.erase(txt.begin() + inject_cursor_pos_ - 1);
                inject_cursor_pos_--;
            } else if (e->key() == Qt::Key_Left && inject_cursor_pos_ > 0) {
                inject_cursor_pos_--;
            } else if (e->key() == Qt::Key_Right && inject_cursor_pos_ < static_cast<int>(txt.size())) {
                inject_cursor_pos_++;
            } else {
                QString ch = e->text();
                if (!ch.isEmpty() && ch[0].isDigit()) {
                    txt.insert(txt.begin() + inject_cursor_pos_, ch[0].toLatin1());
                    inject_cursor_pos_++;
                }
            }
            // Parse layer count
            try { inject_layer_count_ = std::stoi(txt); } catch (...) {}
            return;
        }

        if (inject_waiting_confirm_ && inject_confirm_focused_) {
            auto& txt = inject_confirm_text_;
            if (e->key() == Qt::Key_Backspace && !txt.empty()) {
                txt.pop_back();
            } else if (e->key() == Qt::Key_Enter || e->key() == Qt::Key_Return) {
                send_inject_confirm(QString::fromStdString(txt));
            } else {
                QString ch = e->text();
                if (!ch.isEmpty() && ch[0].isPrint()) {
                    txt += ch[0].toLatin1();
                }
            }
            return;
        }
    }
}

// ── Page switching ──────────────────────────────────────────
void RenderCanvas::switch_page(Page page) {
    if (page == current_page_ && !page_switching_) return;
    do_switch_page(page);
}

void RenderCanvas::do_switch_page(Page page) {
    page_switching_ = true;
    pending_page_ = page;

    page_out_trans_.pos_x = 0;
    page_out_trans_.opacity = 1;
    page_out_trans_.slide_out_left(content_w_, 250);

    page_in_trans_.pos_x = content_w_;
    page_in_trans_.opacity = 0;
    page_in_trans_.slide_in_from_right(content_w_, 250);

    // Update sidebar indicator
    float target_y = (page == Page::Optimize) ? win_h_ / 2.0f - 30 : win_h_ / 2.0f + 30;
    sidebar_indicator_anim_.to(0, target_y, 1, 300, Easing::EaseOut);
}

// ── State transitions ───────────────────────────────────────
void RenderCanvas::start_file_dialog() {
    QString path = QFileDialog::getOpenFileName(this, QString::fromUtf8("选择图片"), "",
        "Images (*.png *.jpg *.jpeg)");
    if (path.isEmpty()) return;

    source_path_ = path.toStdString();
    source_image_.load(path);
    if (source_image_.isNull()) return;

    source_w_ = source_image_.width();
    source_h_ = source_image_.height();
    opt_res_w_ = source_w_;
    opt_res_h_ = source_h_;

    // Extract filename
    int sep = path.lastIndexOf('/');
    int sep2 = path.lastIndexOf('\\');
    int last_sep = std::max(sep, sep2);
    source_filename_ = path.mid(last_sep + 1).toStdString();

    // Initialize input fields
    input_texts_[0] = std::to_string(opt_res_w_);
    input_texts_[1] = std::to_string(opt_res_h_);
    input_texts_[2] = std::to_string(num_cycles_);
    input_texts_[3] = std::to_string(num_shapes_);
    input_texts_[4] = std::to_string(lr_start_);
    input_texts_[5] = std::to_string(lr_end_);
    for (int i = 0; i < 6; i++) cursor_pos_[i] = static_cast<int>(input_texts_[i].size());

    transition_to_config();
}

void RenderCanvas::transition_to_config() {
    opt_state_ = OptState::Config;
    config_trans_.pos_x = content_w_;
    config_trans_.opacity = 0;
    config_trans_.slide_in_from_right(content_w_, 250);
}

void RenderCanvas::transition_to_running() {
    apply_config_params();
    opt_state_ = OptState::Running;
    opt_running_ = true;
    opt_stopping_ = false;

    // Source image: already visible at config position, just stays (will be drawn at top-left in running state)
    src_image_trans_.pos_x = 0;
    src_image_trans_.pos_y = 0;
    src_image_trans_.opacity = 1;
    src_image_trans_.active = false;

    // Running page: fade in (no slide from right)
    running_trans_.pos_x = 0;
    running_trans_.opacity = 0;
    running_trans_.fade_in(300);

    // Build output path from input path
    std::string output_path = source_path_;
    auto dot_pos = output_path.rfind('.');
    if (dot_pos != std::string::npos) {
        output_path = output_path.substr(0, dot_pos) + ".json";
    } else {
        output_path += ".json";
    }
    opt_output_path_ = output_path;

    // Start worker
    PipelineConfig pcfg;
    pcfg.input_path = source_path_;
    pcfg.output_path = output_path;
    pcfg.num_shapes = num_shapes_;
    pcfg.canvas_w = opt_res_w_;
    pcfg.canvas_h = opt_res_h_;
    pcfg.num_cycles = num_cycles_;
    pcfg.lr_start = lr_start_;
    pcfg.lr_end = lr_end_;

    progress_ = 0;
    current_mse_ = 0;
    current_cycle_ = 0;
    total_cycles_ = num_cycles_;
    mse_history_.clear();
    preview_image_ = QImage();

    opt_worker_ = new OptimizeWorker(pcfg, this);
    connect(opt_worker_, &OptimizeWorker::progress, this, &RenderCanvas::on_opt_progress);
    connect(opt_worker_, &OptimizeWorker::frame_ready, this, &RenderCanvas::on_opt_frame);
    connect(opt_worker_, &OptimizeWorker::done, this, &RenderCanvas::on_opt_finished);
    connect(opt_worker_, &QThread::finished, opt_worker_, &QObject::deleteLater);
    opt_worker_->start();
}

void RenderCanvas::transition_to_idle() {
    opt_state_ = OptState::Idle;
    opt_running_ = false;
    idle_btn_scale_ = 1.0f;
    idle_btn_trans_.pos_x = 1.0f;
    idle_btn_trans_.active = false;
}

void RenderCanvas::apply_config_params() {
    try { opt_res_w_ = std::stoi(input_texts_[0]); } catch (...) {}
    try { opt_res_h_ = std::stoi(input_texts_[1]); } catch (...) {}
    try { num_cycles_ = std::stoi(input_texts_[2]); } catch (...) {}
    try { num_shapes_ = std::stoi(input_texts_[3]); } catch (...) {}
    try { lr_start_ = std::stof(input_texts_[4]); } catch (...) {}
    try { lr_end_ = std::stof(input_texts_[5]); } catch (...) {}
}

bool RenderCanvas::validate_config() {
    try {
        int w = std::stoi(input_texts_[0]);
        int h = std::stoi(input_texts_[1]);
        int c = std::stoi(input_texts_[2]);
        int n = std::stoi(input_texts_[3]);
        float ls = std::stof(input_texts_[4]);
        float le = std::stof(input_texts_[5]);
        return w > 0 && h > 0 && c > 0 && n > 0 && ls > 0 && le > 0 && le <= ls;
    } catch (...) {
        return false;
    }
}

// ── Optimization ────────────────────────────────────────────
void RenderCanvas::start_optimization() {
    // Will be called from transition_to_running
}

void RenderCanvas::stop_optimization() {
    if (opt_worker_) {
        opt_stopping_ = true;
        opt_worker_->request_stop();
    }
}

void RenderCanvas::on_opt_progress(float pct, float mse, int iter) {
    progress_ = pct;
    current_mse_ = mse;
    current_cycle_ = static_cast<int>(pct * total_cycles_);
    mse_history_.push_back({iter, mse});
    if (mse_history_.size() > 500) mse_history_.erase(mse_history_.begin());
}

void RenderCanvas::on_opt_frame(QImage frame) {
    preview_image_ = frame;
}

void RenderCanvas::on_opt_finished() {
    opt_running_ = false;
    opt_stopping_ = false;
    opt_state_ = OptState::Done;
    opt_worker_ = nullptr; // auto-cleaned via finished->deleteLater

    // Async re-render preview from final JSON (mirrors best-MSE params)
    if (!opt_output_path_.empty()) {
        opt_preview_loading_ = true;
        std::string json_path = opt_output_path_;
        QFuture<QImage> future = QtConcurrent::run([json_path]() -> QImage {
            return RenderCanvas::render_json_preview(json_path);
        });
        opt_preview_watcher_.setFuture(future);
    }

    emit optimization_finished();
}

void RenderCanvas::on_opt_preview_ready() {
    opt_preview_loading_ = false;
    preview_image_ = opt_preview_watcher_.result();
    update();
}

// ── Inject ──────────────────────────────────────────────────
void RenderCanvas::start_inject_file_dialog() {
    QString path = QFileDialog::getOpenFileName(this, QString::fromUtf8("选择 JSON 文件"), "",
        "JSON (*.json)");
    if (path.isEmpty()) return;

    inject_json_path_ = path.toStdString();
    int sep = path.lastIndexOf('/');
    int sep2 = path.lastIndexOf('\\');
    int last_sep = std::max(sep, sep2);
    inject_json_filename_ = path.mid(last_sep + 1).toStdString();

    // Async preview rendering
    inject_preview_loaded_ = false;
    inject_preview_loading_ = true;
    inject_preview_ = QImage();

    std::string json_path = inject_json_path_;
    QFuture<QImage> future = QtConcurrent::run([json_path]() -> QImage {
        return RenderCanvas::render_json_preview(json_path);
    });
    inject_preview_watcher_.setFuture(future);
}

void RenderCanvas::on_inject_preview_ready() {
    inject_preview_ = inject_preview_watcher_.result();
    inject_preview_loaded_ = !inject_preview_.isNull();
    inject_preview_loading_ = false;
}

void RenderCanvas::start_inject() {
    if (inject_json_path_.empty() || inject_running_) return;

    inject_running_ = true;
    inject_logs_.clear();
    inject_waiting_confirm_ = false;

    InjectConfig cfg;
    cfg.json_path = inject_json_path_;
    cfg.layer_count = inject_layer_count_;
    cfg.process_id = 0; // auto-detect

    inject_worker_ = new InjectWorker(cfg, this);
    connect(inject_worker_, &InjectWorker::log_message, this, &RenderCanvas::on_inject_log);
    connect(inject_worker_, &InjectWorker::confirm_requested, this, &RenderCanvas::on_inject_confirm_requested);
    connect(inject_worker_, &InjectWorker::done, this, &RenderCanvas::on_inject_finished);
    connect(inject_worker_, &QThread::finished, inject_worker_, &QObject::deleteLater);
    inject_worker_->start();
}

void RenderCanvas::send_inject_confirm(const QString& text) {
    if (!inject_waiting_confirm_) return;
    inject_waiting_confirm_ = false;
    inject_confirm_focused_ = false;

    QString trimmed = text.trimmed();
    bool yes = (trimmed == "y" || trimmed == "Y");

    if (!yes) {
        on_inject_log(QString::fromUtf8("[*] 已取消导入"), false);
    }

    if (inject_worker_) {
        inject_worker_->confirm(yes);
    }
}

void RenderCanvas::on_inject_log(QString msg, bool is_error) {
    inject_logs_.push_back({msg, is_error});
    if (inject_logs_.size() > 500) inject_logs_.erase(inject_logs_.begin());
}

void RenderCanvas::on_inject_confirm_requested() {
    inject_waiting_confirm_ = true;
    inject_confirm_focused_ = true;
    inject_confirm_text_.clear();
}

void RenderCanvas::on_inject_finished(bool success) {
    inject_running_ = false;
    inject_waiting_confirm_ = false;
    on_inject_log(success ? QString::fromUtf8("注入完成") : QString::fromUtf8("注入失败"), !success);
    inject_worker_ = nullptr; // auto-cleaned via finished->deleteLater
    emit inject_finished();
}

QImage RenderCanvas::render_json_preview(const std::string& json_path) {
    // Use Canvas to render a quick preview
    try {
        auto shapes = validate_json(json_path);
        if (shapes.empty()) return QImage();

        // Compute bounding box from all shapes
        float min_x = 1e9f, min_y = 1e9f, max_x = -1e9f, max_y = -1e9f;
        for (auto& s : shapes) {
            auto& d = s["data"];
            float cx = d[0], cy = d[1], rx = d[2], ry = d[3];
            float angle = d.size() >= 5 ? d[4].get<float>() : 0;
            float cos_a = std::cos(angle * M_PI / 180.0f);
            float sin_a = std::sin(angle * M_PI / 180.0f);
            float hx = std::abs(rx * cos_a) + std::abs(ry * sin_a);
            float hy = std::abs(rx * sin_a) + std::abs(ry * cos_a);
            min_x = std::min(min_x, cx - hx);
            min_y = std::min(min_y, cy - hy);
            max_x = std::max(max_x, cx + hx);
            max_y = std::max(max_y, cy + hy);
        }
        int pad = 4;
        int cw = std::max(1, static_cast<int>(std::ceil(max_x - min_x)) + pad * 2);
        int ch = std::max(1, static_cast<int>(std::ceil(max_y - min_y)) + pad * 2);
        float ox = min_x - pad;
        float oy = min_y - pad;

        Canvas canvas(cw, ch);
        std::vector<Shape> shape_vec;
        for (auto& s : shapes) {
            int type = s["type"];
            auto& data = s["data"];
            auto& color = s["color"];
            Color c(static_cast<uint8_t>(color[0].get<int>()),
                    static_cast<uint8_t>(color[1].get<int>()),
                    static_cast<uint8_t>(color[2].get<int>()),
                    static_cast<uint8_t>(color[3].get<int>()));

            Shape shape;
            shape.type = type;
            shape.cx = data[0].get<float>() - ox;
            shape.cy = data[1].get<float>() - oy;
            shape.rx = data[2]; shape.ry = data[3];
            shape.angle = data.size() >= 5 ? data[4].get<float>() : 0;
            shape.color = c;
            shape.opacity = 1.0f;
            shape_vec.push_back(shape);
        }
        canvas.add_shapes(shape_vec);

        cv::Mat bgr = canvas.clone_image();
        QImage img(bgr.data, bgr.cols, bgr.rows, bgr.step, QImage::Format_RGB888);
        return img.copy();
    } catch (...) {
        return QImage();
    }
}

} // namespace vinylizer
