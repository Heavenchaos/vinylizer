#pragma once

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QTimer>
#include <QImage>
#include <QElapsedTimer>
#include <QFutureWatcher>
#include <QPointer>
#include <QtConcurrent>
#include <vector>
#include <string>
#include <memory>
#include "gui/transition.h"

namespace vinylizer {

class OptimizeWorker;
class InjectWorker;

enum class Page { Optimize, Inject };
enum class OptState { Idle, Config, Running, Done };

struct MsePoint {
    int iter;
    float mse;
};

class RenderCanvas : public QOpenGLWidget, protected QOpenGLFunctions {
    Q_OBJECT
public:
    explicit RenderCanvas(QWidget* parent = nullptr);
    ~RenderCanvas();

    void switch_page(Page page);

signals:
    void optimization_finished();
    void inject_finished();

protected:
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;

    void mousePressEvent(QMouseEvent* e) override;
    void mouseReleaseEvent(QMouseEvent* e) override;
    void mouseMoveEvent(QMouseEvent* e) override;
    void keyPressEvent(QKeyEvent* e) override;

private:
    // Animation timer
    QTimer anim_timer_;
    QElapsedTimer frame_timer_;

    // Layout
    int sidebar_w_ = 80;
    int content_x_ = 80;
    int content_w_ = 0;
    int content_h_ = 0;
    int win_w_ = 1200, win_h_ = 800;

    // Page state
    Page current_page_ = Page::Optimize;
    OptState opt_state_ = OptState::Idle;

    // Sidebar animation
    float sidebar_indicator_y_ = 0;
    Transition sidebar_indicator_anim_;

    // Page transitions
    Transition page_out_trans_;
    Transition page_in_trans_;
    bool page_switching_ = false;
    Page pending_page_ = Page::Optimize;

    // ── Optimize: Idle ──
    float idle_btn_scale_ = 1.0f;
    Transition idle_btn_trans_;
    bool idle_btn_hovered_ = false;
    bool idle_btn_pressed_ = false;

    // ── Optimize: Config ──
    Transition config_trans_;
    QImage source_image_;
    std::string source_path_;
    int source_w_ = 0, source_h_ = 0;
    std::string source_filename_;
    int opt_res_w_ = 0, opt_res_h_ = 0;
    int num_cycles_ = 25;
    int num_shapes_ = 1500;
    float lr_start_ = 1.0f, lr_end_ = 0.6f;
    int focused_input_ = -1;
    std::string input_texts_[6];
    bool start_btn_hovered_ = false;
    int cursor_pos_[6] = {};

    // ── Optimize: Running ──
    Transition running_trans_;
    Transition src_image_trans_;
    float progress_ = 0;
    float current_mse_ = 0;
    int current_cycle_ = 0;
    int total_cycles_ = 0;
    std::vector<MsePoint> mse_history_;
    QImage preview_image_;
    std::string opt_output_path_;
    QFutureWatcher<QImage> opt_preview_watcher_;
    bool opt_preview_loading_ = false;
    bool stop_btn_hovered_ = false;
    bool opt_running_ = false;
    bool opt_stopping_ = false;
    bool done_btn_hovered_ = false;

    // ── Inject page ──
    Transition inject_page_trans_;
    std::string inject_json_path_;
    std::string inject_json_filename_;
    int inject_layer_count_ = 2000;
    QImage inject_preview_;
    bool inject_preview_loaded_ = false;
    bool inject_preview_loading_ = false;
    QFutureWatcher<QImage> inject_preview_watcher_;
    struct LogEntry { QString msg; bool is_error; };
    std::vector<LogEntry> inject_logs_;
    bool inject_waiting_confirm_ = false;
    std::string inject_confirm_text_;
    bool inject_confirm_focused_ = false;
    bool inject_json_btn_hovered_ = false;
    bool inject_btn_hovered_ = false;
    bool inject_send_btn_hovered_ = false;
    bool inject_running_ = false;
    int inject_layer_input_focus_ = 0; // 0=none, 1=layer count
    std::string inject_layer_text_ = "2000";
    int inject_cursor_pos_ = 4;
    float inject_scroll_offset_ = 0;

    // Workers
    QPointer<OptimizeWorker> opt_worker_;
    QPointer<InjectWorker> inject_worker_;

    // Drawing helpers
    void draw_sidebar(QPainter& p);
    void draw_optimize_page(QPainter& p);
    void draw_idle_state(QPainter& p, float offset_x, float alpha);
    void draw_done_state(QPainter& p);
    void draw_config_state(QPainter& p, float offset_x, float alpha);
    void draw_running_state(QPainter& p, float offset_x, float alpha);
    void draw_inject_page(QPainter& p);
    void draw_rounded_rect(QPainter& p, const QRectF& rect, const QColor& color, float radius = 6);
    void draw_text_centered(QPainter& p, const QRectF& rect, const QString& text, int size, const QColor& color);
    void draw_input_field(QPainter& p, const QRectF& rect, const std::string& text, int cursor, bool focused, bool numeric_only = true);
    void draw_mse_chart(QPainter& p, const QRectF& rect);

    // Hit testing
    QRect idle_btn_rect() const;
    QRect done_btn_rect() const;
    QRect start_btn_rect() const;
    QRect stop_btn_rect() const;
    QRect inject_json_btn_rect() const;
    QRect inject_btn_rect() const;
    QRect inject_send_btn_rect() const;
    QRect inject_input_rect() const;
    QRect inject_layer_input_rect() const;
    std::vector<QRect> config_input_rects() const;
    QRect config_image_rect() const;

    // State transitions
    void transition_to_config();
    void transition_to_running();
    void transition_to_idle();
    void do_switch_page(Page page);

    // Helpers
    void apply_config_params();
    bool validate_config();
    static QImage render_json_preview(const std::string& json_path);

    // Actions
    void start_file_dialog();
    void start_optimization();
    void stop_optimization();
    void start_inject_file_dialog();
    void start_inject();
    void send_inject_confirm(const QString& text);

private slots:
    void on_anim_tick();
    void on_opt_progress(float pct, float mse, int iter);
    void on_opt_frame(QImage frame);
    void on_opt_finished();
    void on_opt_preview_ready();
    void on_inject_log(QString msg, bool is_error);
    void on_inject_confirm_requested();
    void on_inject_finished(bool success);
    void on_inject_preview_ready();
};

} // namespace vinylizer
