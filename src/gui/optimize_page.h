#pragma once

// 页 1「矢量化」：选图 → 手填 N / 选模板策略 / （可选）高级选项 → 生成。
// 运行中复用 GPU→GUI pinned 双缓冲的实时预览；完成后可保存 PNG / shapes.json
// 或直接带着 shapes.json 去注入页。

#include <QWidget>
#include <QImage>
#include <QString>
#include <QFutureWatcher>
#include <string>

#include "gui/optimize_params.h"

class QLabel;
class QPushButton;
class QSpinBox;
class QDoubleSpinBox;
class QComboBox;
class QProgressBar;
class QStackedWidget;
class QToolButton;

namespace vinylizer {

class OptimizeWorker;
class PreviewGLWidget;

class OptimizePage : public QWidget {
    Q_OBJECT
public:
    explicit OptimizePage(QWidget* parent = nullptr);
    ~OptimizePage() override;

    // 供 --autotest 参考：界面默认参数（与控件初值一致）
    static OptimizeParams default_params();

signals:
    void goto_inject_page(const QString& json_path);

protected:
    void dragEnterEvent(QDragEnterEvent* e) override;
    void dropEvent(QDropEvent* e) override;
    void resizeEvent(QResizeEvent* e) override;

private slots:
    void choose_image();
    void on_num_shapes_changed(int n);
    void on_advanced_toggled(bool on);
    void start_optimization();
    void abort_optimization();
    void on_progress(float pct, float mse, int iter);
    void on_finished();
    void on_result_preview_ready();
    void save_png();
    void save_json();
    void goto_inject();

private:
    enum class State { Empty, Config, Running, Done };

    QWidget* build_config_box();
    QWidget* build_running_box();
    QWidget* build_done_box();
    void load_image(const QString& path);
    void set_state(State s);
    void update_n_warning(int n);
    void update_pad_info();
    OptimizeParams collect_params() const;
    void refresh_views();

    // 顶部：选图
    QPushButton* pick_btn_ = nullptr;
    QLabel* file_label_ = nullptr;
    QLabel* size_label_ = nullptr;
    QLabel* hint_label_ = nullptr;

    // 中部：预览（0=原图/占位 1=实时 GL 预览 2=结果）
    QStackedWidget* preview_stack_ = nullptr;
    QLabel* source_view_ = nullptr;
    PreviewGLWidget* gl_view_ = nullptr;
    QLabel* result_view_ = nullptr;

    // 配置区
    QWidget* config_box_ = nullptr;
    QSpinBox* n_spin_ = nullptr;
    QLabel* n_warn_ = nullptr;
    QComboBox* template_combo_ = nullptr;
    QToolButton* adv_toggle_ = nullptr;
    QWidget* adv_panel_ = nullptr;
    QComboBox* init_combo_ = nullptr;
    QDoubleSpinBox* alpha_spin_ = nullptr;
    QSpinBox* grid_spin_ = nullptr;
    QDoubleSpinBox* radius_spin_ = nullptr;
    QSpinBox* lloyd_spin_ = nullptr;
    QComboBox* backend_combo_ = nullptr;
    QSpinBox* seed_spin_ = nullptr;
    QSpinBox* iters_spin_ = nullptr;
    // 学习率退火（高级选项）
    QComboBox* lr_schedule_combo_ = nullptr;
    QDoubleSpinBox* lr_start_spin_ = nullptr;
    QDoubleSpinBox* lr_end_spin_ = nullptr;
    QSpinBox* lr_warmup_spin_ = nullptr;
    // 模板模糊 sigma 退火 + DARTS 锁定比例（高级选项）
    QComboBox* sigma_schedule_combo_ = nullptr;
    QDoubleSpinBox* sigma_start_spin_ = nullptr;
    QDoubleSpinBox* sigma_end_spin_ = nullptr;
    QSpinBox* sigma_warmup_spin_ = nullptr;
    QDoubleSpinBox* darts_lock_spin_ = nullptr;
    QLabel* pad_info_label_ = nullptr;
    QPushButton* start_btn_ = nullptr;

    // 运行区
    QWidget* running_box_ = nullptr;
    QLabel* status_label_ = nullptr;
    QProgressBar* progress_bar_ = nullptr;
    QLabel* running_mse_label_ = nullptr;
    QPushButton* abort_btn_ = nullptr;

    // 完成区
    QWidget* done_box_ = nullptr;
    QLabel* done_info_label_ = nullptr;
    QPushButton* save_png_btn_ = nullptr;
    QPushButton* save_json_btn_ = nullptr;
    QPushButton* goto_inject_btn_ = nullptr;

    // 状态
    State state_ = State::Empty;
    QString image_path_;
    int image_w_ = 0, image_h_ = 0;
    std::string output_json_path_;
    float last_mse_ = 0.0f;
    bool iterating_ = false;
    QImage source_image_;
    QImage result_image_;
    QFutureWatcher<QImage> result_watcher_;
    OptimizeWorker* worker_ = nullptr;
};

} // namespace vinylizer
