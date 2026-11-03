#include "gui/optimize_page.h"
#include "gui/optimize_worker.h"
#include "gui/preview_gl_widget.h"
#include "common/preview_render.h"
#include "common/logger.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QLabel>
#include <QPushButton>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QComboBox>
#include <QProgressBar>
#include <QStackedWidget>
#include <QToolButton>
#include <QFileDialog>
#include <QFileInfo>
#include <QMessageBox>
#include <QDir>
#include <QStandardPaths>
#include <QFile>
#include <QResizeEvent>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QMimeData>
#include <QUrl>
#include <QtConcurrent>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>

namespace vinylizer {

static const char* kImageFilter = "图片 (*.png *.jpg *.jpeg *.bmp *.webp)";

static QSpinBox* make_spin(int lo, int hi, int val) {
    auto* s = new QSpinBox;
    s->setRange(lo, hi);
    s->setValue(val);
    s->setMinimumWidth(120);
    return s;
}

static QDoubleSpinBox* make_dspin(double lo, double hi, double step, double val) {
    auto* s = new QDoubleSpinBox;
    s->setRange(lo, hi);
    s->setSingleStep(step);
    s->setDecimals(2);
    s->setValue(val);
    s->setMinimumWidth(120);
    return s;
}

OptimizePage::OptimizePage(QWidget* parent) : QWidget(parent) {
    setAcceptDrops(true);

    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(16, 16, 16, 16);
    root->setSpacing(10);

    // ── 选图 ──
    auto* top = new QHBoxLayout;
    pick_btn_ = new QPushButton(QString::fromUtf8("选择图片"));
    pick_btn_->setMinimumHeight(32);
    hint_label_ = new QLabel(QString::fromUtf8("也可以把图片拖到这里"));
    file_label_ = new QLabel(QString::fromUtf8("文件名：—"));
    size_label_ = new QLabel(QString::fromUtf8("尺寸：—"));
    top->addWidget(pick_btn_);
    top->addWidget(hint_label_);
    top->addSpacing(24);
    top->addWidget(file_label_);
    top->addWidget(size_label_);
    top->addStretch();
    root->addLayout(top);

    // ── 预览 ──
    preview_stack_ = new QStackedWidget;
    preview_stack_->setMinimumHeight(260);
    source_view_ = new QLabel(QString::fromUtf8("尚未选择图片\n点击「选择图片」或把图片拖进来"));
    source_view_->setAlignment(Qt::AlignCenter);
    source_view_->setStyleSheet("background:#0d0d1a; color:#a0a0b0; border:1px solid #333355;");
    gl_view_ = new PreviewGLWidget;
    result_view_ = new QLabel(QString::fromUtf8("生成中…"));
    result_view_->setAlignment(Qt::AlignCenter);
    result_view_->setStyleSheet("background:#0d0d1a; color:#a0a0b0; border:1px solid #333355;");
    preview_stack_->addWidget(source_view_);   // 0
    preview_stack_->addWidget(gl_view_);       // 1
    preview_stack_->addWidget(result_view_);   // 2
    root->addWidget(preview_stack_, 1);

    root->addWidget(build_config_box());
    root->addWidget(build_running_box());
    root->addWidget(build_done_box());

    connect(pick_btn_, &QPushButton::clicked, this, &OptimizePage::choose_image);
    connect(&result_watcher_, &QFutureWatcher<QImage>::finished,
            this, &OptimizePage::on_result_preview_ready);

    update_n_warning(n_spin_->value());
    set_state(State::Empty);
}

OptimizePage::~OptimizePage() {
    if (worker_) {
        worker_->request_stop();
        worker_->wait();
    }
    result_watcher_.waitForFinished();
}

OptimizeParams OptimizePage::default_params() {
    OptimizeParams p;
    p.num_shapes = 1500;
    p.template_mode = TemplateMode::Auto;
    p.init_mode = "kmedoids";
    p.km_alpha = 5.0f;
    p.km_grid = 4;
    p.km_radius_scale = 2.5f;
    p.km_iters = 15;
    p.km_backend = "auto";
    p.random_seed = 42;
    p.total_iters = 5000;
    // 学习率退火默认值（与 CLI / PipelineConfig 三方一致）：1.0 → 0.0 cosine。
    p.lr_start = 1.0f;
    p.lr_end = 0.0f;
    p.lr_schedule = "cosine";
    p.lr_warmup_iters = 0;
    // 模板模糊 sigma 退火 + DARTS 锁定比例（与 CLI / PipelineConfig 三方一致）
    p.blur_sigma_start = 2.0f;
    p.blur_sigma_end = 0.0f;
    p.blur_sigma_schedule = "cosine";
    p.blur_sigma_warmup = 0;
    p.darts_lock_ratio = 1.0f;
    return p;
}

// ── 配置区 ──────────────────────────────────────────────────
QWidget* OptimizePage::build_config_box() {
    config_box_ = new QWidget;
    auto* v = new QVBoxLayout(config_box_);
    v->setContentsMargins(0, 0, 0, 0);
    v->setSpacing(8);

    auto* nrow = new QHBoxLayout;
    nrow->addWidget(new QLabel(QString::fromUtf8("形状数量 N")));
    n_spin_ = make_spin(1, 1000000, default_params().num_shapes);
    nrow->addWidget(n_spin_);
    nrow->addWidget(new QLabel(QString::fromUtf8("（手动输入，无预设档位）")));
    nrow->addStretch();
    v->addLayout(nrow);

    n_warn_ = new QLabel;
    n_warn_->setWordWrap(true);
    v->addWidget(n_warn_);

    auto* trow = new QHBoxLayout;
    trow->addWidget(new QLabel(QString::fromUtf8("形状模板")));
    template_combo_ = new QComboBox;
    template_combo_->addItem(QString::fromUtf8("自动选择模板（四型搜索）"));
    template_combo_->addItem(QString::fromUtf8("锁定：软圆 228"));
    template_combo_->addItem(QString::fromUtf8("锁定：椭圆 16"));
    template_combo_->addItem(QString::fromUtf8("锁定：矩形 1"));
    template_combo_->addItem(QString::fromUtf8("锁定：三角 103"));
    template_combo_->setCurrentIndex(0);
    template_combo_->setMinimumWidth(220);
    trow->addWidget(template_combo_);
    trow->addStretch();
    v->addLayout(trow);

    adv_toggle_ = new QToolButton;
    adv_toggle_->setText(QString::fromUtf8("高级选项"));
    adv_toggle_->setCheckable(true);
    adv_toggle_->setChecked(false);
    adv_toggle_->setArrowType(Qt::RightArrow);
    adv_toggle_->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    v->addWidget(adv_toggle_, 0, Qt::AlignLeft);

    adv_panel_ = new QWidget;
    auto* g = new QGridLayout(adv_panel_);
    g->setContentsMargins(16, 0, 0, 0);
    g->setHorizontalSpacing(12);
    g->setVerticalSpacing(6);

    int r = 0;
    g->addWidget(new QLabel(QString::fromUtf8("初始化方式")), r, 0);
    init_combo_ = new QComboBox;
    init_combo_->addItem(QString::fromUtf8("智能布局初始化"));
    init_combo_->addItem(QString::fromUtf8("方差初始化"));
    init_combo_->setCurrentIndex(0);
    g->addWidget(init_combo_, r, 1);
    g->addWidget(new QLabel(QString::fromUtf8("推荐：智能布局初始化")), r, 2);
    ++r;

    g->addWidget(new QLabel(QString::fromUtf8("细节强度 α")), r, 0);
    alpha_spin_ = make_dspin(0.0, 20.0, 0.5, default_params().km_alpha);
    g->addWidget(alpha_spin_, r, 1);
    g->addWidget(new QLabel(QString::fromUtf8("3~5 跨图稳健")), r, 2);
    ++r;

    g->addWidget(new QLabel(QString::fromUtf8("网格")), r, 0);
    grid_spin_ = make_spin(1, 64, default_params().km_grid);
    g->addWidget(grid_spin_, r, 1);
    ++r;

    g->addWidget(new QLabel(QString::fromUtf8("半径系数")), r, 0);
    radius_spin_ = make_dspin(0.1, 10.0, 0.1, default_params().km_radius_scale);
    g->addWidget(radius_spin_, r, 1);
    ++r;

    g->addWidget(new QLabel(QString::fromUtf8("Lloyd 迭代")), r, 0);
    lloyd_spin_ = make_spin(0, 200, default_params().km_iters);
    g->addWidget(lloyd_spin_, r, 1);
    ++r;

    g->addWidget(new QLabel(QString::fromUtf8("计算后端")), r, 0);
    backend_combo_ = new QComboBox;
    backend_combo_->addItem(QString::fromUtf8("自动"));
    backend_combo_->addItem(QString::fromUtf8("CPU"));
    backend_combo_->addItem(QString::fromUtf8("GPU"));
    backend_combo_->setCurrentIndex(0);
    g->addWidget(backend_combo_, r, 1);
    ++r;

    g->addWidget(new QLabel(QString::fromUtf8("随机种子")), r, 0);
    seed_spin_ = make_spin(0, 2147483647, default_params().random_seed);
    g->addWidget(seed_spin_, r, 1);
    ++r;

    g->addWidget(new QLabel(QString::fromUtf8("总迭代数")), r, 0);
    iters_spin_ = make_spin(100, 100000, default_params().total_iters);
    iters_spin_->setToolTip(QString::fromUtf8(
        "优化总迭代步数（默认 5000）。数值越大越精细，耗时线性增长。"));
    g->addWidget(iters_spin_, r, 1);
    ++r;

    // ── 学习率退火 ────────────────────────────────────────────
    g->addWidget(new QLabel(QString::fromUtf8("学习率调度")), r, 0);
    lr_schedule_combo_ = new QComboBox;
    lr_schedule_combo_->addItem(QString::fromUtf8("cosine（余弦退火，推荐）"));
    lr_schedule_combo_->addItem(QString::fromUtf8("linear（线性退火）"));
    lr_schedule_combo_->addItem(QString::fromUtf8("piecewise（分段退火）"));
    lr_schedule_combo_->setCurrentIndex(0);
    lr_schedule_combo_->setToolTip(QString::fromUtf8(
        "学习率随迭代下降的曲线形状。默认 cosine。\n"
        "注意：若起始值 = 终止值，任何调度都退化为“恒定学习率”。"));
    g->addWidget(lr_schedule_combo_, r, 1);
    g->addWidget(new QLabel(QString::fromUtf8("推荐 cosine")), r, 2);
    ++r;

    g->addWidget(new QLabel(QString::fromUtf8("学习率 起始")), r, 0);
    lr_start_spin_ = make_dspin(0.0, 20.0, 0.1, default_params().lr_start);
    lr_start_spin_->setToolTip(QString::fromUtf8(
        "退火起点的学习率（默认 1.0）。\n"
        "Adam 满步长阶段的实际步长 ≈ 学习率 × sign(梯度)，与梯度大小无关，\n"
        "所以这个值直接决定前期“走多快”。"));
    g->addWidget(lr_start_spin_, r, 1);
    ++r;

    g->addWidget(new QLabel(QString::fromUtf8("学习率 终止")), r, 0);
    lr_end_spin_ = make_dspin(0.0, 20.0, 0.05, default_params().lr_end);
    lr_end_spin_->setToolTip(QString::fromUtf8(
        "退火终点的学习率（默认 0.0）。\n"
        "承重项：必须退到接近 0，否则参数会在最优点附近一直抖动，\n"
        "训练末期无法精修（实测残差 ∝ 学习率²）。"));
    g->addWidget(lr_end_spin_, r, 1);
    g->addWidget(new QLabel(QString::fromUtf8("建议保持 0.0")), r, 2);
    ++r;

    g->addWidget(new QLabel(QString::fromUtf8("学习率预热步数")), r, 0);
    lr_warmup_spin_ = make_spin(0, 1000, default_params().lr_warmup_iters);
    lr_warmup_spin_->setToolTip(QString::fromUtf8(
        "前 N 步把学习率从 0 线性升到“起始”值（0 = 关闭，默认）。\n"
        "从零训练一般不需要；但从已有结果继续微调时建议设 5，\n"
        "可避免第 1~2 步的满步长把已有解踢出低谷。"));
    g->addWidget(lr_warmup_spin_, r, 1);
    g->addWidget(new QLabel(QString::fromUtf8("0=关；微调建议 5")), r, 2);
    ++r;

    // ── 模板模糊 sigma 退火 ────────────────────────────────────
    g->addWidget(new QLabel(QString::fromUtf8("模板模糊调度")), r, 0);
    sigma_schedule_combo_ = new QComboBox;
    sigma_schedule_combo_->addItem(QString::fromUtf8("cosine（余弦退火，推荐）"));
    sigma_schedule_combo_->addItem(QString::fromUtf8("linear（线性退火）"));
    sigma_schedule_combo_->addItem(QString::fromUtf8("piecewise（分段退火）"));
    sigma_schedule_combo_->addItem(QString::fromUtf8("none（恒定，不退火）"));
    sigma_schedule_combo_->setCurrentIndex(0);
    sigma_schedule_combo_->setToolTip(QString::fromUtf8(
        "控制形状边缘的“软硬”随训练如何变化。\n"
        "选 none 时固定用“起始”值。"));
    g->addWidget(sigma_schedule_combo_, r, 1);
    g->addWidget(new QLabel(QString::fromUtf8("推荐 cosine")), r, 2);
    ++r;

    g->addWidget(new QLabel(QString::fromUtf8("模板模糊 起始")), r, 0);
    sigma_start_spin_ = make_dspin(0.0, 10.0, 0.1, default_params().blur_sigma_start);
    sigma_start_spin_->setToolTip(QString::fromUtf8(
        "训练起点的模糊强度（默认 2.0）。数值越大边缘越软。\n"
        "早期适度模糊有助于先铺出大结构。"));
    g->addWidget(sigma_start_spin_, r, 1);
    ++r;

    g->addWidget(new QLabel(QString::fromUtf8("模板模糊 终止")), r, 0);
    sigma_end_spin_ = make_dspin(0.0, 10.0, 0.1, default_params().blur_sigma_end);
    sigma_end_spin_->setToolTip(QString::fromUtf8(
        "训练终点的模糊强度（默认 0.0 = 完全锐利）。\n"
        "注意：模糊量正比于形状尺寸（sigma_px = sigma × 形状半径 / 64）。\n"
        "实测退到 0.0 会损失小形状的抗锯齿；设 0.5 在测试图上略优。"));
    g->addWidget(sigma_end_spin_, r, 1);
    g->addWidget(new QLabel(QString::fromUtf8("0=最锐；0.5 略优")), r, 2);
    ++r;

    g->addWidget(new QLabel(QString::fromUtf8("模糊预热步数")), r, 0);
    sigma_warmup_spin_ = make_spin(0, 5000, default_params().blur_sigma_warmup);
    sigma_warmup_spin_->setToolTip(QString::fromUtf8(
        "前 N 步保持“起始”模糊强度，之后再开始退火（0 = 立即开始）。"));
    g->addWidget(sigma_warmup_spin_, r, 1);
    ++r;

    // ── DARTS 软/硬渲染切换点 ──────────────────────────────────
    g->addWidget(new QLabel(QString::fromUtf8("类型锁定位置")), r, 0);
    darts_lock_spin_ = make_dspin(0.0, 1.0, 0.05, default_params().darts_lock_ratio);
    darts_lock_spin_->setToolTip(QString::fromUtf8(
        "四型模板搜索从“软混合”切换为“硬渲染”的训练进度点（0~1）。\n"
        "1.0 = 只在最后一个迭代切（默认）；0.0 = 全程按硬渲染搜索。\n"
        "实测 0.4 与 1.0 统计上没有差别，0.7 反而最差 —— 这一项对结果不敏感，\n"
        "没有特别理由就保持 1.0。"));
    g->addWidget(darts_lock_spin_, r, 1);
    g->addWidget(new QLabel(QString::fromUtf8("不敏感；建议 1.0")), r, 2);
    ++r;

    g->addWidget(new QLabel(QString::fromUtf8("透明边距")), r, 0);
    pad_info_label_ = new QLabel;
    pad_info_label_->setToolTip(QString::fromUtf8(
        "为防止形状被画布边界截断，在图像四周加的透明边距。按图像尺寸自动取"
        "（约 5.3%，下限 16px），大图 ≈100px、小图自动收缩，无需手动设置。"));
    g->addWidget(pad_info_label_, r, 1);
    g->addWidget(new QLabel(QString::fromUtf8("自动按图像尺寸比例")), r, 2);
    ++r;
    update_pad_info();  // 初始占位（未选图）

    g->setColumnStretch(3, 1);
    adv_panel_->setVisible(false);
    v->addWidget(adv_panel_);

    start_btn_ = new QPushButton(QString::fromUtf8("开始生成"));
    start_btn_->setMinimumHeight(38);
    v->addWidget(start_btn_);

    connect(n_spin_, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &OptimizePage::on_num_shapes_changed);
    connect(adv_toggle_, &QToolButton::toggled, this, &OptimizePage::on_advanced_toggled);
    connect(start_btn_, &QPushButton::clicked, this, &OptimizePage::start_optimization);
    return config_box_;
}

// ── 运行区 ──────────────────────────────────────────────────
QWidget* OptimizePage::build_running_box() {
    running_box_ = new QWidget;
    auto* v = new QVBoxLayout(running_box_);
    v->setContentsMargins(0, 0, 0, 0);
    v->setSpacing(6);

    status_label_ = new QLabel(QString::fromUtf8("正在初始化形状…"));
    v->addWidget(status_label_);

    progress_bar_ = new QProgressBar;
    progress_bar_->setRange(0, 1000);
    progress_bar_->setValue(0);
    v->addWidget(progress_bar_);

    auto* row = new QHBoxLayout;
    running_mse_label_ = new QLabel(QString::fromUtf8("MSE: —"));
    row->addWidget(running_mse_label_);
    row->addStretch();
    abort_btn_ = new QPushButton(QString::fromUtf8("中止"));
    row->addWidget(abort_btn_);
    v->addLayout(row);

    connect(abort_btn_, &QPushButton::clicked, this, &OptimizePage::abort_optimization);
    running_box_->setVisible(false);
    return running_box_;
}

// ── 完成区 ──────────────────────────────────────────────────
QWidget* OptimizePage::build_done_box() {
    done_box_ = new QWidget;
    auto* v = new QVBoxLayout(done_box_);
    v->setContentsMargins(0, 0, 0, 0);
    v->setSpacing(8);

    done_info_label_ = new QLabel;
    done_info_label_->setStyleSheet("color:#ff9f43; font-family:Consolas;");
    v->addWidget(done_info_label_);

    auto* row = new QHBoxLayout;
    save_png_btn_ = new QPushButton(QString::fromUtf8("保存 PNG"));
    save_json_btn_ = new QPushButton(QString::fromUtf8("保存 shapes.json"));
    goto_inject_btn_ = new QPushButton(QString::fromUtf8("前往注入页"));
    row->addWidget(save_png_btn_);
    row->addWidget(save_json_btn_);
    row->addWidget(goto_inject_btn_);
    row->addStretch();
    v->addLayout(row);

    connect(save_png_btn_, &QPushButton::clicked, this, &OptimizePage::save_png);
    connect(save_json_btn_, &QPushButton::clicked, this, &OptimizePage::save_json);
    connect(goto_inject_btn_, &QPushButton::clicked, this, &OptimizePage::goto_inject);

    done_box_->setVisible(false);
    return done_box_;
}

// ── 选图 ────────────────────────────────────────────────────
void OptimizePage::choose_image() {
    QString path = QFileDialog::getOpenFileName(this, QString::fromUtf8("选择图片"), QString(),
                                                QString::fromUtf8(kImageFilter));
    if (path.isEmpty()) return;
    load_image(path);
}

void OptimizePage::load_image(const QString& path) {
    QImage img;
    if (!img.load(path)) {
        QMessageBox::warning(this, QString::fromUtf8("无法打开图片"),
                             QString::fromUtf8("这个文件读不出来，请换一张图片。"));
        return;
    }
    image_path_ = path;
    source_image_ = img;
    image_w_ = img.width();
    image_h_ = img.height();
    file_label_->setText(QString::fromUtf8("文件名：") + QFileInfo(path).fileName());
    size_label_->setText(QString::fromUtf8("尺寸：%1 x %2").arg(image_w_).arg(image_h_));
    update_pad_info();
    result_image_ = QImage();
    result_view_->setPixmap(QPixmap());
    preview_stack_->setCurrentWidget(source_view_);
    refresh_views();
    set_state(State::Config);
}

void OptimizePage::dragEnterEvent(QDragEnterEvent* e) {
    if (state_ == State::Running) return;
    if (!e->mimeData()->hasUrls()) return;
    const QString p = e->mimeData()->urls().first().toLocalFile();
    const QString ext = QFileInfo(p).suffix().toLower();
    if (ext == "png" || ext == "jpg" || ext == "jpeg" || ext == "bmp" || ext == "webp") {
        e->acceptProposedAction();
    }
}

void OptimizePage::dropEvent(QDropEvent* e) {
    if (state_ == State::Running) return;
    if (!e->mimeData()->hasUrls()) return;
    const QString p = e->mimeData()->urls().first().toLocalFile();
    if (p.isEmpty()) return;
    e->acceptProposedAction();
    load_image(p);
}

void OptimizePage::resizeEvent(QResizeEvent* e) {
    QWidget::resizeEvent(e);
    refresh_views();
}

void OptimizePage::refresh_views() {
    QSize avail = preview_stack_->size();
    if (avail.width() < 80 || avail.height() < 80) avail = QSize(720, 480);
    avail -= QSize(8, 8);
    if (!source_image_.isNull()) {
        source_view_->setPixmap(QPixmap::fromImage(
            source_image_.scaled(avail, Qt::KeepAspectRatio, Qt::SmoothTransformation)));
    }
    if (!result_image_.isNull()) {
        result_view_->setPixmap(QPixmap::fromImage(
            result_image_.scaled(avail, Qt::KeepAspectRatio, Qt::SmoothTransformation)));
    }
}

void OptimizePage::on_num_shapes_changed(int n) { update_n_warning(n); }

void OptimizePage::update_n_warning(int n) {
    const QString base = QString::fromUtf8("提示：N 不能超过你在 FH6 涂装编辑器里摆放的占位图层数。");
    if (n > 200000) {
        n_warn_->setText(base + QString::fromUtf8(
            "\n注意：N 已超过 200000，可能受显存与游戏图层数限制。"));
        n_warn_->setStyleSheet("color:#ff9f43;");
    } else {
        n_warn_->setText(base);
        n_warn_->setStyleSheet("color:#a0a0b0;");
    }
}

void OptimizePage::on_advanced_toggled(bool on) {
    adv_toggle_->setArrowType(on ? Qt::DownArrow : Qt::RightArrow);
    adv_panel_->setVisible(on);
    if (on) update_pad_info();
}

void OptimizePage::update_pad_info() {
    if (!pad_info_label_) return;
    if (image_w_ <= 0 || image_h_ <= 0) {
        pad_info_label_->setText(QString::fromUtf8("—（选图后显示）"));
        return;
    }
    const int pad = effective_boundary_pad(image_w_, image_h_, -1,
                                           kGuiPadRatio, kGuiPadMin, kGuiPadMax);
    const int cw = image_w_ + 2 * pad, ch = image_h_ + 2 * pad;
    const double border = 1.0 - double(image_w_) * image_h_ / (double(cw) * ch);
    pad_info_label_->setText(QString::fromUtf8("%1 px（画布 %2×%3，边框 %4%）")
                                 .arg(pad).arg(cw).arg(ch)
                                 .arg(border * 100.0, 0, 'f', 1));
}

void OptimizePage::set_state(State s) {
    state_ = s;
    config_box_->setVisible(s == State::Empty || s == State::Config);
    running_box_->setVisible(s == State::Running);
    done_box_->setVisible(s == State::Done);
    pick_btn_->setEnabled(s != State::Running);
    start_btn_->setEnabled(s == State::Config);
}

OptimizeParams OptimizePage::collect_params() const {
    OptimizeParams p = default_params();
    p.input_path = image_path_.toStdString();
    p.canvas_w = image_w_;
    p.canvas_h = image_h_;
    p.num_shapes = n_spin_->value();
    p.template_mode = static_cast<TemplateMode>(template_combo_->currentIndex());
    p.init_mode = (init_combo_->currentIndex() == 0) ? "kmedoids" : "variance";
    p.km_alpha = static_cast<float>(alpha_spin_->value());
    p.km_grid = grid_spin_->value();
    p.km_radius_scale = static_cast<float>(radius_spin_->value());
    p.km_iters = lloyd_spin_->value();
    switch (backend_combo_->currentIndex()) {
        case 1:  p.km_backend = "cpu";  break;
        case 2:  p.km_backend = "cuda"; break;
        default: p.km_backend = "auto"; break;
    }
    p.random_seed = seed_spin_->value();
    p.total_iters = iters_spin_->value();
    // 学习率退火：索引与 OptimizeParams::lr_schedule 的取值一一对应
    static const char* kSchedules[] = {"cosine", "linear", "piecewise"};
    const int si = lr_schedule_combo_->currentIndex();
    p.lr_schedule = kSchedules[(si >= 0 && si < 3) ? si : 0];
    p.lr_start = static_cast<float>(lr_start_spin_->value());
    p.lr_end = static_cast<float>(lr_end_spin_->value());
    p.lr_warmup_iters = lr_warmup_spin_->value();
    // 模板模糊 sigma 退火：索引对应 OptimizeParams::blur_sigma_schedule
    static const char* kSigmaSched[] = {"cosine", "linear", "piecewise", "none"};
    const int ss = sigma_schedule_combo_->currentIndex();
    p.blur_sigma_schedule = kSigmaSched[(ss >= 0 && ss < 4) ? ss : 0];
    p.blur_sigma_start = static_cast<float>(sigma_start_spin_->value());
    p.blur_sigma_end = static_cast<float>(sigma_end_spin_->value());
    p.blur_sigma_warmup = sigma_warmup_spin_->value();
    p.darts_lock_ratio = static_cast<float>(darts_lock_spin_->value());
    return p;
}

// ── 运行 ────────────────────────────────────────────────────
void OptimizePage::start_optimization() {
    if (state_ == State::Running || image_path_.isEmpty()) return;

    OptimizeParams params = collect_params();

    VIN_INFO("optimize: img=%s N=%d total_iters=%d init=%s grid=%d alpha=%.2f rs=%.2f "
             "lloyd=%d backend=%s seed=%d template=%d canvas=%dx%d "
             "lr=%.3f->%.3f(%s) warmup=%d sigma=%.2f->%.2f(%s) sw=%d lock=%.2f",
             params.input_path.c_str(), params.num_shapes, params.total_iters,
             params.init_mode.c_str(), params.km_grid, params.km_alpha,
             params.km_radius_scale, params.km_iters, params.km_backend.c_str(),
             params.random_seed, static_cast<int>(params.template_mode),
             params.canvas_w, params.canvas_h,
             params.lr_start, params.lr_end, params.lr_schedule.c_str(),
             params.lr_warmup_iters,
             params.blur_sigma_start, params.blur_sigma_end,
             params.blur_sigma_schedule.c_str(), params.blur_sigma_warmup,
             params.darts_lock_ratio);

    QString dir = QStandardPaths::writableLocation(QStandardPaths::TempLocation);
    if (dir.isEmpty()) dir = QDir::tempPath();
    const QString out_dir = dir + "/vinylizer_out";
    QDir().mkpath(out_dir);
    const QString base = QFileInfo(image_path_).completeBaseName();
    const QString out_path =
        QString("%1/%2_N%3.json").arg(out_dir).arg(base).arg(params.num_shapes);
    output_json_path_ = out_path.toStdString();

    PipelineConfig cfg = build_pipeline_config(params, output_json_path_);

    // 优化器内部画布尺寸（与 pipeline.cpp/optimizer.cpp 同一公式；**含透明 padding**）
    // 与要显示的内容区窗口。pinned 缓冲必须按画布尺寸分配（D2H 整块写入）。
    int canvas_w = 0, canvas_h = 0, view_x = 0, view_y = 0, view_w = 0, view_h = 0;
    optimizer_canvas_view(params, canvas_w, canvas_h, view_x, view_y, view_w, view_h);

    preview_stack_->setCurrentWidget(gl_view_);
    if (!gl_view_->prepare(canvas_w, canvas_h)) {
        QMessageBox::warning(this, QString::fromUtf8("显存分配失败"),
                             QString::fromUtf8("无法为实时预览分配显存，请降低图片尺寸后重试。"));
        preview_stack_->setCurrentWidget(source_view_);
        return;
    }
    gl_view_->set_view_rect(view_x, view_y, view_w, view_h);
    cfg.viz_pinned_a = gl_view_->pinned(0);
    cfg.viz_event_a = gl_view_->event(0);
    cfg.viz_pinned_b = gl_view_->pinned(1);
    cfg.viz_event_b = gl_view_->event(1);
    cfg.viz_read_idx_atomic = gl_view_->read_idx();

    last_mse_ = 0.0f;
    iterating_ = false;
    result_image_ = QImage();
    result_view_->setPixmap(QPixmap());
    result_view_->setText(QString::fromUtf8("生成中…"));
    progress_bar_->setValue(0);
    running_mse_label_->setText(QString::fromUtf8("MSE: —"));
    status_label_->setText(QString::fromUtf8("正在初始化形状…"));
    abort_btn_->setEnabled(true);
    set_state(State::Running);

    worker_ = new OptimizeWorker(cfg, this);
    connect(worker_, &OptimizeWorker::progress, this, &OptimizePage::on_progress);
    connect(worker_, &OptimizeWorker::done, this, &OptimizePage::on_finished);
    connect(worker_, &QThread::finished, worker_, &QObject::deleteLater);
    worker_->start();
}

void OptimizePage::abort_optimization() {
    if (!worker_) return;
    status_label_->setText(QString::fromUtf8("正在停止…"));
    abort_btn_->setEnabled(false);
    worker_->request_stop();
}

void OptimizePage::on_progress(float pct, float mse, int iter) {
    progress_bar_->setValue(static_cast<int>(std::min(1.0f, std::max(0.0f, pct)) * 1000.0f));
    if (mse > 0.0f) {
        last_mse_ = mse;
        const float psnr = 10.0f * std::log10(255.0f * 255.0f / mse);
        running_mse_label_->setText(QString("MSE: %1    PSNR: %2 dB")
                                        .arg(mse, 0, 'f', 1)
                                        .arg(psnr, 0, 'f', 2));
        if (!iterating_) {
            iterating_ = true;
            status_label_->setText(QString::fromUtf8("优化中…"));
        }
    }
}

void OptimizePage::on_finished() {
    worker_ = nullptr;
    gl_view_->release_buffers();
    preview_stack_->setCurrentWidget(result_view_);

    if (last_mse_ > 0.0f) {
        const float psnr = 10.0f * std::log10(255.0f * 255.0f / last_mse_);
        done_info_label_->setText(QString::fromUtf8("最终 MSE: %1    PSNR: %2 dB")
                                      .arg(last_mse_, 0, 'f', 1)
                                      .arg(psnr, 0, 'f', 2));
    } else {
        done_info_label_->setText(QString::fromUtf8("已完成"));
    }

    set_state(State::Done);

    if (!output_json_path_.empty()) {
        const std::string jp = output_json_path_;
        QFuture<QImage> fut = QtConcurrent::run([jp]() { return render_preview_from_json(jp); });
        result_watcher_.setFuture(fut);
    }
}

void OptimizePage::on_result_preview_ready() {
    result_image_ = result_watcher_.result();
    if (result_image_.isNull()) {
        result_view_->setText(QString::fromUtf8("渲染预览失败"));
        return;
    }
    result_view_->setText(QString());
    refresh_views();
}

// ── 保存 / 跳转 ─────────────────────────────────────────────
void OptimizePage::save_png() {
    if (result_image_.isNull()) {
        QMessageBox::information(this, QString::fromUtf8("提示"),
                                 QString::fromUtf8("还没有可保存的图像。"));
        return;
    }
    const QString def = QFileInfo(image_path_).completeBaseName() +
                        QString("_N%1.png").arg(n_spin_->value());
    const QString path = QFileDialog::getSaveFileName(this, QString::fromUtf8("保存 PNG"), def,
                                                      "PNG (*.png)");
    if (path.isEmpty()) return;
    if (result_image_.save(path, "PNG")) {
        QMessageBox::information(this, QString::fromUtf8("已保存"), path);
    } else {
        QMessageBox::warning(this, QString::fromUtf8("保存失败"), path);
    }
}

void OptimizePage::save_json() {
    if (output_json_path_.empty()) return;
    const QString def = QFileInfo(image_path_).completeBaseName() +
                        QString("_N%1.json").arg(n_spin_->value());
    const QString path = QFileDialog::getSaveFileName(this, QString::fromUtf8("保存 shapes.json"),
                                                      def, "JSON (*.json)");
    if (path.isEmpty()) return;
    QFile::remove(path);
    if (QFile::copy(QString::fromStdString(output_json_path_), path)) {
        QMessageBox::information(this, QString::fromUtf8("已保存"), path);
    } else {
        QMessageBox::warning(this, QString::fromUtf8("保存失败"), path);
    }
}

void OptimizePage::goto_inject() {
    if (output_json_path_.empty()) return;
    emit goto_inject_page(QString::fromStdString(output_json_path_));
}

} // namespace vinylizer
