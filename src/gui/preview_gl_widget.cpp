#include "gui/preview_gl_widget.h"
#include "common/logger.h"

#include <QOpenGLContext>
#include <QPainter>
#include <cuda_runtime.h>
#include <algorithm>

namespace vinylizer {

PreviewGLWidget::PreviewGLWidget(QWidget* parent) : QOpenGLWidget(parent) {
    connect(&poll_timer_, &QTimer::timeout, this, [this]() { update(); });
    poll_timer_.start(33);  // ~30fps 轮询
}

PreviewGLWidget::~PreviewGLWidget() {
    poll_timer_.stop();
    if (context()) {
        makeCurrent();
        destroy_texture();
        vao_.destroy();
        vbo_.destroy();
        if (prog_) { delete prog_; prog_ = nullptr; }
        doneCurrent();
    }
    release_buffers();
}

bool PreviewGLWidget::prepare(int canvas_w, int canvas_h) {
    release_buffers();
    if (canvas_w <= 0 || canvas_h <= 0) return false;

    const size_t bytes = static_cast<size_t>(canvas_w) * static_cast<size_t>(canvas_h) * 3;
    cudaError_t ea = cudaMallocHost(&pinned_[0], bytes);
    cudaError_t eb = cudaMallocHost(&pinned_[1], bytes);
    if (ea != cudaSuccess || eb != cudaSuccess) {
        VIN_ERROR("PreviewGLWidget: cudaMallocHost failed A=%s B=%s",
                  cudaGetErrorString(ea), cudaGetErrorString(eb));
        if (ea == cudaSuccess) cudaFreeHost(pinned_[0]);
        pinned_[0] = pinned_[1] = nullptr;
        return false;
    }
    cudaEvent_t ev_a = nullptr, ev_b = nullptr;
    cudaEventCreate(&ev_a);
    cudaEventCreate(&ev_b);
    events_[0] = ev_a;
    events_[1] = ev_b;
    read_idx_ = new std::atomic<int>(-1);

    canvas_w_ = canvas_w;
    canvas_h_ = canvas_h;
    // 默认显示整块画布（调用方一般会用 set_view_rect 收窄到内容区）
    view_x_ = 0; view_y_ = 0; view_w_ = canvas_w; view_h_ = canvas_h;
    has_frame_ = false;
    VIN_INFO("PreviewGLWidget: pinned buffers ready %dx%d (%zu bytes each)",
             canvas_w, canvas_h, bytes);
    update();
    return true;
}

void PreviewGLWidget::set_view_rect(int x, int y, int w, int h) {
    if (canvas_w_ <= 0 || canvas_h_ <= 0) return;   // prepare() 之后才有效
    x = std::max(0, std::min(x, canvas_w_ - 1));
    y = std::max(0, std::min(y, canvas_h_ - 1));
    view_x_ = x;
    view_y_ = y;
    view_w_ = std::max(1, std::min(w, canvas_w_ - x));
    view_h_ = std::max(1, std::min(h, canvas_h_ - y));
    VIN_INFO("PreviewGLWidget: view rect [%d,%d)+%dx%d within canvas %dx%d",
             view_x_, view_y_, view_w_, view_h_, canvas_w_, canvas_h_);
    update();
}

void PreviewGLWidget::release_buffers() {
    for (int i = 0; i < 2; ++i) {
        if (events_[i]) {
            cudaEventDestroy(static_cast<cudaEvent_t>(events_[i]));
            events_[i] = nullptr;
        }
        if (pinned_[i]) {
            cudaFreeHost(pinned_[i]);
            pinned_[i] = nullptr;
        }
    }
    delete read_idx_;
    read_idx_ = nullptr;
    canvas_w_ = canvas_h_ = 0;
    view_x_ = view_y_ = view_w_ = view_h_ = 0;
    has_frame_ = false;
}

void PreviewGLWidget::init_shader() {
    prog_ = new QOpenGLShaderProgram(this);
    prog_->addShaderFromSourceCode(QOpenGLShader::Vertex,
        "attribute vec2 a_pos;\n"
        "attribute vec2 a_uv;\n"
        "varying vec2 v_uv;\n"
        "void main() {\n"
        "  gl_Position = vec4(a_pos, 0.0, 1.0);\n"
        "  v_uv = a_uv;\n"
        "}\n");
    prog_->addShaderFromSourceCode(QOpenGLShader::Fragment,
        "varying vec2 v_uv;\n"
        "uniform sampler2D u_tex;\n"
        "void main() {\n"
        "  gl_FragColor = texture2D(u_tex, v_uv);\n"
        "}\n");
    prog_->bindAttributeLocation("a_pos", 0);
    prog_->bindAttributeLocation("a_uv", 1);
    if (!prog_->link()) {
        VIN_ERROR("PreviewGLWidget: shader link failed: %s", prog_->log().toUtf8().constData());
        return;
    }

    // 注意：顶点位置必须是 NDC 的 [-1,1]（旧代码用 0..1，导致画面只占控件右上角
    // 四分之一的区域，其余部分是未清屏的"黑"——表现为"实时预览全黑"）。
    static const float kVerts[] = {
        -1.0f, -1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 1.0f,
        -1.0f,  1.0f,  0.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 0.0f,
    };
    vbo_.create();
    vbo_.bind();
    vbo_.allocate(kVerts, sizeof(kVerts));
    vbo_.release();

    vao_.create();
    vao_.bind();
    vbo_.bind();
    prog_->bind();
    prog_->enableAttributeArray(0);
    prog_->enableAttributeArray(1);
    prog_->setAttributeBuffer(0, GL_FLOAT, 0, 2, 16);
    prog_->setAttributeBuffer(1, GL_FLOAT, 8, 2, 16);
    vao_.release();
    vbo_.release();
    prog_->release();
}

void PreviewGLWidget::initializeGL() {
    initializeOpenGLFunctions();
    glClearColor(0.059f, 0.059f, 0.137f, 1.0f);
    init_shader();
    ensure_texture();
}

void PreviewGLWidget::resizeGL(int, int) {}

void PreviewGLWidget::ensure_texture() {
    if (view_w_ <= 0 || view_h_ <= 0) return;
    if (tex_ != 0 && (tex_w_ != view_w_ || tex_h_ != view_h_)) destroy_texture();
    if (tex_ != 0) return;
    glGenTextures(1, &tex_);
    glBindTexture(GL_TEXTURE_2D, tex_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, view_w_, view_h_, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);
    tex_w_ = view_w_;
    tex_h_ = view_h_;
}

void PreviewGLWidget::destroy_texture() {
    if (tex_ != 0) {
        glDeleteTextures(1, &tex_);
        tex_ = 0;
    }
    tex_w_ = tex_h_ = 0;
}

bool PreviewGLWidget::upload_frame() {
    if (tex_ == 0 || !read_idx_) return false;
    int idx = read_idx_->load();
    if (idx < 0 || idx > 1) return false;
    read_idx_->store(-1);

    // 源是整块画布（canvas_w_ × canvas_h_），只取内容区子矩形上传；
    // 指针给画布起点，偏移交给 UNPACK_SKIP_*，行距交给 UNPACK_ROW_LENGTH。
    glBindTexture(GL_TEXTURE_2D, tex_);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, canvas_w_);
    glPixelStorei(GL_UNPACK_SKIP_PIXELS, view_x_);
    glPixelStorei(GL_UNPACK_SKIP_ROWS, view_y_);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, view_w_, view_h_, GL_RGB, GL_UNSIGNED_BYTE, pinned_[idx]);
    glPixelStorei(GL_UNPACK_SKIP_ROWS, 0);
    glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glBindTexture(GL_TEXTURE_2D, 0);
    return true;
}

void PreviewGLWidget::paintGL() {
    qreal dpr = devicePixelRatioF();
    if (dpr < 1.0) dpr = 1.0;
    const int fb_w = std::max(1, static_cast<int>(width() * dpr));
    const int fb_h = std::max(1, static_cast<int>(height() * dpr));

    // 清屏（并显式关掉 scissor/blend/depth：Qt 的 paint engine 可能留下这些状态，
    // 导致 clear 只覆盖裁剪区，出现"半屏脏帧"）
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
    glViewport(0, 0, fb_w, fb_h);
    glClearColor(0.059f, 0.059f, 0.137f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    if (ready()) {
        ensure_texture();
        if (upload_frame()) has_frame_ = true;
    }

    // 首帧到达之前（init/KM 阶段）纹理内容是未定义的：不要画它，改成提示文字，
    // 否则玩家看到的是"全黑预览"（无法区分"正在初始化"和"出错了"）。
    if (!has_frame_ || tex_ == 0 || !prog_ || !prog_->isLinked()) {
        QPainter painter(this);
        painter.setPen(QColor("#a0a0b0"));
        painter.drawText(rect(), Qt::AlignCenter, QString::fromUtf8("正在初始化形状…等待第一帧"));
        return;
    }

    // letterbox：保持画布宽高比，居中显示
    const float s = std::min(static_cast<float>(fb_w) / static_cast<float>(view_w_),
                             static_cast<float>(fb_h) / static_cast<float>(view_h_));
    const int dw = std::max(1, static_cast<int>(view_w_ * s));
    const int dh = std::max(1, static_cast<int>(view_h_ * s));
    glViewport((fb_w - dw) / 2, (fb_h - dh) / 2, dw, dh);

    prog_->bind();
    vao_.bind();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_);
    prog_->setUniformValue("u_tex", 0);
    glDisable(GL_BLEND);
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_DEPTH_TEST);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindTexture(GL_TEXTURE_2D, 0);
    vao_.release();
    prog_->release();
}

} // namespace vinylizer
