#pragma once

// 优化过程中的实时预览：GPU 直接把画布 D2H 拷进 pinned memory（双缓冲），
// GUI 线程轮询共享原子拿到「已完成的帧号」，再 glTexSubImage2D 上传纹理。
// 这里只负责「纹理 + pinned 双缓冲」这一层，不参与任何参数/状态逻辑。

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLBuffer>
#include <QOpenGLVertexArrayObject>
#include <QTimer>
#include <atomic>
#include <cstdint>

namespace vinylizer {

class PreviewGLWidget : public QOpenGLWidget, protected QOpenGLFunctions {
    Q_OBJECT
public:
    explicit PreviewGLWidget(QWidget* parent = nullptr);
    ~PreviewGLWidget() override;

    // 分配 pinned 双缓冲 + cudaEvent + 共享原子（纯 CUDA，不需要 GL context）。
    // 尺寸必须是优化器内部 *画布* 尺寸（W_opt x H_opt，含透明 padding）——
    // D2H 每次整块写入 canvas_w*canvas_h*3 字节。
    bool prepare(int canvas_w, int canvas_h);
    // 设置要显示的"窗口"（画布内的内容区/原图区域）。纹理按该尺寸创建，上传时用
    // GL_UNPACK_ROW_LENGTH/SKIP_PIXELS/SKIP_ROWS 从整块画布里取子矩形。
    void set_view_rect(int x, int y, int w, int h);
    // 释放 pinned/event/原子；GL 纹理在析构时随 context 释放。
    void release_buffers();

    uint8_t* pinned(int i) const { return pinned_[i]; }
    void* event(int i) const { return events_[i]; }
    std::atomic<int>* read_idx() const { return read_idx_; }
    bool ready() const { return canvas_w_ > 0 && canvas_h_ > 0 && pinned_[0] && pinned_[1]; }

protected:
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;

private:
    void init_shader();
    void ensure_texture();
    void destroy_texture();
    // 若原子指示有新帧则上传，返回是否上传成功
    bool upload_frame();

    int canvas_w_ = 0, canvas_h_ = 0;      // pinned 缓冲 = 优化器画布（含 padding）
    int view_x_ = 0, view_y_ = 0;          // 内容区在画布内的偏移
    int view_w_ = 0, view_h_ = 0;          // 内容区尺寸 = 纹理尺寸
    int tex_w_ = 0, tex_h_ = 0;            // 当前纹理实际尺寸
    bool has_frame_ = false;               // 是否已经显示过至少一帧（否则显示"等待首帧"）
    uint8_t* pinned_[2] = {nullptr, nullptr};
    void* events_[2] = {nullptr, nullptr};
    std::atomic<int>* read_idx_ = nullptr;

    QOpenGLShaderProgram* prog_ = nullptr;
    QOpenGLBuffer vbo_{QOpenGLBuffer::VertexBuffer};
    QOpenGLVertexArrayObject vao_;
    unsigned int tex_ = 0;
    QTimer poll_timer_;
};

} // namespace vinylizer
