#pragma once

// 页 2「FH6 注入」：选 shapes.json → 填 FH6 里摆放的占位图层数 → 注入。
// 注入前会把 FH6 图层表统计 + 随机采样 6 个图层打到日志里，并弹 Y/N 确认。

#include <QWidget>
#include <QPointer>
#include <QString>
#include <QImage>
#include <QFutureWatcher>

class QLabel;
class QPushButton;
class QSpinBox;
class QPlainTextEdit;

namespace vinylizer {

class InjectWorker;

class InjectPage : public QWidget {
    Q_OBJECT
public:
    explicit InjectPage(QWidget* parent = nullptr);
    ~InjectPage() override;

    // 从页 1 直接带 shapes.json 过来
    void set_json_path(const QString& path);

private slots:
    void choose_json();
    void start_inject();
    void on_log(QString msg, bool is_error);
    void on_confirm_requested();
    void on_finished(bool success);
    void on_preview_ready();
    void clear_log();

private:
    QString json_path_;
    bool running_ = false;

    QPushButton* json_btn_ = nullptr;
    QLabel* json_label_ = nullptr;
    QSpinBox* layer_spin_ = nullptr;
    QLabel* preview_label_ = nullptr;
    QPushButton* inject_btn_ = nullptr;
    QPushButton* clear_btn_ = nullptr;
    QPlainTextEdit* log_view_ = nullptr;

    QImage preview_image_;
    QFutureWatcher<QImage> preview_watcher_;
    QPointer<InjectWorker> worker_;
};

} // namespace vinylizer
