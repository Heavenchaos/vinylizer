#pragma once

// 顶层窗口：左侧两个页签（矢量化 / FH6 注入），右侧 QStackedWidget。
// 页 1 生成完可以「带着 shapes.json」跳到页 2。

#include <QMainWindow>
#include <QString>

class QStackedWidget;
class QPushButton;

namespace vinylizer {

class OptimizePage;
class InjectPage;

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);

    void goto_inject_with(const QString& json_path);

private:
    void switch_page(int index);

    QStackedWidget* stack_ = nullptr;
    QPushButton* nav_optimize_ = nullptr;
    QPushButton* nav_inject_ = nullptr;
    OptimizePage* optimize_page_ = nullptr;
    InjectPage* inject_page_ = nullptr;
};

} // namespace vinylizer
