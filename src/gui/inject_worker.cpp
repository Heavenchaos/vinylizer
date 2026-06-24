#include "gui/inject_worker.h"
#include "common/logger.h"
#include "common/canvas.h"
#include "output/json_writer.h"

#include <QMutexLocker>

namespace vinylizer {

InjectWorker::InjectWorker(const InjectConfig& config, QObject* parent)
    : QThread(parent), config_(config) {}

void InjectWorker::run() {
    // 1. Validate JSON
    emit log_message(QString::fromUtf8("正在验证 JSON 文件..."), false);
    auto shapes = validate_json(config_.json_path);
    if (shapes.empty()) {
        emit log_message(QString::fromUtf8("JSON 验证失败"), true);
        emit done(false);
        return;
    }
    emit log_message(QString("JSON validated: %1 shapes").arg(shapes.size()), false);

    int layer_count = config_.layer_count > 0 ? config_.layer_count : static_cast<int>(shapes.size());

    // 2. Find FH6 process
    emit log_message(QString::fromUtf8("正在查找 FH6 进程..."), false);
    int pid = config_.process_id;
    std::string proc_name;
    if (pid == 0) {
        auto [found_pid, found_name] = find_fh6_process();
        pid = found_pid;
        proc_name = found_name;
    }
    if (pid == 0) {
        emit log_message(QString::fromUtf8("未找到 FH6 进程"), true);
        emit done(false);
        return;
    }
    emit log_message(QString("Found FH6: PID=%1").arg(pid), false);

    // Open process handle ONCE for all subsequent operations
#ifdef _WIN32
    HANDLE h = OpenProcess(PROCESS_VM_READ | PROCESS_VM_WRITE |
                             PROCESS_VM_OPERATION | PROCESS_QUERY_INFORMATION,
                             FALSE, pid);
    if (!h) {
        emit log_message(QString::fromUtf8("无法打开 FH6 进程 (权限不足)"), true);
        emit done(false);
        return;
    }
#else
    void* h = nullptr;
    emit log_message(QString::fromUtf8("Inject only supported on Windows"), true);
    emit done(false);
    return;
#endif

    // 3. Locate layer table
    emit log_message(QString::fromUtf8("正在定位 layer table（可能需要一些时间）..."), false);
    uint64_t table_address = locate_layer_table(pid, layer_count);
    if (table_address == 0) {
        emit log_message(QString::fromUtf8("Layer table 定位失败"), true);
        CloseHandle(h);
        emit done(false);
        return;
    }
    emit log_message(QString("Found layer table at 0x%1").arg(table_address, 0, 16).toUpper(), false);

    // 4. Statistics and sampling for user confirmation
    auto stats = count_valid_layers(pid, table_address, layer_count);
    emit log_message(QString::fromUtf8("层表统计: 总共 %1 个有效图层, 椭圆(102): %2 个, 矩形(101): %3 个")
                         .arg(stats.total_valid).arg(stats.count_102).arg(stats.count_101), false);

    emit log_message(QString::fromUtf8("随机采样 6 个图层:"), false);
    auto samples = sample_layers(pid, table_address, layer_count, 6);
    for (const auto& s : samples) {
        const char* shape_name = (s.shape_id == 102) ? "ELLIPSE" : (s.shape_id == 101) ? "RECT" : "OTHER";
        emit log_message(QString("  层[%1]  pos=(%2, %3)  scale=(%4, %5)  rot=%6  rgba=(%7,%8,%9,%10)  shape=%11(%12)  mask=%13")
                             .arg(s.index)
                             .arg(s.pos_x, 0, 'f', 2).arg(s.pos_y, 0, 'f', 2)
                             .arg(s.scale_x, 0, 'f', 4).arg(s.scale_y, 0, 'f', 4)
                             .arg(s.rotation, 0, 'f', 1)
                             .arg(s.color[0]).arg(s.color[1]).arg(s.color[2]).arg(s.color[3])
                             .arg(s.shape_id).arg(shape_name)
                             .arg(s.mask), false);
    }

    emit log_message(QString::fromUtf8("即将导入 %1 个 shape 到 %2 层模板中").arg(shapes.size()).arg(layer_count), false);
    emit log_message(QString::fromUtf8("多余的 %1 层将被清零").arg(layer_count - static_cast<int>(shapes.size())), false);
    emit log_message(QString::fromUtf8("请确认以上图层信息是否为您在 FH6 中创建的占位图形"), false);
    emit log_message(QString::fromUtf8("输入 Y 确认导入，输入其他内容取消"), false);

    emit confirm_requested();

    // Wait for confirmation
    {
        QMutexLocker locker(&mutex_);
        wait_cond_.wait(&mutex_, 30000); // 30s timeout
        if (cancelled_ || !confirmed_) {
            if (!cancelled_) {
                cancelled_ = true;
            }
            CloseHandle(h);
            emit done(false);
            return;
        }
    }

    // 5. Write shapes (reuse handle for all layers)
    emit log_message(QString::fromUtf8("开始写入..."), false);
    int success = 0, fail = 0;
    int write_count = std::min(static_cast<int>(shapes.size()), layer_count);

    for (int i = 0; i < layer_count; ++i) {
        bool ok;
        if (i < write_count) {
            ok = write_layer_h(h, table_address, i, shapes[i]);
        } else {
            ok = write_clear_layer_h(h, table_address, i);
        }
        if (ok) success++; else fail++;

        // Progress every 100 layers
        if (i % 100 == 0 || i == layer_count - 1) {
            emit log_message(QString("写入进度: %1 / %2").arg(i + 1).arg(layer_count), false);
        }

        if (fail >= 10 && success == 0) {
            emit log_message(QString::fromUtf8("写入失败过多，中止"), true);
            CloseHandle(h);
            emit done(false);
            return;
        }
    }

    CloseHandle(h);

    emit log_message(QString("Inject complete: %1 success, %2 fail").arg(success).arg(fail), fail > 0);
    emit done(fail == 0);
}

void InjectWorker::confirm(bool yes) {
    QMutexLocker locker(&mutex_);
    if (yes) {
        confirmed_ = true;
    } else {
        cancelled_ = true;
    }
    wait_cond_.wakeAll();
}

} // namespace vinylizer
