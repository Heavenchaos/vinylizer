#pragma once

#include <QThread>
#include <QMutex>
#include <QWaitCondition>
#include <string>
#include "inject/inject.h"

namespace vinylizer {

class InjectWorker : public QThread {
    Q_OBJECT
public:
    InjectWorker(const InjectConfig& config, QObject* parent = nullptr);
    void run() override;
    void confirm(bool yes);

signals:
    void log_message(QString msg, bool is_error);
    void confirm_requested();
    void done(bool success);

private:
    InjectConfig config_;
    QMutex mutex_;
    QWaitCondition wait_cond_;
    bool confirmed_ = false;
    bool cancelled_ = false;
};

} // namespace vinylizer
