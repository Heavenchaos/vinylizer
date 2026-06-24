#pragma once

#include <QThread>
#include <QImage>
#include <string>
#include <atomic>
#include "core/pipeline.h"

namespace vinylizer {

class OptimizeWorker : public QThread {
    Q_OBJECT
public:
    OptimizeWorker(const PipelineConfig& config, QObject* parent = nullptr);
    void run() override;
    void request_stop() { stop_requested_ = true; }

signals:
    void progress(float pct, float mse, int iter);
    void frame_ready(QImage frame);
    void done();

private:
    PipelineConfig config_;
    std::atomic<bool> stop_requested_{false};
};

} // namespace vinylizer
