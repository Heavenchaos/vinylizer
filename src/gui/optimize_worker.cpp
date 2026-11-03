#include "gui/optimize_worker.h"
#include "core/pipeline.h"
#include <cuda_runtime.h>
#include <QElapsedTimer>

namespace vinylizer {

OptimizeWorker::OptimizeWorker(const PipelineConfig& config, QObject* parent)
    : QThread(parent), config_(config) {}

void OptimizeWorker::run() {
    cudaSetDevice(0);

    QElapsedTimer progress_timer;
    progress_timer.start();

    auto cb = [this, &progress_timer](float pct, float mse, int iter) {
        if (stop_requested_) config_.stop_requested = true;
        if (progress_timer.elapsed() > 50) {
            emit progress(pct, mse, iter);
            progress_timer.start();
        }
    };

    run_pipeline(config_, cb);

    emit progress(1.0f, 0.0f, 0);
    emit done();
}

} // namespace vinylizer
