#pragma once

#include <chrono>
#include <cstdio>

namespace vinylizer {

class Timer {
public:
    explicit Timer(const char* label = "", bool auto_report = true)
        : label_(label), auto_report_(auto_report),
          start_(std::chrono::high_resolution_clock::now()) {}

    double elapsed_ms() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now - start_).count();
    }

    double elapsed_s() const {
        return elapsed_ms() / 1000.0;
    }

    void report() const {
        std::fprintf(stderr, "[%s] %.3fs\n", label_, elapsed_s());
    }

    void set_auto_report(bool v) { auto_report_ = v; }

    ~Timer() { if (auto_report_) report(); }

private:
    const char* label_;
    bool auto_report_;
    std::chrono::high_resolution_clock::time_point start_;
};

} // namespace vinylizer
