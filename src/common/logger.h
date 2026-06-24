#pragma once

#include <cstdio>
#include <cstdarg>
#include <string>
#include <chrono>
#include <mutex>

namespace vinylizer {

enum class LogLevel {
    TRACE,
    DEBUG,
    INFO,
    WARN,
    ERR_L
};

class Logger {
public:
    static Logger& instance() {
        static Logger logger;
        return logger;
    }

    void set_level(LogLevel level) { level_ = level; }

    void log(LogLevel level, const char* file, int line, const char* fmt, ...) {
        if (level < level_) return;

        static const char* tags[] = {"TRC", "DBG", "INF", "WRN", "ERR"};

        auto now = std::chrono::system_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        auto t = std::chrono::system_clock::to_time_t(now);
        struct tm tm_buf;
        localtime_s(&tm_buf, &t);

        char time_buf[32];
        std::snprintf(time_buf, sizeof(time_buf), "%02d:%02d:%02d.%03d",
                      tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec,
                      static_cast<int>(ms.count()));

        char msg_buf[1024];
        va_list args;
        va_start(args, fmt);
        std::vsnprintf(msg_buf, sizeof(msg_buf), fmt, args);
        va_end(args);

        // Extract filename from path
        const char* basename = file;
        for (const char* p = file; *p; ++p) {
            if (*p == '/' || *p == '\\') basename = p + 1;
        }

        std::lock_guard<std::mutex> lock(mutex_);
        std::fprintf(stderr, "[%s] [%s] %s:%d | %s\n",
                     time_buf, tags[static_cast<int>(level)],
                     basename, line, msg_buf);
        fflush(stderr);

        if (!log_file_opened_) {
            log_file_ = std::fopen("vinylizer.log", "w");
            if (log_file_) setbuf(log_file_, nullptr);
            log_file_opened_ = true;
        }
        if (log_file_) {
            std::fprintf(log_file_, "[%s] [%s] %s:%d | %s\n",
                         time_buf, tags[static_cast<int>(level)],
                         basename, line, msg_buf);
            fflush(log_file_);
        }
    }

    ~Logger() {
        if (log_file_) std::fclose(log_file_);
    }

private:
    Logger() = default;
    LogLevel level_ = LogLevel::INFO;
    std::mutex mutex_;
    FILE* log_file_ = nullptr;
    bool log_file_opened_ = false;
};

#define VIN_LOG(level, fmt, ...) \
    ::vinylizer::Logger::instance().log(::vinylizer::LogLevel::level, \
        __FILE__, __LINE__, fmt, ##__VA_ARGS__)

#define VIN_TRACE(fmt, ...) VIN_LOG(TRACE, fmt, ##__VA_ARGS__)
#define VIN_DEBUG(fmt, ...) VIN_LOG(DEBUG, fmt, ##__VA_ARGS__)
#define VIN_INFO(fmt, ...)  VIN_LOG(INFO,  fmt, ##__VA_ARGS__)
#define VIN_WARN(fmt, ...)  VIN_LOG(WARN,  fmt, ##__VA_ARGS__)
#define VIN_ERROR(fmt, ...) VIN_LOG(ERR_L, fmt, ##__VA_ARGS__)

} // namespace vinylizer
