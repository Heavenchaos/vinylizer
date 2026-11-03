#pragma once

#include <cstdio>
#include <cstdarg>
#include <cstring>
#include <string>
#include <chrono>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <atomic>
#include <array>

namespace vinylizer {

enum class LogLevel {
    TRACE,
    DEBUG,
    INFO,
    WARN,
    ERR_L
};

// 原同步 Logger（保留以兼容其他潜在引用，但 VIN_* 宏已改用 AsyncLogger）
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
#ifdef _WIN32
        localtime_s(&tm_buf, &t);
#else
        localtime_r(&t, &tm_buf);
#endif

        char time_buf[32];
        std::snprintf(time_buf, sizeof(time_buf), "%02d:%02d:%02d.%03d",
                      tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec,
                      static_cast<int>(ms.count()));

        char msg_buf[1024];
        va_list args;
        va_start(args, fmt);
        std::vsnprintf(msg_buf, sizeof(msg_buf), fmt, args);
        va_end(args);

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

// ============================================================
// AsyncLogger: 异步日志（lock-free 单生产者 + 后台消费者线程）
// ============================================================
// 设计目标：
//   - hot loop 调用 log() 不阻塞（无 mutex、无系统调用、无 I/O）
//   - 时间戳/格式化在调用线程完成，仅 memcpy 到 ring buffer
//   - 后台线程批量 flush 到 stderr + log_file
//   - 缓冲区满时丢弃最旧条目并计数（hot loop 不能阻塞）
//   - VIN_ERROR 触发同步 flush（致命错误立即可见）
//   - 进程退出时析构 flush 剩余日志

class AsyncLogger {
public:
    static AsyncLogger& instance() {
        static AsyncLogger logger;
        return logger;
    }

    void set_level(LogLevel level) { level_ = level; }

    void log(LogLevel level, const char* file, int line, const char* fmt, ...) {
        if (level < level_.load(std::memory_order_relaxed)) return;

        // 格式化完整一行（含时间戳），在调用线程完成
        char full_buf[1280];

        auto now = std::chrono::system_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        auto t = std::chrono::system_clock::to_time_t(now);
        struct tm tm_buf;
#ifdef _WIN32
        localtime_s(&tm_buf, &t);
#else
        localtime_r(&t, &tm_buf);
#endif

        char time_buf[32];
        std::snprintf(time_buf, sizeof(time_buf), "%02d:%02d:%02d.%03d",
                      tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec,
                      static_cast<int>(ms.count()));

        static const char* tags[] = {"TRC", "DBG", "INF", "WRN", "ERR"};
        const char* basename = file;
        for (const char* p = file; *p; ++p) {
            if (*p == '/' || *p == '\\') basename = p + 1;
        }

        int prefix_len = std::snprintf(full_buf, sizeof(full_buf), "[%s] [%s] %s:%d | ",
                                        time_buf, tags[static_cast<int>(level)],
                                        basename, line);

        va_list args;
        va_start(args, fmt);
        int msg_len = std::vsnprintf(full_buf + prefix_len,
                                      sizeof(full_buf) - prefix_len - 1, fmt, args);
        va_end(args);

        if (msg_len < 0) msg_len = 0;
        int total_len = prefix_len + msg_len;
        if (total_len >= (int)sizeof(full_buf) - 1) {
            total_len = (int)sizeof(full_buf) - 2;
        }
        full_buf[total_len++] = '\n';
        full_buf[total_len] = '\0';

        // 写入 ring buffer（lock-free 单生产者）
        uint64_t slot = write_idx_.fetch_add(1, std::memory_order_relaxed);
        uint64_t idx = slot & (CAPACITY - 1);

        // 等待槽位空闲（state==1 表示消费者正在读，极少发生）
        // 使用 spin-wait，因为这种情况几乎不会出现
        int spin = 0;
        while (entries_[idx].state.load(std::memory_order_acquire) == 1) {
            if (++spin > 1000) {
                // 极端情况：消费者卡死，丢弃本条日志
                dropped_count_.fetch_add(1, std::memory_order_relaxed);
                return;
            }
            std::this_thread::yield();
        }

        // 如果 state==2（ready），说明消费者还没读，覆盖会丢失数据
        // 计数但不阻塞（hot loop 优先于日志完整性）
        if (entries_[idx].state.load(std::memory_order_acquire) == 2) {
            dropped_count_.fetch_add(1, std::memory_order_relaxed);
        }

        std::memcpy(entries_[idx].buf, full_buf, total_len);
        entries_[idx].len = total_len;
        entries_[idx].state.store(2, std::memory_order_release);  // 2 = ready

        cv_.notify_one();
    }

    // 同步 flush 所有待写条目（用于 VIN_ERROR 或进程退出）
    void flush() {
        std::unique_lock<std::mutex> lk(flush_mutex_);
        drain_all_locked();
        if (log_file_) fflush(log_file_);
        fflush(stderr);
    }

    ~AsyncLogger() {
        // 停止后台线程
        stop_.store(true, std::memory_order_release);
        cv_.notify_all();
        if (worker_.joinable()) worker_.join();
        // flush 剩余日志
        flush();
        if (log_file_) std::fclose(log_file_);
    }

private:
    AsyncLogger() {
        log_file_ = std::fopen("vinylizer.log", "w");
        if (log_file_) setbuf(log_file_, nullptr);
        worker_ = std::thread(&AsyncLogger::worker_loop, this);
    }

    void worker_loop() {
        while (!stop_.load(std::memory_order_acquire)) {
            {
                std::unique_lock<std::mutex> lk(flush_mutex_);
                cv_.wait_for(lk, std::chrono::milliseconds(50));
            }
            flush();
        }
        // 最后一次 flush
        flush();
    }

    // 调用者持有 flush_mutex_
    void drain_all_locked() {
        while (true) {
            uint64_t r = read_idx_.load(std::memory_order_relaxed);
            uint64_t w = write_idx_.load(std::memory_order_relaxed);
            if (r >= w) break;
            uint64_t idx = r & (CAPACITY - 1);
            if (entries_[idx].state.load(std::memory_order_acquire) != 2) break;

            std::fwrite(entries_[idx].buf, 1, entries_[idx].len, stderr);
            if (log_file_) std::fwrite(entries_[idx].buf, 1, entries_[idx].len, log_file_);

            entries_[idx].state.store(0, std::memory_order_release);  // 0 = empty
            read_idx_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    struct LogEntry {
        char buf[1280];
        int len;
        std::atomic<int> state;  // 0=empty, 1=reading, 2=ready
        LogEntry() : len(0), state(0) {}
    };

    static constexpr size_t CAPACITY = 4096;  // 必须是 2 的幂

    std::atomic<LogLevel> level_{LogLevel::INFO};
    LogEntry entries_[CAPACITY];
    std::atomic<uint64_t> write_idx_{0};
    std::atomic<uint64_t> read_idx_{0};
    std::atomic<bool> stop_{false};
    std::atomic<uint64_t> dropped_count_{0};
    std::thread worker_;
    std::mutex flush_mutex_;
    std::condition_variable cv_;
    FILE* log_file_ = nullptr;
};

} // namespace vinylizer

#ifdef __NODEBUG__
    // 日志完全禁用：宏展开为空，编译器零开销消除
    #define VIN_LOG(level, fmt, ...) ((void)0)
    #define VIN_TRACE(fmt, ...) ((void)0)
    #define VIN_DEBUG(fmt, ...) ((void)0)
    #define VIN_INFO(fmt, ...)  ((void)0)
    #define VIN_WARN(fmt, ...)  ((void)0)
    // VIN_ERROR 始终保留，致命错误必须可见，并触发同步 flush
    #define VIN_ERROR(fmt, ...) do { \
        ::vinylizer::AsyncLogger::instance().log(::vinylizer::LogLevel::ERR_L, \
            __FILE__, __LINE__, fmt, ##__VA_ARGS__); \
        ::vinylizer::AsyncLogger::instance().flush(); \
    } while(0)
#else
    #define VIN_LOG(level, fmt, ...) \
        ::vinylizer::AsyncLogger::instance().log(::vinylizer::LogLevel::level, \
            __FILE__, __LINE__, fmt, ##__VA_ARGS__)
    #define VIN_TRACE(fmt, ...) VIN_LOG(TRACE, fmt, ##__VA_ARGS__)
    #define VIN_DEBUG(fmt, ...) VIN_LOG(DEBUG, fmt, ##__VA_ARGS__)
    #define VIN_INFO(fmt, ...)  VIN_LOG(INFO,  fmt, ##__VA_ARGS__)
    #define VIN_WARN(fmt, ...)  VIN_LOG(WARN,  fmt, ##__VA_ARGS__)
    // VIN_ERROR 触发同步 flush，保证致命错误立即可见
    #define VIN_ERROR(fmt, ...) do { \
        ::vinylizer::AsyncLogger::instance().log(::vinylizer::LogLevel::ERR_L, \
            __FILE__, __LINE__, fmt, ##__VA_ARGS__); \
        ::vinylizer::AsyncLogger::instance().flush(); \
    } while(0)
#endif
