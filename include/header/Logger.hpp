#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <chrono>
#include <array>
namespace zeza {
    enum class LogLevel {
        DEBUG,
        INFO,
        WARN,
        ERR,
        FATAL
    };

    struct LogMessage {
        LogLevel level;
        std::chrono::system_clock::time_point timestamp;
        std::thread::id thread_id;
        std::string message;
    };

    class Logger {
    private:
        // producer-consumer buffer
        std::queue<LogMessage> buffer;
        std::mutex mut;
        std::condition_variable cond;
        std::thread worker;
        std::atomic<bool> running;
        std::array<std::atomic<uint64_t>, 5> log_counts;
        std::ofstream file;

        Logger();
        ~Logger();

        // background infinite loop
        void process_queue();
        LogLevel current_level = LogLevel::DEBUG;
    public:
        // Singleton Access
        void set_level(LogLevel level) { current_level = level; }
        LogLevel get_level() const { return current_level; }
        static Logger& get();

        // The frontend pusher
        void log(LogLevel level, const std::string& msg);

        // Deleted copy semantics
        Logger(const Logger&) = delete;
        Logger& operator=(const Logger&) = delete;
    };

#define LOG_DEBUG(msg) if (Logger::get().get_level() <= LogLevel::DEBUG) Logger::get().log(LogLevel::DEBUG, msg)
#define LOG_INFO(msg)  if (Logger::get().get_level() <= LogLevel::INFO)  Logger::get().log(LogLevel::INFO, msg)
#define LOG_WARN(msg)  if (Logger::get().get_level() <= LogLevel::WARN)  Logger::get().log(LogLevel::WARN, msg)
#define LOG_ERR(msg)   if (Logger::get().get_level() <= LogLevel::ERR)   Logger::get().log(LogLevel::ERR, std::string(__FILE__) + ":" + std::to_string(__LINE__) + " " + msg)
#define LOG_FATAL(msg) if (Logger::get().get_level() <= LogLevel::FATAL) Logger::get().log(LogLevel::FATAL, msg)

}// namespace zeza