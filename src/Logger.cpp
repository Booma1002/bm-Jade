#include "header/Logger.hpp"
#include <filesystem>
using namespace zeza;
void Logger::log(LogLevel level, const std::string &msg) {
    log_counts[static_cast<int>(level)].fetch_add(1, std::memory_order_relaxed);

    LogMessage message;
    message.level = level;
    message.timestamp = std::chrono::system_clock::now();
    message.thread_id = std::this_thread::get_id();
    message.message = msg;

    {
        std::lock_guard<std::mutex> lock(mut);
        buffer.push(message);
    }
    cond.notify_one();
}

Logger &Logger::get() {
    static Logger instance;
    return instance;
}

void Logger::process_queue() {
    while (running || !buffer.empty()) {
        std::queue<LogMessage> hijacking_buffer;

        {
            std::unique_lock<std::mutex> lock(mut);
            // mesa semantics guard:
            // thread sleeps here until the buffer has data OR we are shutting down.
            cond.wait(lock, [this](){return !buffer.empty() || !running;});
            std::swap(hijacking_buffer, buffer);
        }

        // write the stolen data to the slow disk without holding any locks
        while (!hijacking_buffer.empty()) {
            const auto& msg = hijacking_buffer.front();
            std::string level_str;
            switch (msg.level) {
                case LogLevel::DEBUG: level_str = "DEBUG"; break;
                case LogLevel::INFO:  level_str = "INFO "; break;
                case LogLevel::WARN:  level_str = "WARN "; break;
                case LogLevel::ERR:   level_str = "ERROR"; break;
                case LogLevel::FATAL: level_str = "FATAL"; break;
            }

            std::time_t time_t_val = std::chrono::system_clock::to_time_t(msg.timestamp);
            std::tm* tm_info = std::localtime(&time_t_val);

            file << "[" << std::put_time(tm_info, "%Y-%m-%d %H:%M:%S") << "] "
                        << "[" << level_str << "] "
                        << "[Thread " << msg.thread_id << "] "
                        << msg.message << "\n";

            if (msg.level == LogLevel::ERR || msg.level == LogLevel::FATAL) {
                file.flush();
            }
            hijacking_buffer.pop();
        }
    }
}

Logger::Logger() : running(true){
    for (auto& count : log_counts) {
        count.store(0, std::memory_order_relaxed);
    }
    std::string log_dir = "logs";
    if (const char* env_dir = std::getenv("TILE_LOG_DIR")) {
        log_dir = env_dir;
    }
    std::filesystem::create_directories(log_dir);
    std::string log_file = log_dir + "/framework.log";

    file.open(log_file, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Fatal: Could not open framework.log for writing!" << std::endl;
    }
    worker = std::thread(&Logger::process_queue, this);
}

Logger::~Logger(){
    running = false;
    cond.notify_one();
    if (worker.joinable()) worker.join();

    if (file.is_open()) {
        file << "\n==================================================\n";
        file << "[SYSTEM] Execution Terminated. Telemetry Summary:\n";
        file << "  -> FATAL : " << log_counts[static_cast<int>(LogLevel::FATAL)].load(std::memory_order_relaxed) << "\n";
        file << "  -> ERROR : " << log_counts[static_cast<int>(LogLevel::ERR)].load(std::memory_order_relaxed) << "\n";
        file << "  -> WARN  : " << log_counts[static_cast<int>(LogLevel::WARN)].load(std::memory_order_relaxed) << "\n";
        file << "  -> INFO  : " << log_counts[static_cast<int>(LogLevel::INFO)].load(std::memory_order_relaxed) << "\n";
        file << "  -> DEBUG : " << log_counts[static_cast<int>(LogLevel::DEBUG)].load(std::memory_order_relaxed) << "\n";
        file << "==================================================\n\n";
        file.close();
    }
}



