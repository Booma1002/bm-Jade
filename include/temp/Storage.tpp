#pragma once
namespace zeza {
    template<typename T>
    T *Storage::data() {
        return static_cast<T *>(memory);
    }

    template<typename T>
    const T *Storage::data() const {
        return static_cast<const T *>(memory);
    }

    template<typename T>
    T Storage::get(size_t i) const {
        if (device != Device::CPU) {
            std::string msg = "Cannot read GPU memory directly from CPU";
            LOG_FATAL(msg);
            throw std::runtime_error(msg);
        }
        if (i * item_size >= capacity_bytes) {
            std::string msg = "Storage bounds error";
            LOG_FATAL(msg);
            throw std::out_of_range(msg);
        }
        return static_cast<T *>(memory)[i];
    }

    template<typename T>
    void Storage::set(size_t i, T val) {
        if (device != Device::CPU) {
            std::string msg = "Cannot read GPU memory directly from CPU";
            LOG_FATAL(msg);
            throw std::runtime_error(msg);
        }
        if (i * item_size >= capacity_bytes) {
            std::string msg = "Storage bounds error";
            LOG_FATAL(msg);
            throw std::out_of_range(msg);
        }
        static_cast<T *>(memory)[i] = val;
    }


}