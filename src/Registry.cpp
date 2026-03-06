#include "header/Registry.hpp"
using namespace zeza;

class RegistryException : public std::exception {
    std::string msg;
public:
    RegistryException(const std::string &message) : msg(message) {}
    virtual const char *what() const noexcept override { return msg.c_str(); }
};

Registry::Registry() {
    for(auto & op : kernel_table)
        for(auto & dev : op)
            dev = nullptr;
}

Registry& Registry::get() {
    static Registry instance;
    return instance;
}

void Registry::register_kernel(OpCode op, Device dev, Kernel func) {
    auto op_id = static_cast<uint32_t>(op);
    auto dev_id = static_cast<uint8_t>(dev);
    if (op_id >= static_cast<uint32_t>(OpCode::MAX_OPS))
    {
        std::string msg ="[Registry] Error: OpCode index out of bounds during registration.";
        LOG_FATAL(msg);
        throw RegistryException(msg);
    }

    if (dev_id >= static_cast<uint8_t>(Device::MAX_DEVICES))
    {
        std::string msg ="[Registry] Error: Device index out of bounds during registration.";
        LOG_FATAL(msg);
        throw RegistryException(msg);
    }

    kernel_table[op_id][dev_id] = func;
    std::string msg = std::format("[Registry] Registered Kernel: Op[{:#x}] on Dev[{:#x}]", op_id, (int)dev_id);
    LOG_INFO(msg);
}

Kernel Registry::lookup(OpCode op, Device dev) {
    auto op_id = static_cast<uint32_t>(op);
    auto dev_id = static_cast<uint8_t>(dev);

    Kernel func = kernel_table[op_id][dev_id];
    if (func == nullptr) {
        std::string msg = "[Registry] Kernel implementation missing for this Op/Device combination.";
        LOG_FATAL(msg);
        throw std::runtime_error(msg);
    }
    return func;
}
