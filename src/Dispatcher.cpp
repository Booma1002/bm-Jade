#include "header/Dispatcher.hpp"
using namespace bm;

void Dispatcher::execute_binary(OpCode op, Jade& out, const Jade& a, const Jade& b) {
    // TODO: check if A, B are on the same device
    Device target_device = Device::CPU;
    JadeReactor react;
    if (op == OpCode::MATMUL) react = JadeReactor::react_matmul(out, a, b);
    else react = JadeReactor::react_binary(out, a, b);
    Kernel kernel_func = Registry::get().lookup(op, target_device);
    kernel_func(react);
    std::string msg = std::format("[Dispatcher] Executed Binary Reaction: OpCode={:#x}" ,static_cast<int>(op));
    LOG_DEBUG(msg);
}

void Dispatcher::execute_unary(OpCode op, Jade& out, const Jade& a, const double left, const double right) {
    Device target_device = Device::CPU;
    JadeReactor react = JadeReactor::react_unary(out, a, left, right);
    Kernel kernel_func = Registry::get().lookup(op, target_device);
    kernel_func(react);
    std::string msg = std::format("[Dispatcher] Executed Unary Reaction: OpCode={:#x}" ,static_cast<int>(op));
    LOG_DEBUG(msg);
}

void Dispatcher::execute_scalar(OpCode op, Jade& out, const double a) {
    Device target_device = Device::CPU;
    JadeReactor react = JadeReactor::react_scalar(out, a);
    Kernel kernel_func = Registry::get().lookup(op, target_device);
    kernel_func(react);
    std::string msg = std::format("[Dispatcher] Executed Scalar Reaction: OpCode={:#x}" ,static_cast<int>(op));
    LOG_DEBUG(msg);
}

