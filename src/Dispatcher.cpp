#include "header/Dispatcher.hpp"
using namespace zeza;

void Dispatcher::execute_binary(OpCode op, Tile& out, const Tile& a, const Tile& b) {
    // TODO: check if A, B are on the same device
    Device target_device = Device::CPU;
    TileOperator opr;
    if (op == OpCode::MATMUL) opr = TileOperator::operate_matmul(out, a, b);
    else opr = TileOperator::operate_binary(out, a, b);
    Kernel kernel_func = Registry::get().lookup(op, target_device);
    kernel_func(opr);
    std::string msg = std::format("[Dispatcher] Executed Binary Operation: OpCode={:#x}" ,static_cast<int>(op));
    LOG_DEBUG(msg);
}

void Dispatcher::execute_unary(OpCode op, Tile& out, const Tile& a, const double left, const double right) {
    Device target_device = Device::CPU;
    TileOperator opr = TileOperator::operate_unary(out, a, left, right);
    Kernel kernel_func = Registry::get().lookup(op, target_device);
    kernel_func(opr);
    std::string msg = std::format("[Dispatcher] Executed Unary Operation: OpCode={:#x}" ,static_cast<int>(op));
    LOG_DEBUG(msg);
}

void Dispatcher::execute_scalar(OpCode op, Tile& out, const double a) {
    Device target_device = Device::CPU;
    TileOperator opr = TileOperator::operate_scalar(out, a);
    Kernel kernel_func = Registry::get().lookup(op, target_device);
    kernel_func(opr);
    std::string msg = std::format("[Dispatcher] Executed Scalar Operation: OpCode={:#x}" ,static_cast<int>(op));
    LOG_DEBUG(msg);
}

