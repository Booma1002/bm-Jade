#pragma once
#include "header/JadeReactor.hpp"

namespace bm{

    // ==============================================================================
    // -------------------------    FORWARD DISPATCHERS    --------------------------
    // ==============================================================================

    template<typename... Args>
    void Dispatcher::execute_binary(OpCode op, Jade& out, const Jade& a, const Jade& b, Args&... args) {
        // TODO: check if A, B are on the same device
        Device target_device = Device::CPU;
        JadeReactor react;
        if (op == OpCode::MATMUL) react = JadeReactor::react_matmul(op, out, a, b, args...);
        else react = JadeReactor::react_binary(op, out, a, b, args...);
        Kernel kernel_func = Registry::get().lookup(op, target_device);
        kernel_func(react);
        std::string msg = std::format("[Dispatcher] Executed Binary Reaction: OpCode={:#x}" ,static_cast<int>(op));
        LOG_DEBUG(msg);

        // --- Vein Wiring ---
        bool a_req = a.vein && a.vein->requires_grad;
        bool b_req = b.vein && b.vein->requires_grad;

        if (a_req || b_req) {
            if (!out.vein) out.vein = std::make_shared<Vein>();
            out.vein->requires_grad = true;

            uint8_t p_idx = 0;
            if (a_req) { out.vein->parents[p_idx++] = a.vein; }
            if (b_req) { out.vein->parents[p_idx++] = b.vein; }
            out.vein->num_parents = p_idx;

            // captures raw pointer to break circular dependency.
            // captures a and b by value to keep physical memory alive during backprop.
            Vein* out_vein_ptr = out.vein.get();
            out.vein->backward_op = [op, a, b, out_vein_ptr]() mutable {
                Dispatcher::execute_backward_binary(op, a, b, out_vein_ptr);
            };
        }
    }

    template<typename... Args>
    void Dispatcher::execute_unary(OpCode op, Jade& out, const Jade& a, Args&... args) {
        // TODO: check if A, B are on the same device
        Device target_device = Device::CPU;
        JadeReactor react = JadeReactor::react_unary(op, out, a, args...);
        Kernel kernel_func = Registry::get().lookup(op, target_device);
        kernel_func(react);
        std::string msg = std::format("[Dispatcher] Executed Unary Reaction: OpCode={:#x}" ,static_cast<int>(op));
        LOG_DEBUG(msg);

        // --- Vein Wiring ---
        if (a.vein && a.vein->requires_grad) {
            if (!out.vein) out.vein = std::make_shared<Vein>();
            out.vein->requires_grad = true;
            out.vein->parents[0] = a.vein;
            out.vein->num_parents = 1;

            Vein* out_vein_ptr = out.vein.get();
            out.vein->backward_op = [op, a, out_vein_ptr]() mutable {
                Dispatcher::execute_backward_unary(op, a, out_vein_ptr);
            };
        }
    }

    template<typename... Args>
    void Dispatcher::execute_scalar(OpCode op, Jade& out, const double a, Args&... args) {
        // TODO: check if A, B are on the same device
        Device target_device = Device::CPU;
        JadeReactor react = JadeReactor::react_scalar(op, out, a, args...);
        Kernel kernel_func = Registry::get().lookup(op, target_device);
        kernel_func(react);
        std::string msg = std::format("[Dispatcher] Executed Scalar Reaction: OpCode={:#x}" ,static_cast<int>(op));
        LOG_DEBUG(msg);

        // --- Vein Wiring ---
        // Scalar ops (like FILL) act in-place. In-place modification of a tracked tensor
        // normally breaks the graph, but we leave the vein intact for structural integrity.
        if (out.vein && out.vein->requires_grad) {
            LOG_DEBUG("[Dispatcher] Warning: In-place scalar operation applied to a tracked graph tensor.");
        }
    }

    template<typename... Args>
    void Dispatcher::execute_reduction_unary(OpCode op, Jade& out, const Jade& a, Args&... args) {
        // TODO: check if A, B are on the same device
        Device target_device = Device::CPU;
        JadeReactor react = JadeReactor::react_reduction(op, out, a, args...);
        Kernel kernel_func = Registry::get().lookup(op, target_device);
        kernel_func(react);
        std::string msg = std::format("[Dispatcher] Executed Unary Reduction Reaction: OpCode={:#x}" ,static_cast<int>(op));
        LOG_DEBUG(msg);

        // --- Vein Wiring ---
        if (a.vein && a.vein->requires_grad) {
            if (!out.vein) out.vein = std::make_shared<Vein>();
            out.vein->requires_grad = true;
            out.vein->parents[0] = a.vein;
            out.vein->num_parents = 1;

            Vein* out_vein_ptr = out.vein.get();
            out.vein->backward_op = [op, a, out_vein_ptr]() mutable {
                Dispatcher::execute_backward_reduction_unary(op, a, out_vein_ptr);
            };
        }
    }

    template<typename... Args>
    void Dispatcher::execute_reduction_binary(OpCode op, Jade& out, const Jade& a, const Jade& b, Args&... args) {
        // TODO: check if A, B are on the same device
        Device target_device = Device::CPU;
        JadeReactor react = JadeReactor::react_reduction_binary(op, out, a, b, args...);
        Kernel kernel_func = Registry::get().lookup(op, target_device);
        kernel_func(react);
        std::string msg = std::format("[Dispatcher] Executed Binary Reduction Reaction: OpCode={:#x}" ,static_cast<int>(op));
        LOG_DEBUG(msg);

        // --- Vein Wiring ---
        bool a_req = a.vein && a.vein->requires_grad;
        bool b_req = b.vein && b.vein->requires_grad;

        if (a_req || b_req) {
            if (!out.vein) out.vein = std::make_shared<Vein>();
            out.vein->requires_grad = true;

            uint8_t p_idx = 0;
            if (a_req) { out.vein->parents[p_idx++] = a.vein; }
            if (b_req) { out.vein->parents[p_idx++] = b.vein; }
            out.vein->num_parents = p_idx;

            Vein* out_vein_ptr = out.vein.get();
            out.vein->backward_op = [op, a, b, out_vein_ptr]() mutable {
                Dispatcher::execute_backward_reduction_binary(op, a, b, out_vein_ptr);
            };
        }
    }

    // ==============================================================================
    // --------------------------    BACKWARD DISPATCHERS    ------------------------
    // ==============================================================================

    template<typename... Args>
    void Dispatcher::execute_backward_binary(OpCode fwd_op, Jade a, Jade b, Vein* out_vein) {
        if (fwd_op == OpCode::MATMUL) {
            Jade dC = out_vein->grad;

            // dA = dC @ B.T
            if (a.vein && a.vein->requires_grad) {
                Jade dA = dC.matmul(b.transpose());
                a.vein->grad += dA;
            }

            // dB = A.T @ dC
            if (b.vein && b.vein->requires_grad) {
                Jade dB = a.transpose().matmul(dC);
                b.vein->grad += dB;
            }
        }
        else if (fwd_op == OpCode::ADD) {
            // d(A+B)/dA = 1. d(A+B)/dB = 1.
            if (a.vein && a.vein->requires_grad) a.vein->grad += out_vein->grad;
            if (b.vein && b.vein->requires_grad) b.vein->grad += out_vein->grad;
        }
        else if (fwd_op == OpCode::MUL) {
            // d(A*B)/dA = B. d(A*B)/dB = A.
            if (a.vein && a.vein->requires_grad) a.vein->grad += (b * out_vein->grad);
            if (b.vein && b.vein->requires_grad) b.vein->grad += (a * out_vein->grad);
        }
        LOG_DEBUG(std::format("[Dispatcher] Routing Backward Binary: FwdOpCode={:#x}", static_cast<int>(fwd_op)));
    }

    template<typename... Args>
    void Dispatcher::execute_backward_unary(OpCode fwd_op, Jade a, Vein* out_vein) {
        LOG_DEBUG(std::format("[Dispatcher] Routing Backward Unary: FwdOpCode={:#x}", static_cast<int>(fwd_op)));
    }

    template<typename... Args>
    void Dispatcher::execute_backward_reduction_unary(OpCode fwd_op, Jade a, Vein* out_vein) {
        LOG_DEBUG(std::format("[Dispatcher] Routing Backward Reduction Unary: FwdOpCode={:#x}", static_cast<int>(fwd_op)));
    }

    template<typename... Args>
    void Dispatcher::execute_backward_reduction_binary(OpCode fwd_op, Jade a, Jade b, Vein* out_vein) {
        LOG_DEBUG(std::format("[Dispatcher] Routing Backward Reduction Binary: FwdOpCode={:#x}", static_cast<int>(fwd_op)));
    }

}