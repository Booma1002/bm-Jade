#pragma once
#include "header/Jade.hpp"
namespace bm {

    template<typename... Args>
    JadeReactor JadeReactor::react_binary(OpCode opcode, Jade& out, const Jade& a, const Jade& b, Args... args) {
        if (a.dtype != b.dtype) {
            std::string msg = "DType Mismatch: Type Promotion not yet supported.";
            LOG_WARN(msg);
            throw std::runtime_error(msg);
        }
        try{
            auto shapes = Jade::broadcast(a, b);
            auto ndm = std::max(a.ndims, b.ndims);
            auto y = Jade::broadcast(out.shape.get(), out.ndims, shapes.get(), ndm);
        }
        catch(...){
            std::string msg;
            msg += std::format("[Binary Reactor] Shape Mismatch: Cannot Broadcast "
                    ,a.repr() , ", With ", b.repr());
            LOG_ERR(msg);
            throw ShapeMismatchException(msg);
        }
        JadeReactor react;
        react.opcode = (int)opcode;
        int arg_idx = 0;
        ([&] {
            if (arg_idx < RE_MAX_ARGS) {
                react.args[arg_idx++] = const_cast<void*>(static_cast<const void*>(&args));
            }
        }(), ...);
        react.ndims = out.ndims;
        for(long long i = 0; i < react.ndims; ++i) {
            react.shape[i] = out.shape[i];
            react.strides[0][i] = out.strides[i];
            long long dim_a = i - (static_cast<long long>(react.ndims) - a.ndims);
            if (dim_a >= 0) {
                if (a.shape[dim_a] == 1 && out.shape[i] > 1) {
                    react.strides[1][i] = 0;
                }
                else {
                    react.strides[1][i] = a.strides[dim_a];
                }
            }
            else {
                react.strides[1][i] = 0;
            }
            long long dim_b = i - (static_cast<long long>(react.ndims) - b.ndims);
            if (dim_b >= 0) {
                if (b.shape[dim_b] == 1 && out.shape[i] > 1) {
                    react.strides[2][i] = 0;
                } else {
                    react.strides[2][i] = b.strides[dim_b];
                }
            } else {
                react.strides[2][i] = 0;
            }
        }
        react.nelm = out.get_size();
        react.phys[0] = out.data_ptr();
        react.phys[1] = a.data_ptr();
        react.phys[2] = b.data_ptr();
        react.dtype = out.dtype;

        std::string info = "[Binary Reactor] Shape (";
        for (int i=0; i < a.ndims; ++i) info+= std::to_string(a.shape[i]) + ((i!=a.ndims-1)?", ":"");
        info+= ") [+] ";
        info+="Shape (";
        for (int i=0; i < b.ndims; ++i) info+= std::to_string(b.shape[i]) + ((i!=b.ndims-1)?", ":"");
        info+= ")  --> ";

        react.merge_dims();
        react.is_contiguous = true;
        for (int _ = 0; _ < 3; ++_) {
            if (react.ndims == 1 && react.strides[_][0] != 1)
                react.is_contiguous = false;
            if (react.ndims > 1) react.is_contiguous = false;
        }
        info+= "Shape (";
        for (int i=0; i < react.ndims; ++i) info+= std::to_string(react.shape[i]) + ((i != react.ndims - 1) ? ", " : "");
        info+= ").";
        if(react.ndims) LOG_DEBUG(info);
        info = "[Binary Reactor] New ndims: " + std::to_string(react.ndims ) + ".";
        LOG_INFO(info);
        info = "[Binary Reactor] Saved Reactor Settings Successfully.";
        LOG_INFO(info);
        return react;
    }

    template<typename... Args>
    JadeReactor JadeReactor::react_unary(OpCode opcode, Jade& out, const Jade& a, Args... args){
        if (out.ndims != a.ndims) {
            std::string msg;
            msg += std::format("[Unary Reactor] Rank Mismatch. \nA: ",  a.repr() ,  "\nOutput: ",  out.repr());
            LOG_WARN(msg);
            throw ShapeMismatchException(msg);
        }
        for(int i=0; i<out.ndims; ++i)
            if (out.shape[i] != a.shape[i]) {
                std::string msg;
                msg += std::format("[Unary Reactor] Shape Mismatch. \nA: ",  a.repr() ,  "\nOutput: ",  out.repr());
                LOG_ERR(msg);
                throw ShapeMismatchException(msg);
            }

        JadeReactor react;
        react.opcode = (int)opcode;
        int arg_idx = 0;
        ([&] {
            if (arg_idx < RE_MAX_ARGS) {
                react.args[arg_idx++] = const_cast<void*>(static_cast<const void*>(&args));
            }
        }(), ...);
        react.dtype = out.dtype;
        std::string msg;
        react.nelm = out.get_size();
        react.ndims = out.ndims;
        for(int i=0; i < react.ndims; ++i) {
            react.shape[i] = out.shape[i];
            react.strides[0][i] = out.strides[i]; // Output
            react.strides[1][i] = a.strides[i];   // Input
            react.strides[2][i] = 0;              // Dummy
        }
        react.phys[0] = out.data_ptr();
        react.phys[1] = a.data_ptr();
        react.phys[2] = nullptr; // Dummy


        msg= std::format("[Unary Reactor] Shape (");
        for (int i=0; i < a.ndims; ++i) msg+= std::to_string(a.shape[i]) +((i!=a.ndims-1)?", ":"");
        msg+= ") --> ";
        react.merge_dims();

        react.is_contiguous = true;
        if (react.ndims == 1 && (react.strides[0][0] != 1 || react.strides[1][0] != 1))
            react.is_contiguous = false;
        if (react.ndims > 1) react.is_contiguous = false;

        msg+= std::format("Shape (");
        for (int i=0; i < react.ndims; ++i) msg+= std::to_string(react.shape[i]) + ((i != react.ndims - 1) ? ", " : "");
        msg+= std::format(").\n");
        if(react.ndims) LOG_DEBUG(msg);
        msg= std::format("[Unary Reactor] New ndims: {}.", std::to_string(react.ndims));
        LOG_INFO(msg);
        msg= "[Unary Reactor] Saved Reactor Settings Successfully.";
        LOG_INFO(msg);
        return react;
    }

    template<typename... Args>
    JadeReactor JadeReactor::react_scalar(OpCode opcode, Jade& out, Args... args){
        JadeReactor react;
        react.dtype = out.dtype;
        react.opcode = (int)opcode;
        int arg_idx = 0;
        ([&] {
            if (arg_idx < RE_MAX_ARGS) {
                react.args[arg_idx++] = const_cast<void*>(static_cast<const void*>(&args));
            }
        }(), ...);
        react.nelm = out.get_size();
        react.ndims = out.ndims;
        for(int i=0; i < react.ndims; ++i) {
            react.shape[i] = out.shape[i];
            react.strides[0][i] = out.strides[i]; // Output
            react.strides[1][i] = 0;   // Input
            react.strides[2][i] = 0;   // Dummy
        }
        react.phys[0] = out.data_ptr();
        react.phys[1] = out.data_ptr();
        react.phys[2] = nullptr; // Dummy

        std::string msg;
        msg+= std::format("[Scalar Reactor] Shape (");
        for (int i=0; i < out.ndims; ++i) msg += std::to_string(out.shape[i]) + ((i!=out.ndims-1)?", ":"");
        msg+= std::format(") --> ");
        react.merge_dims();

        react.is_contiguous = true;
        if (react.ndims == 1 && (react.strides[0][0] != 1 || react.strides[1][0] != 1))
            react.is_contiguous = false;
        if (react.ndims > 1) react.is_contiguous = false;

        msg+= std::format("Shape (");
        for (int i=0; i < react.ndims; ++i) msg+= std::to_string(react.shape[i]) + ((i != react.ndims - 1) ? ", " : "");
        msg+= std::format(").\n");
        if(react.ndims) LOG_DEBUG(msg);
        msg= std::format("[Scalar Reactor] New ndims: {}.", std::to_string(react.ndims));
        LOG_INFO(msg);
        msg= "[Scalar Reactor] Saved Reactor Settings Successfully.";
        LOG_INFO(msg);
        return react;
    }


    template<typename... Args>
    JadeReactor JadeReactor::react_matmul(OpCode opcode, Jade& out, const Jade& a, const Jade& b, Args... args) {
        if (a.dtype != b.dtype){
            std::string msg = "DType Mismatch: Type Promotion not yet supported.";
            LOG_WARN(msg);
            throw std::runtime_error(msg);
        }
        JadeReactor react;
        react.opcode = (int)opcode;
        int arg_idx = 0;
        ([&] {
            if (arg_idx < RE_MAX_ARGS) {
                react.args[arg_idx++] = const_cast<void*>(static_cast<const void*>(&args));
            }
        }(), ...);
        react.dtype = out.dtype;
        react.inner_k = a.shape[a.ndims - 1];
        react.ndims = out.ndims;
        react.strides[0][react.ndims - 1] = out.strides[out.ndims - 1];
        if(out.ndims>1)
            react.strides[0][react.ndims - 2] = out.strides[out.ndims - 2];
        else react.strides[0][react.ndims - 2] = 0;
        react.shape[react.ndims - 1] = out.shape[out.ndims - 1];
        if(out.ndims > 1)
            react.shape[react.ndims - 2] = out.shape[out.ndims - 2];
        else
            react.shape[react.ndims - 2] = 1;

        react.strides[1][react.ndims - 1] = a.strides[a.ndims - 1];
        if(a.ndims>1)
            react.strides[1][react.ndims - 2] = a.strides[a.ndims - 2];
        else react.strides[1][react.ndims - 2] = 0;
        react.strides[2][react.ndims - 1] = b.strides[b.ndims - 1];
        if(b.ndims>1)
            react.strides[2][react.ndims - 2] = b.strides[b.ndims - 2];
        else react.strides[2][react.ndims - 2] =0;

        for(long long i=0; i < react.ndims - 2; ++i) {
            long long dim_a = i - (static_cast<long long>(react.ndims) - a.ndims);
            long long dim_b = i - (static_cast<long long>(react.ndims) - b.ndims);
            react.shape[i] = out.shape[i];
            react.strides[0][i] = out.strides[i];
            if(dim_a>=0)
                react.strides[1][i] = (a.shape[dim_a] == 1 && out.shape[i] > 1) ? 0 : a.strides[dim_a];
            else react.strides[1][i] = 0;
            if(dim_b>=0)
                react.strides[2][i] = (b.shape[dim_b] == 1 && out.shape[i] > 1) ? 0 : b.strides[dim_b];
            else react.strides[2][i] = 0;
        }

        react.nelm = out.get_size();
        react.phys[0] = out.data_ptr();
        react.phys[1] = a.data_ptr();
        react.phys[2] = b.data_ptr();
        react.is_contiguous = false;
        std::string msg;
        msg+= std::format("[MatMul Reactor] Saved Reactor Settings Successfully.");
        LOG_INFO(msg);
        return react;
    }



    template<auto MemberFunc, typename T, typename Ret, typename... Args>
    struct Binder<MemberFunc, Ret (T::*)(Args...)> {
        static void bind(JadeReactor *re, ReactorMethod id) {
            re->template create_thunk<T, Args..., MemberFunc>(id);
            std::string msg = std::format("Bound Reactor to ReactorMethod. Method ID={:#x}", (uint32_t)id);
            LOG_DEBUG(msg);
        }
    };

    template<auto MemberFunc, typename T, typename Ret, typename... Args>
    struct Binder<MemberFunc, Ret (T::*)(Args...) const> {
        static void bind(JadeReactor *re, ReactorMethod id) {
            re->template create_thunk<T, Args..., MemberFunc>(id);
            std::string msg = std::format("Bound Reactor to ReactorMethod. Method ID={:#x}", (uint32_t)id);
            LOG_DEBUG(msg);
        }
    };

    template<auto MemberFunc, typename T, typename Ret, typename... Args>
    struct Binder<MemberFunc, Ret (T::*)(Args...) noexcept> {
        static void bind(JadeReactor *re, ReactorMethod id) {
            re->template create_thunk<T, Args..., MemberFunc>(id);
            std::string msg = std::format("Bound Reactor to ReactorMethod. Method ID={:#x}", (uint32_t)id);
            LOG_DEBUG(msg);
        }
    };

    template<auto MemberFunc, typename T, typename Ret, typename... Args>
    struct Binder<MemberFunc, Ret (T::*)(Args...) const noexcept> {
        static void bind(JadeReactor *re, ReactorMethod id) {
            re->template create_thunk<T, Args..., MemberFunc>(id);
            std::string msg = std::format("Bound Reactor to ReactorMethod. Method ID={:#x}", (uint32_t)id);
            LOG_DEBUG(msg);
        }
    };


    template<typename... Args>
    void JadeReactor::call(ReactorMethod id, Args... args) {
        int idx = static_cast<int>(id);
        if (idx >= MAX_RE_METHODS || idx < 0)
        {
            std::string msg="[Reactor] Method ID out of bounds.";
            LOG_FATAL(msg);
            throw ReactorException(msg);
        }
        if (!methods[idx])
        {
            std::string msg="[Reactor] Call To Undefined ReactorMethod ID (Not Bound).";
            LOG_FATAL(msg);
            throw ReactorException(msg);
        }

        using TypedFunc = void (*)(void *, Args...);
        auto func = reinterpret_cast<TypedFunc>(methods[idx]);
        func(bound_obj, args...);

        if (bound_obj) phys[0] = static_cast<Jade *>(bound_obj)->memory->template data<float>();
    }

    template<typename T, typename... Args, auto MemberPtr>
    void JadeReactor::create_thunk(ReactorMethod id) {
        int idx = static_cast<int>(id);
        if (idx >= MAX_RE_METHODS || idx < 0)
        {
            std::string msg="[Reactor] Method ID out of bounds.";
            LOG_FATAL(msg);
            throw ReactorException(msg);
        }

        auto thunk = [](void *ctx, Args... args) {
            T *instance = static_cast<T *>(ctx);
            (instance->*MemberPtr)(args...);
        };
        methods[idx] = reinterpret_cast<GenericFunc>(+thunk);
    }

    constexpr bool JadeReactor::has(ReactorMethod id) const {
        return methods[static_cast<int>(id)] != nullptr;
    }

    template<typename Func>
    void JadeReactor::bind(ReactorMethod id, Func &&f) {
        int idx = static_cast<int>(id);
        if (idx >= MAX_RE_METHODS || idx < 0)
        {
            std::string msg="[Reactor] Method ID out of bounds.";
            LOG_FATAL(msg);
            throw ReactorException(msg);
        }

        // Safe conversion of stateless lambda to function pointer
        auto ptr = reinterpret_cast<GenericFunc>(+f);
        methods[idx] = ptr;
    }

    template<auto MemberFunc>
    void JadeReactor::bind_private(ReactorMethod id) {
        Binder<MemberFunc>::bind(this, id);
    }
}