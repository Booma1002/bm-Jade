#include "header/JadeReactor.hpp"
using namespace bm;
JadeReactor JadeReactor::react_binary(Jade& out, const Jade& a, const Jade& b) {
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
    react.num_elements = out.get_size();
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

JadeReactor JadeReactor::react_unary(Jade& out, const Jade& a, const double left, const double right){
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
    react.dtype = out.dtype;
    std::string msg;
    react.Left = left;
    react.Right = right;
    msg += std::format("Left = {:f}, Right = {:f}", react.Left, react.Right);
    LOG_DEBUG(msg);
    react.num_elements = out.get_size();
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
    for (int i=0; i < react.ndims; ++i) std::to_string(react.shape[i]) + ((i != react.ndims - 1) ? ", " : "");
    msg+= std::format(").\n");
    if(react.ndims) LOG_DEBUG(msg);
    msg= std::format("[Unary Reactor] New ndims: {}.", std::to_string(react.ndims));
    LOG_INFO(msg);
    msg= "[Unary Reactor] Saved Reactor Settings Successfully.";
    LOG_INFO(msg);
    return react;
}

JadeReactor JadeReactor::react_scalar(Jade& out, double Val){
    JadeReactor react;
    react.dtype = out.dtype;
    react.Val = Val;
    react.num_elements = out.get_size();
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


JadeReactor JadeReactor::react_matmul(Jade& out, const Jade& a, const Jade& b) {
    if (a.dtype != b.dtype){
        std::string msg = "DType Mismatch: Type Promotion not yet supported.";
        LOG_WARN(msg);
        throw std::runtime_error(msg);
    }
    JadeReactor react;
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

    react.num_elements = out.get_size();
    react.phys[0] = out.data_ptr();
    react.phys[1] = a.data_ptr();
    react.phys[2] = b.data_ptr();
    react.is_contiguous = false;
    std::string msg;
    msg+= std::format("[MatMul Reactor] Saved Reactor Settings Successfully.");
    LOG_INFO(msg);
    return react;
}



void JadeReactor::merge_dims() {
    if (ndims <= 1) return;
    for (int cur = ndims - 1; cur > 0; --cur) {
        int mother = cur - 1;
        bool can_do_collapse = true;
        for (int _ = 0; _ < RE_MAX_REACTANTS; ++_) {
            if (strides[_][mother] != shape[cur] * strides[_][cur]) {
                can_do_collapse = false;
                break;
            }
        }
        if (can_do_collapse) {
            // mother copies current metadata:
            shape[mother] *= shape[cur];
            for (int _ = 0; _ < RE_MAX_REACTANTS; ++_)
                strides[_][mother] = strides[_][cur];

            // current tracker copies its daughter metadata
            for (int inward = cur; inward < ndims - 1; ++inward) {
                int daughter = inward + 1;

                shape[inward] = shape[daughter];
                for (int _ = 0; _ < RE_MAX_REACTANTS; ++_)
                    strides[_][inward] = strides[_][daughter];
            }
            ndims--;
        }
    }
}



