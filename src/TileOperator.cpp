#include "header/TileOperator.hpp"
using namespace zeza;
TileOperator TileOperator::operate_binary(Tile& out, const Tile& a, const Tile& b) {
    if (a.dtype != b.dtype) {
        std::string msg = "DType Mismatch: Type Promotion not yet supported.";
        LOG_WARN(msg);
        throw std::runtime_error(msg);
    }
    try{
        auto shapes = Tile::broadcast(a, b);
        auto ndm = std::max(a.ndims, b.ndims);
        auto y = Tile::broadcast(out.shape.get(), out.ndims, shapes.get(), ndm);
    }
    catch(...){
        std::string msg;
        msg += std::format("[Binary Operator] Shape Mismatch: Cannot Broadcast "
                              ,a.repr() , ", With ", b.repr());
        LOG_ERR(msg);
        throw ShapeMismatchException(msg);
    }
    TileOperator oper;
    oper.ndims = out.ndims;
    for(long long i = 0; i < oper.ndims; ++i) {
        oper.shape[i] = out.shape[i];
        oper.strides[0][i] = out.strides[i];
        long long dim_a = i - (static_cast<long long>(oper.ndims) - a.ndims);
        if (dim_a >= 0) {
            if (a.shape[dim_a] == 1 && out.shape[i] > 1) {
                oper.strides[1][i] = 0;
            }
            else {
                oper.strides[1][i] = a.strides[dim_a];
            }
        }
        else {
            oper.strides[1][i] = 0;
        }
        long long dim_b = i - (static_cast<long long>(oper.ndims) - b.ndims);
        if (dim_b >= 0) {
            if (b.shape[dim_b] == 1 && out.shape[i] > 1) {
                oper.strides[2][i] = 0;
            } else {
                oper.strides[2][i] = b.strides[dim_b];
            }
        } else {
            oper.strides[2][i] = 0;
        }
    }
    oper.num_elements = out.get_size();
    oper.phys[0] = out.data_ptr();
    oper.phys[1] = a.data_ptr();
    oper.phys[2] = b.data_ptr();
    oper.dtype = out.dtype;

    std::string info = "[Binary Operator] Shape (";
    for (int i=0; i < a.ndims; ++i) info+= std::to_string(a.shape[i]) + ((i!=a.ndims-1)?", ":"");
    info+= ") [OP] ";
    info+="Shape (";
    for (int i=0; i < b.ndims; ++i) info+= std::to_string(b.shape[i]) + ((i!=b.ndims-1)?", ":"");
    info+= ")  --> ";

    oper.merge_dims();
    oper.is_contiguous = true;
    for (int _ = 0; _ < 3; ++_) {
        if (oper.ndims == 1 && oper.strides[_][0] != 1)
            oper.is_contiguous = false;
        if (oper.ndims > 1) oper.is_contiguous = false;
    }
    info+= "Shape (";
    for (int i=0; i < oper.ndims; ++i) info+= std::to_string(oper.shape[i]) +((i!=oper.ndims-1)?", ":"");
    info+= ").";
    if(oper.ndims) LOG_DEBUG(info);
    info = "[Binary Operator] New ndims: " + std::to_string(oper.ndims ) +  ".";
    LOG_INFO(info);
    info = "[Binary Operator] Saved Operator Settings Successfully.";
    LOG_INFO(info);
    return oper;
}

TileOperator TileOperator::operate_unary(Tile& out, const Tile& a, const double left, const double right){
    if (out.ndims != a.ndims) {
        std::string msg;
        msg += std::format("[Unary Operator] Rank Mismatch. \nA: ",  a.repr() ,  "\nOutput: ",  out.repr());
        LOG_WARN(msg);
        throw ShapeMismatchException(msg);
    }
    for(int i=0; i<out.ndims; ++i)
        if (out.shape[i] != a.shape[i]) {
            std::string msg;
            msg += std::format("[Unary Operator] Shape Mismatch. \nA: ",  a.repr() ,  "\nOutput: ",  out.repr());
            LOG_ERR(msg);
            throw ShapeMismatchException(msg);
        }

    TileOperator oper;
    oper.dtype = out.dtype;
    std::string msg;
    oper.Left = left;
    oper.Right = right;
    msg += std::format("Left = {:f}, Right = {:f}",oper.Left, oper.Right);
    LOG_DEBUG(msg);
    oper.num_elements = out.get_size();
    oper.ndims = out.ndims;
    for(int i=0; i < oper.ndims; ++i) {
        oper.shape[i] = out.shape[i];
        oper.strides[0][i] = out.strides[i]; // Output
        oper.strides[1][i] = a.strides[i];   // Input
        oper.strides[2][i] = 0;              // Dummy
    }
    oper.phys[0] = out.data_ptr();
    oper.phys[1] = a.data_ptr();
    oper.phys[2] = nullptr; // Dummy

    msg= std::format("[Unary Operator] Shape (");
    for (int i=0; i < a.ndims; ++i) msg+= std::to_string(a.shape[i]) +((i!=a.ndims-1)?", ":"");
    msg+= ") --> ";
    oper.merge_dims();

    oper.is_contiguous = true;
    if (oper.ndims == 1 && (oper.strides[0][0] != 1 || oper.strides[1][0] != 1))
        oper.is_contiguous = false;
    if (oper.ndims > 1) oper.is_contiguous = false;

    msg+= std::format("Shape (");
    for (int i=0; i < oper.ndims; ++i) std::to_string(oper.shape[i]) + ((i!=oper.ndims-1)?", ":"");
    msg+= std::format(").\n");
    if(oper.ndims) LOG_DEBUG(msg);
    msg= std::format("[Unary Operator] New ndims: {}.", std::to_string(oper.ndims));
    LOG_INFO(msg);
    msg= "[Unary Operator] Saved Operator Settings Successfully.";
    LOG_INFO(msg);
    return oper;
}

TileOperator TileOperator::operate_scalar(Tile& out, double Val){
    TileOperator oper;
    oper.dtype = out.dtype;
    oper.Val = Val;
    oper.num_elements = out.get_size();
    oper.ndims = out.ndims;
    for(int i=0; i < oper.ndims; ++i) {
        oper.shape[i] = out.shape[i];
        oper.strides[0][i] = out.strides[i]; // Output
        oper.strides[1][i] = 0;   // Input
        oper.strides[2][i] = 0;   // Dummy
    }
    oper.phys[0] = out.data_ptr();
    oper.phys[1] = out.data_ptr();
    oper.phys[2] = nullptr; // Dummy

    std::string msg;
    msg+= std::format("[Scalar Operator] Shape (");
    for (int i=0; i < out.ndims; ++i) msg += std::to_string(out.shape[i]) + ((i!=out.ndims-1)?", ":"");
    msg+= std::format(") --> ");
    oper.merge_dims();

    oper.is_contiguous = true;
    if (oper.ndims == 1 && (oper.strides[0][0] != 1 || oper.strides[1][0] != 1))
        oper.is_contiguous = false;
    if (oper.ndims > 1) oper.is_contiguous = false;

    msg+= std::format("Shape (");
    for (int i=0; i < oper.ndims; ++i) msg+= std::to_string(oper.shape[i]) + ((i!=oper.ndims-1)?", ":"");
    msg+= std::format(").\n");
    if(oper.ndims) LOG_DEBUG(msg);
    msg= std::format("[Scalar Operator] New ndims: {}.", std::to_string(oper.ndims));
    LOG_INFO(msg);
    msg= "[Scalar Operator] Saved Operator Settings Successfully.";
    LOG_INFO(msg);
    return oper;
}


TileOperator TileOperator::operate_matmul(Tile& out, const Tile& a, const Tile& b) {
    if (a.dtype != b.dtype){
        std::string msg = "DType Mismatch: Type Promotion not yet supported.";
        LOG_WARN(msg);
        throw std::runtime_error(msg);
    }
    TileOperator oper;
    oper.dtype = out.dtype;
    oper.inner_k = a.shape[a.ndims - 1];
    oper.ndims = out.ndims;
    oper.strides[0][oper.ndims-1] = out.strides[out.ndims-1];
    if(out.ndims>1)
        oper.strides[0][oper.ndims-2] = out.strides[out.ndims-2];
    else oper.strides[0][oper.ndims-2] = 0;
    oper.shape[oper.ndims-1] = out.shape[out.ndims-1];
    if(out.ndims > 1)
        oper.shape[oper.ndims-2] = out.shape[out.ndims-2];
    else
        oper.shape[oper.ndims-2] = 1;

    oper.strides[1][oper.ndims-1] = a.strides[a.ndims-1];
    if(a.ndims>1)
        oper.strides[1][oper.ndims-2] = a.strides[a.ndims-2];
    else oper.strides[1][oper.ndims-2] = 0;
    oper.strides[2][oper.ndims-1] = b.strides[b.ndims-1];
    if(b.ndims>1)
        oper.strides[2][oper.ndims-2] = b.strides[b.ndims-2];
    else oper.strides[2][oper.ndims-2] =0;

    for(long long i=0; i < oper.ndims-2; ++i) {
        long long dim_a = i - (static_cast<long long>(oper.ndims) - a.ndims);
        long long dim_b = i - (static_cast<long long>(oper.ndims) - b.ndims);
        oper.shape[i] = out.shape[i];
        oper.strides[0][i] = out.strides[i];
        if(dim_a>=0)
            oper.strides[1][i] = (a.shape[dim_a] == 1 && out.shape[i] > 1)? 0 : a.strides[dim_a];
        else oper.strides[1][i] = 0;
        if(dim_b>=0)
            oper.strides[2][i] = (b.shape[dim_b] == 1 && out.shape[i] > 1)? 0 : b.strides[dim_b];
        else oper.strides[2][i] = 0;
    }

    oper.num_elements = out.get_size();
    oper.phys[0] = out.data_ptr();
    oper.phys[1] = a.data_ptr();
    oper.phys[2] = b.data_ptr();
    oper.is_contiguous = false;
    std::string msg;
    msg+= std::format("[MatMul Operator] Saved Operator Settings Successfully.");
    LOG_INFO(msg);
    return oper;
}



void TileOperator::merge_dims() {
    if (ndims <= 1) return;
    for (int cur = ndims - 1; cur > 0; --cur) {
        int mother = cur - 1;
        bool can_do_collapse = true;
        for (int _ = 0; _ < OPER_MAX_OPERANDS; ++_) {
            if (strides[_][mother] != shape[cur] * strides[_][cur]) {
                can_do_collapse = false;
                break;
            }
        }
        if (can_do_collapse) {
            // mother copies current metadata:
            shape[mother] *= shape[cur];
            for (int _ = 0; _ < OPER_MAX_OPERANDS; ++_)
                strides[_][mother] = strides[_][cur];

            // current tracker copies its daughter metadata
            for (int inward = cur; inward < ndims - 1; ++inward) {
                int daughter = inward + 1;

                shape[inward] = shape[daughter];
                for (int _ = 0; _ < OPER_MAX_OPERANDS; ++_)
                    strides[_][inward] = strides[_][daughter];
            }
            ndims--;
        }
    }
}



