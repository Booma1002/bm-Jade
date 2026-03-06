#include "header/TileKernelsCpu.hpp"
#include <limits>
#include <cmath>

namespace zeza{
#define DISPATCH_DTYPE(TYPE_ENUM, FUNCTOR, ...)                         \
switch(TYPE_ENUM) {                                                     \
        case DType::FLOAT32: FUNCTOR<float>(__VA_ARGS__); break;        \
        case DType::FLOAT64: FUNCTOR<double>(__VA_ARGS__); break;       \
        case DType::INT32:   FUNCTOR<int32_t>(__VA_ARGS__); break;      \
        case DType::INT16:   FUNCTOR<int16_t>(__VA_ARGS__); break;      \
        case DType::UINT8:   FUNCTOR<uint8_t>(__VA_ARGS__); break;      \
        case DType::UINT16:  FUNCTOR<uint16_t>(__VA_ARGS__); break;     \
        case DType::UINT32:  FUNCTOR<uint32_t>(__VA_ARGS__); break;     \
        case DType::UINT64:  FUNCTOR<uint64_t>(__VA_ARGS__); break;     \
        case DType::INT64:  FUNCTOR<int64_t>(__VA_ARGS__); break;     \
        default: {                                                      \
            LOG_ERR("Unsupported DType in Dispatcher");                 \
            throw std::runtime_error("Unsupported DType in Dispatcher");\
        }                                                               \
    }

constexpr void get_cursor(uint64_t linear_idx, uint64_t* cursor, const uint64_t* shape, const uint64_t ndims) {
    for (uint64_t dim2 = 0; dim2 < ndims; ++dim2) {
        uint64_t dim = ndims - dim2 - 1;
        cursor[dim] = linear_idx % shape[dim];
        linear_idx /= shape[dim];
    }
}

// ======================================================
// =========={..........Kernels..........}===============
// ======================================================
;
void cpu_add_kernel(TileOperator& op){
    DISPATCH_DTYPE(op.dtype, cpu_elementwise_binary_invoke, op, [](auto a, auto b) { return a + b; });
}

void cpu_mul_kernel(TileOperator& op){
    DISPATCH_DTYPE(op.dtype, cpu_elementwise_binary_invoke, op, [](auto a, auto b) {return a * b;});
}

void cpu_sub_kernel(TileOperator& op) {
    DISPATCH_DTYPE(op.dtype, cpu_elementwise_binary_invoke, op, [](auto a, auto b) { return a - b; });
}

void cpu_matmul_kernel(TileOperator& oper) {
    DISPATCH_DTYPE(oper.dtype, cpu_MatMul_binary_invoke, oper);
}

void cpu_copy_kernel(TileOperator& op) {
    DISPATCH_DTYPE(op.dtype, cpu_elementwise_unary_invoke, op, [](auto a) { return a; });
}

void cpu_fill_kernel(TileOperator& op) {
    DISPATCH_DTYPE(op.dtype, cpu_elementwise_scalar_invoke, op, [&op](auto a) { return op.Val; });
}

void cpu_sin_kernel(TileOperator& op) {
    DISPATCH_DTYPE(op.dtype, cpu_elementwise_unary_invoke, op, [](auto a) { return static_cast<decltype(a)>(std::sin(a)); });
}

void cpu_cos_kernel(TileOperator& op) {
    DISPATCH_DTYPE(op.dtype, cpu_elementwise_unary_invoke, op, [](auto a) { return static_cast<decltype(a)>(std::cos(a)); });
}

void cpu_tan_kernel(TileOperator& op) {
    DISPATCH_DTYPE(op.dtype, cpu_elementwise_unary_invoke, op, [](auto a) { return static_cast<decltype(a)>(std::tan(a)); });
}

void cpu_exp_kernel(TileOperator& op) {
    DISPATCH_DTYPE(op.dtype, cpu_elementwise_unary_invoke, op, [](auto a) { return static_cast<decltype(a)>(std::exp(a)); });
}

void cpu_log_kernel(TileOperator& op) {
    DISPATCH_DTYPE(op.dtype, cpu_elementwise_unary_invoke, op, [](auto a) { return static_cast<decltype(a)>(std::log(a)); });
}

void cpu_clip_kernel(TileOperator& oper) {
    DISPATCH_DTYPE(oper.dtype, cpu_elementwise_unary_invoke, oper,
                   [&oper](auto a) {
                       auto l = oper.Left;
                       auto r = oper.Right;
                       auto res = (a < l) ? l :
                                  (a > r) ? r : a;
                       return static_cast<decltype(a)>(res);
                   });
}

}