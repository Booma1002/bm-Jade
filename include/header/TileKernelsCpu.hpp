#pragma once
#include <cmath>
#include <omp.h>
#include "Registry.hpp"
#include "TileOperator.hpp"
namespace zeza {

// ======================================================
// =========={..........Helpers..........}===============
// ======================================================
    ;

    constexpr void get_cursor(uint64_t linear_idx, uint64_t *cursor, const uint64_t *shape, uint64_t ndims);

// ==================================================================
// =========={..........CPU K-nary Invoking..........}===============
// ==================================================================
    ;

    template<typename Func>
    void cpu_elementwise_unary_invoke(TileOperator &oper, Func op);

    template<typename Func>
    void cpu_elementwise_scalar_invoke(TileOperator &oper, Func op);

    template<typename Func>
    void cpu_elementwise_binary_invoke(TileOperator &oper, Func op);

    void cpu_MatMul_binary_invoke(TileOperator &oper);



// ======================================================
// =========={..........Kernels..........}===============
// ======================================================
    ;

    void cpu_add_kernel(TileOperator &op);

    void cpu_sub_kernel(TileOperator &op);

    void cpu_matmul_kernel(TileOperator &oper);

    void cpu_copy_kernel(TileOperator &op);

    void cpu_fill_kernel(TileOperator &op);

    void cpu_mul_kernel(TileOperator &op);

    void cpu_mul_kernel(TileOperator &op);

    void cpu_sin_kernel(TileOperator &op);

    void cpu_cos_kernel(TileOperator &op);

    void cpu_tan_kernel(TileOperator &op);

    void cpu_exp_kernel(TileOperator &op);

    void cpu_log_kernel(TileOperator &op);

    void cpu_clip_kernel(TileOperator &oper)

// ===========================================================
// =========={..........Registration..........}===============
// ===========================================================
    ;

    REGISTER_KERNEL(ADD, CPU, cpu_add_kernel);
    REGISTER_KERNEL(SUB, CPU, cpu_sub_kernel);
    REGISTER_KERNEL(MATMUL, CPU, cpu_matmul_kernel);
    REGISTER_KERNEL(COPY, CPU, cpu_copy_kernel);
    REGISTER_KERNEL(FILL, CPU, cpu_fill_kernel);
    REGISTER_KERNEL(MUL, CPU, cpu_mul_kernel);
    REGISTER_KERNEL(SIN, CPU, cpu_sin_kernel);
    REGISTER_KERNEL(COS, CPU, cpu_cos_kernel);
    REGISTER_KERNEL(TAN, CPU, cpu_tan_kernel);
    REGISTER_KERNEL(EXP, CPU, cpu_exp_kernel);
    REGISTER_KERNEL(LOG, CPU, cpu_log_kernel);
    REGISTER_KERNEL(CLIP, CPU, cpu_clip_kernel);


}

#include "temp/TileKernelsCpu.tpp"