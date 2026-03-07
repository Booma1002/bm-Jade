#pragma once
namespace bm {
// ==================================================================
// =========={..........CPU K-nary Invoking..........}===============
// ==================================================================
    ;

    template<typename T, typename Func>
    void cpu_elementwise_unary_invoke(JadeReactor &react, Func op) {
        std::string msg = std::format("[CPU Invoker] Performing Contiguous Jade Unary Reaction. Reactor Ndims={}{}",
                                      std::to_string(react.ndims), ".");
        LOG_INFO(msg);
        if (react.is_contiguous) {
            auto out = static_cast<T *>(react.phys[0]);
            auto in = static_cast<T *>(react.phys[1]);
////////////////////////////////////////////////####{
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) default(none) shared(react, out, in, op)
#endif
////////////////////////////////////////////////####}
            for (uint64_t i = 0; i < react.num_elements; ++i) {
                out[i] = op(in[i]);
            }
            return;
        }

////////////////////////////////////////////////####{
#if defined(_OPENMP)
#pragma omp parallel default(none) shared(react, op, RE_MAX_DIMS)
#endif
////////////////////////////////////////////////####}
        {
            int thread = 0;
            int num_threads = 1;
////////////////////////////////////////////////####{
#if defined(_OPENMP)
            thread = omp_get_thread_num();
            num_threads = omp_get_num_threads();
#endif
////////////////////////////////////////////////####}
            uint64_t chop = react.num_elements / num_threads;
            uint64_t r = react.num_elements % num_threads;
            uint64_t begin = thread * chop + std::min((uint64_t) thread, r);
            uint64_t piece = chop + (thread < r ? 1 : 0);
            uint64_t end = begin + piece;

            if (piece > 0) {
                uint64_t foot_step[RE_MAX_DIMS] = {0};
                get_cursor(begin, foot_step, react.shape, react.ndims);

                uint64_t off[2] = {0, 0};
                for (uint64_t d = 0; d < react.ndims; ++d) {
                    off[0] += foot_step[d] * react.strides[0][d];
                    off[1] += foot_step[d] * react.strides[1][d];
                }

                auto phys_out = static_cast<T *>(react.phys[0]);
                auto phys_in = static_cast<T *>(react.phys[1]);

                for (uint64_t i = begin; i < end; ++i) {
                    phys_out[off[0]] = op(phys_in[off[1]]);

                    for (long long dim = react.ndims - 1; dim >= 0; --dim) {
                        foot_step[dim]++;
                        off[0] += react.strides[0][dim];
                        off[1] += react.strides[1][dim];
                        if (foot_step[dim] < react.shape[dim]) break;
                        foot_step[dim] = 0;
                        off[0] -= react.shape[dim] * react.strides[0][dim];
                        off[1] -= react.shape[dim] * react.strides[1][dim];
                    }
                }
            }
        }
    }

    template<typename T, typename Func>
    void cpu_elementwise_scalar_invoke(JadeReactor &react, Func op) {
        if (react.is_contiguous) {
            auto out = static_cast<T *>(react.phys[0]);
////////////////////////////////////////////////####{
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) default(none) shared(react, out, op)
#endif
////////////////////////////////////////////////####}
            for (uint64_t i = 0; i < react.num_elements; ++i) {
                out[i] = op(out[i]);
            }
            return;
        }

////////////////////////////////////////////////####{
#if defined(_OPENMP)
#pragma omp parallel default(none) shared(react, op, RE_MAX_DIMS)
#endif
////////////////////////////////////////////////####}
        {
            int thread = 0;
            int num_threads = 1;
////////////////////////////////////////////////####{
#if defined(_OPENMP)
            thread = omp_get_thread_num();
            num_threads = omp_get_num_threads();
#endif
////////////////////////////////////////////////####}
            uint64_t chop = react.num_elements / num_threads;
            uint64_t r = react.num_elements % num_threads;
            uint64_t begin = thread * chop + std::min((uint64_t) thread, r);
            uint64_t piece = chop + (thread < r ? 1 : 0);
            uint64_t end = begin + piece;

            if (piece > 0) {
                uint64_t foot_step[RE_MAX_DIMS] = {0};
                get_cursor(begin, foot_step, react.shape, react.ndims);

                uint64_t off_out = 0;
                for (uint64_t d = 0; d < react.ndims; ++d) {
                    off_out += foot_step[d] * react.strides[0][d];
                }

                auto phys_out = static_cast<T *>(react.phys[0]);

                for (uint64_t i = begin; i < end; ++i) {
                    phys_out[off_out] = op(react.Val);

                    for (long long dim = react.ndims - 1; dim >= 0; --dim) {
                        foot_step[dim]++;
                        off_out += react.strides[0][dim];
                        if (foot_step[dim] < react.shape[dim]) break;
                        foot_step[dim] = 0;
                        off_out -= react.shape[dim] * react.strides[0][dim];
                    }
                }
            }
        }
    }


    template<typename T, typename Func>
    void cpu_elementwise_binary_invoke(JadeReactor &react, Func op) {
        std::string msg = std::format("[CPU Invoker] Performing Contiguous Jade Binary Reaction. Reactor Ndims={}.",
                                      std::to_string(react.ndims));
        LOG_INFO(msg);
        if (react.is_contiguous) {
            auto OUT = static_cast<T *>(react.phys[0]);
            auto A = static_cast<T *>(react.phys[1]);
            auto B = static_cast<T *>(react.phys[2]);
////////////////////////////////////////////////####{
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) default(none) shared(react, OUT, A, B, op)
#endif
////////////////////////////////////////////////####}
            for (uint64_t i = 0; i < react.num_elements; ++i) {
                OUT[i] = op(A[i], B[i]);
            }
            return;
        }

////////////////////////////////////////////////####{
#if defined(_OPENMP)
#pragma omp parallel default(none) shared(react, op, RE_MAX_DIMS)
#endif
////////////////////////////////////////////////####}
        {
            int thread = 0;
            int num_threads = 1;
////////////////////////////////////////////////####{
#if defined(_OPENMP)
            thread = omp_get_thread_num();
            num_threads = omp_get_num_threads();
#endif
////////////////////////////////////////////////####}
            uint64_t chop = react.num_elements / num_threads;
            uint64_t r = react.num_elements % num_threads;
            uint64_t begin = thread * chop + std::min((uint64_t) thread, r);
            uint64_t piece = chop + (thread < r ? 1 : 0);
            uint64_t end = begin + piece;

            if (piece > 0) {
                uint64_t foot_step[RE_MAX_DIMS] = {0};
                get_cursor(begin, foot_step, react.shape, react.ndims);

                uint64_t off[3] = {0, 0, 0};
                for (uint64_t d = 0; d < react.ndims; ++d) {
                    off[0] += foot_step[d] * react.strides[0][d];
                    off[1] += foot_step[d] * react.strides[1][d];
                    off[2] += foot_step[d] * react.strides[2][d];
                }

                auto phys_out = static_cast<T *>(react.phys[0]);
                auto phys_a = static_cast<T *>(react.phys[1]);
                auto phys_b = static_cast<T *>(react.phys[2]);

                for (uint64_t i = begin; i < end; ++i) {
                    phys_out[off[0]] = op(phys_a[off[1]], phys_b[off[2]]);

                    for (long long dim = react.ndims - 1; dim >= 0; --dim) {
                        foot_step[dim]++;
                        off[0] += react.strides[0][dim];
                        off[1] += react.strides[1][dim];
                        off[2] += react.strides[2][dim];
                        if (foot_step[dim] < react.shape[dim]) break;
                        foot_step[dim] = 0;
                        off[0] -= react.shape[dim] * react.strides[0][dim];
                        off[1] -= react.shape[dim] * react.strides[1][dim];
                        off[2] -= react.shape[dim] * react.strides[2][dim];
                    }
                }
            }
        }
    }

    template<typename T>
    void cpu_MatMul_binary_invoke(JadeReactor &react) {
        std::string msg = std::format("[CPU Invoker] Performing Contiguous Jade MatMul Reaction. Reactor Ndims {}{}",
                                      std::to_string(react.ndims), ".");
        LOG_INFO(msg);
        uint64_t M = react.shape[react.ndims - 2];
        uint64_t N = react.shape[react.ndims - 1];
        uint64_t K = react.inner_k;

        uint64_t strOut_m = react.strides[0][react.ndims - 2];
        uint64_t strOut_n = react.strides[0][react.ndims - 1];
        uint64_t strA_m = react.strides[1][react.ndims - 2];
        uint64_t strA_k = react.strides[1][react.ndims - 1];
        uint64_t strB_k = react.strides[2][react.ndims - 2];
        uint64_t strB_n = react.strides[2][react.ndims - 1];

        long long B_ndim = react.ndims - 2;
        uint64_t BATCH = 1;
        for (int i = 0; i < B_ndim; ++i) BATCH *= react.shape[i];

////////////////////////////////////////////////####{
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) default(none) \
        shared(react, M, N, K, B_ndim, BATCH, RE_MAX_DIMS, \
               strOut_m, strOut_n, strA_m, strA_k, strB_k, strB_n)
#endif
////////////////////////////////////////////////####}
        for (uint64_t b = 0; b < BATCH; ++b) {
            uint64_t foot_step[RE_MAX_DIMS] = {0};
            get_cursor(b, foot_step, react.shape, B_ndim);

            uint64_t off_out = 0, off_a = 0, off_b = 0;
            for (int i = 0; i < B_ndim; ++i) {
                off_out += foot_step[i] * react.strides[0][i];
                off_a += foot_step[i] * react.strides[1][i];
                off_b += foot_step[i] * react.strides[2][i];
            }

            auto OUT = static_cast<T *>(react.phys[0]) + off_out;
            auto A = static_cast<T *>(react.phys[1]) + off_a;
            auto B = static_cast<T *>(react.phys[2]) + off_b;

            for (uint64_t i = 0; i < M; ++i)
                for (uint64_t j = 0; j < N; ++j) OUT[i * strOut_m + j * strOut_n] = 0.0f;

            for (uint64_t i = 0; i < M; ++i) {
                for (uint64_t k = 0; k < K; ++k) {
                    double valA = A[i * strA_m + k * strA_k];
                    for (uint64_t j = 0; j < N; ++j) {
                        OUT[i * strOut_m + j * strOut_n] += valA * B[k * strB_k + j * strB_n];
                    }
                }
            }
        }
    }

}