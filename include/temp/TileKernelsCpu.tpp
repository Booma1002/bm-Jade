#pragma once
namespace zeza {
// ==================================================================
// =========={..........CPU K-nary Invoking..........}===============
// ==================================================================
    ;

    template<typename T, typename Func>
    void cpu_elementwise_unary_invoke(TileOperator &oper, Func op) {
        std::string msg = std::format("[CPU Invoker] Performing Contiguous Tile Unary Operation. Operator Ndims={}{}",
                                      std::to_string(oper.ndims), ".");
        LOG_INFO(msg);
        if (oper.is_contiguous) {
            auto out = static_cast<T *>(oper.phys[0]);
            auto in = static_cast<T *>(oper.phys[1]);
////////////////////////////////////////////////####{
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) default(none) shared(oper, out, in, op)
#endif
////////////////////////////////////////////////####}
            for (uint64_t i = 0; i < oper.num_elements; ++i) {
                out[i] = op(in[i]);
            }
            return;
        }

////////////////////////////////////////////////####{
#if defined(_OPENMP)
#pragma omp parallel default(none) shared(oper, op, OPER_MAX_DIMS)
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
            uint64_t chop = oper.num_elements / num_threads;
            uint64_t r = oper.num_elements % num_threads;
            uint64_t begin = thread * chop + std::min((uint64_t) thread, r);
            uint64_t piece = chop + (thread < r ? 1 : 0);
            uint64_t end = begin + piece;

            if (piece > 0) {
                uint64_t foot_step[OPER_MAX_DIMS] = {0};
                get_cursor(begin, foot_step, oper.shape, oper.ndims);

                uint64_t off[2] = {0, 0};
                for (uint64_t d = 0; d < oper.ndims; ++d) {
                    off[0] += foot_step[d] * oper.strides[0][d];
                    off[1] += foot_step[d] * oper.strides[1][d];
                }

                auto phys_out = static_cast<T *>(oper.phys[0]);
                auto phys_in = static_cast<T *>(oper.phys[1]);

                for (uint64_t i = begin; i < end; ++i) {
                    phys_out[off[0]] = op(phys_in[off[1]]);

                    for (long long dim = oper.ndims - 1; dim >= 0; --dim) {
                        foot_step[dim]++;
                        off[0] += oper.strides[0][dim];
                        off[1] += oper.strides[1][dim];
                        if (foot_step[dim] < oper.shape[dim]) break;
                        foot_step[dim] = 0;
                        off[0] -= oper.shape[dim] * oper.strides[0][dim];
                        off[1] -= oper.shape[dim] * oper.strides[1][dim];
                    }
                }
            }
        }
    }

    template<typename T, typename Func>
    void cpu_elementwise_scalar_invoke(TileOperator &oper, Func op) {
        if (oper.is_contiguous) {
            auto out = static_cast<T *>(oper.phys[0]);
////////////////////////////////////////////////####{
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) default(none) shared(oper, out, op)
#endif
////////////////////////////////////////////////####}
            for (uint64_t i = 0; i < oper.num_elements; ++i) {
                out[i] = op(out[i]);
            }
            return;
        }

////////////////////////////////////////////////####{
#if defined(_OPENMP)
#pragma omp parallel default(none) shared(oper, op, OPER_MAX_DIMS)
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
            uint64_t chop = oper.num_elements / num_threads;
            uint64_t r = oper.num_elements % num_threads;
            uint64_t begin = thread * chop + std::min((uint64_t) thread, r);
            uint64_t piece = chop + (thread < r ? 1 : 0);
            uint64_t end = begin + piece;

            if (piece > 0) {
                uint64_t foot_step[OPER_MAX_DIMS] = {0};
                get_cursor(begin, foot_step, oper.shape, oper.ndims);

                uint64_t off_out = 0;
                for (uint64_t d = 0; d < oper.ndims; ++d) {
                    off_out += foot_step[d] * oper.strides[0][d];
                }

                auto phys_out = static_cast<T *>(oper.phys[0]);

                for (uint64_t i = begin; i < end; ++i) {
                    phys_out[off_out] = op(oper.Val);

                    for (long long dim = oper.ndims - 1; dim >= 0; --dim) {
                        foot_step[dim]++;
                        off_out += oper.strides[0][dim];
                        if (foot_step[dim] < oper.shape[dim]) break;
                        foot_step[dim] = 0;
                        off_out -= oper.shape[dim] * oper.strides[0][dim];
                    }
                }
            }
        }
    }


    template<typename T, typename Func>
    void cpu_elementwise_binary_invoke(TileOperator &oper, Func op) {
        std::string msg = std::format("[CPU Invoker] Performing Contiguous Tile Binary Operation. Operator Ndims={}.",
                                      std::to_string(oper.ndims));
        LOG_INFO(msg);
        if (oper.is_contiguous) {
            auto OUT = static_cast<T *>(oper.phys[0]);
            auto A = static_cast<T *>(oper.phys[1]);
            auto B = static_cast<T *>(oper.phys[2]);
////////////////////////////////////////////////####{
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) default(none) shared(oper, OUT, A, B, op)
#endif
////////////////////////////////////////////////####}
            for (uint64_t i = 0; i < oper.num_elements; ++i) {
                OUT[i] = op(A[i], B[i]);
            }
            return;
        }

////////////////////////////////////////////////####{
#if defined(_OPENMP)
#pragma omp parallel default(none) shared(oper, op, OPER_MAX_DIMS)
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
            uint64_t chop = oper.num_elements / num_threads;
            uint64_t r = oper.num_elements % num_threads;
            uint64_t begin = thread * chop + std::min((uint64_t) thread, r);
            uint64_t piece = chop + (thread < r ? 1 : 0);
            uint64_t end = begin + piece;

            if (piece > 0) {
                uint64_t foot_step[OPER_MAX_DIMS] = {0};
                get_cursor(begin, foot_step, oper.shape, oper.ndims);

                uint64_t off[3] = {0, 0, 0};
                for (uint64_t d = 0; d < oper.ndims; ++d) {
                    off[0] += foot_step[d] * oper.strides[0][d];
                    off[1] += foot_step[d] * oper.strides[1][d];
                    off[2] += foot_step[d] * oper.strides[2][d];
                }

                auto phys_out = static_cast<T *>(oper.phys[0]);
                auto phys_a = static_cast<T *>(oper.phys[1]);
                auto phys_b = static_cast<T *>(oper.phys[2]);

                for (uint64_t i = begin; i < end; ++i) {
                    phys_out[off[0]] = op(phys_a[off[1]], phys_b[off[2]]);

                    for (long long dim = oper.ndims - 1; dim >= 0; --dim) {
                        foot_step[dim]++;
                        off[0] += oper.strides[0][dim];
                        off[1] += oper.strides[1][dim];
                        off[2] += oper.strides[2][dim];
                        if (foot_step[dim] < oper.shape[dim]) break;
                        foot_step[dim] = 0;
                        off[0] -= oper.shape[dim] * oper.strides[0][dim];
                        off[1] -= oper.shape[dim] * oper.strides[1][dim];
                        off[2] -= oper.shape[dim] * oper.strides[2][dim];
                    }
                }
            }
        }
    }

    template<typename T>
    void cpu_MatMul_binary_invoke(TileOperator &oper) {
        std::string msg = std::format("[CPU Invoker] Performing Contiguous Tile MatMul Operation. Operator Ndims {}{}",
                                      std::to_string(oper.ndims), ".");
        LOG_INFO(msg);
        uint64_t M = oper.shape[oper.ndims - 2];
        uint64_t N = oper.shape[oper.ndims - 1];
        uint64_t K = oper.inner_k;

        uint64_t strOut_m = oper.strides[0][oper.ndims - 2];
        uint64_t strOut_n = oper.strides[0][oper.ndims - 1];
        uint64_t strA_m = oper.strides[1][oper.ndims - 2];
        uint64_t strA_k = oper.strides[1][oper.ndims - 1];
        uint64_t strB_k = oper.strides[2][oper.ndims - 2];
        uint64_t strB_n = oper.strides[2][oper.ndims - 1];

        long long B_ndim = oper.ndims - 2;
        uint64_t BATCH = 1;
        for (int i = 0; i < B_ndim; ++i) BATCH *= oper.shape[i];

////////////////////////////////////////////////####{
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) default(none) \
        shared(oper, M, N, K, B_ndim, BATCH, OPER_MAX_DIMS, \
               strOut_m, strOut_n, strA_m, strA_k, strB_k, strB_n)
#endif
////////////////////////////////////////////////####}
        for (uint64_t b = 0; b < BATCH; ++b) {
            uint64_t foot_step[OPER_MAX_DIMS] = {0};
            get_cursor(b, foot_step, oper.shape, B_ndim);

            uint64_t off_out = 0, off_a = 0, off_b = 0;
            for (int i = 0; i < B_ndim; ++i) {
                off_out += foot_step[i] * oper.strides[0][i];
                off_a += foot_step[i] * oper.strides[1][i];
                off_b += foot_step[i] * oper.strides[2][i];
            }

            auto OUT = static_cast<T *>(oper.phys[0]) + off_out;
            auto A = static_cast<T *>(oper.phys[1]) + off_a;
            auto B = static_cast<T *>(oper.phys[2]) + off_b;

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