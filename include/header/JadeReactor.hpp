#pragma once
#include <cstdint>
#include <array>
#include <algorithm>
#include <utility>
#include <string>
#include "Enums.hpp"

namespace bm {
/////////////////////////////////////////////////////////////////////
/////////////////************************************////////////////
/////////////////**  JadeReactor Struct Assets  **////////////////
/////////////////************************************////////////////
/////////////////////////////////////////////////////////////////////
    ;

    class Jade;

/**
 * @brief Exception wrapper specific to the Reactor module.
 * Thrown primarily during dispatch failures, such as calling an unbound
 * method ID or exceeding the maximum method limit.
 */
    class ReactorException : public std::exception {
        std::string msg;
    public:
        explicit ReactorException(std::string message) : msg(std::move(message)) {}

        [[nodiscard]] const char *what() const noexcept override { return msg.c_str(); }
    };

    enum class ReactorMethod {
        ENSURE_CAPACITY = 0,
        CALC_STRIDES = 1,
        RESHAPE = 2,
        MAX_METHODS [[maybe_unused]] = MAX_RE_METHODS
    };

/**
 * @brief Type trait utility for member function signature deduction.
 * Uses SFINAE/template specialization to strip `const` and `noexcept` qualifiers
 * from member function pointers, allowing `bind_private` to extract the return type
 * and parameter pack for thunk generation.
 */
    template<auto MemberFunc, typename FuncType = decltype(MemberFunc)>
    struct Binder;

/////////////////////////////////////////////////////////////////////////////
/////////////////********************************************////////////////
/////////////////**  JadeReactor Struct Initialization  **////////////////
/////////////////********************************************////////////////
/////////////////////////////////////////////////////////////////////////////
    ;

/**
 * @brief Execution context and dispatcher for jade math kernels.
 * This struct bridges high-level `Jade` objects with low-level compute kernels.
 * It normalizes memory layouts, computes strides across multiple reactants,
 * and holds type-erased function pointers (thunks) for lazy evaluation.
 * @note Assumptions:
 * - Maximum dimensions and reactants are strictly bounded by `RE_MAX_DIMS` and `RE_MAX_REACTANTS`.
 * - Memory pointers (`phys`) assume data is stored as `DType`.
 */
    struct JadeReactor {
///////////////////////////////////////////////////////////////////////
/////////////////**************************************////////////////
/////////////////**  JadeReactor Struct Settings  **////////////////
/////////////////**************************************////////////////
///////////////////////////////////////////////////////////////////////
    public:
        DType dtype = DType::NONE;
        bool is_contiguous = false;
        uint64_t ndims = 0;
        uint64_t num_elements = 0;
        uint64_t inner_k = 0;
        double Val = 0.f;
        double Left = 0.f;
        double Right = 0.f;

        uint64_t shape[RE_MAX_DIMS]{};
        uint64_t strides[RE_MAX_REACTANTS][RE_MAX_DIMS]{};
        void* phys[RE_MAX_REACTANTS]{};
    private:
        using GenericFunc = void (*)();
        void *bound_obj = nullptr;

    public:
        GenericFunc methods[MAX_RE_METHODS] = {nullptr};

////////////////////////////////////////////////////////////////
/////////////////********************************///////////////
/////////////////**  JadeReactor Utilities  **///////////////
/////////////////********************************///////////////
////////////////////////////////////////////////////////////////
        ;


/**
 * @brief Binds a class member function to an execution ID.
 * * Leverages the `Binder` template specialization to deduce the member function's
 * signature at compile-time and generate a type-erased trampoline (thunk).
 * * @tparam MemberFunc A non-type template parameter pointing to the member function.
 * @param id The target `ReactorMethod` slot in the dispatch table.
 */
        template<auto MemberFunc>
        void bind_private(ReactorMethod id);

/**
 * @brief Binds a stateless lambda or free function to an execution ID.
 * @warning UB Warning: The provided lambda MUST be stateless. Stateful lambdas
 * (capturing variables) cannot decay to C-style function pointers and will
 * result in memory corruption or compilation failure.
 * @warning UB Warning: If the bound function expects a context object, `bound_obj`
 * MUST be manually assigned before `call()` is invoked.
 * @tparam Func Inferred type of the callable.
 * @param id The target `ReactorMethod` slot.
 * @param f The stateless callable to bind.
 * @example
 * react.bound_obj = &in_out;\n
 * react.bind(ReactorMethod::ENSURE_CAPACITY, \n
 * +[](void* obj, uint64_t size, double scale, bool f, DType val) {\n
 * static_cast<Jade*>(obj)->ensure_capacity(size, scale, f, val);\n
 * });
 */
        template<typename Func>
        void bind(ReactorMethod id, Func &&f);

/**
* @brief Dispatches the bound function associated with the given ID.
* Re-casts the stored `void(*)()` generic function pointer to a strictly typed
* signature and invokes it with `bound_obj` as the first parameter.
* Automatically updates the primary physical pointer if `bound_obj` is heavily mutated.
* @warning UB Warning: The parameter pack `Args...` must EXACTLY match the signature
* of the bound function. A mismatch will cause an ABI boundary violation and undefined behavior.
* @tparam Args Variadic parameter types passed to the underlying function.
* @param id The execution ID to dispatch.
* @param args The runtime arguments forwarded to the thunk.
* @example
*   react.call(ReactorMethod::CALC_STRIDES, react.shape, react.strides[0], react.ndims);
*/
        template<typename... Args>
        void call(ReactorMethod id, Args... args);

        /**
        * @brief Checks if a specific execution ID has an active binding.
        * @param id The `ReactorMethod` slot to check.
        * @return true if a function is bound to the slot, false otherwise.
        */
        [[nodiscard]] constexpr bool has(ReactorMethod id) const;

/**
 * @brief Generates a type-erased trampoline for member function invocation.
 * Creates a stateless lambda that casts the opaque `void*` context back to the
 * concrete instance type `T`, and invokes the member pointer `MemberPtr`.
 * This allows storing disparate member functions in a unified function pointer array.
 * @tparam T The concrete class type of the context object.
 * @tparam Args The parameter pack expected by the member function.
 * @tparam MemberPtr The pointer-to-member function.
 * @param id The target dispatch slot.
 * @example
 * static void bind(JadeReactor* op, ReactorMethod id) {\n
 *      op->template create_thunk<T, Args..., MemberFunc>(id);\n
 *  }
 */
        template<typename T, typename... Args, auto MemberPtr>
        void create_thunk(ReactorMethod id);

/**
* @brief Greedily collapses adjacent contiguous dimensions.
* Analyzes the stride-to-shape ratios across all bound reactants. If the memory
* layout across an inner/outer dimension pair is perfectly sequential for all reactants,
* the dimensions are fused.
*/
        void merge_dims();

///////////////////////////////////////////////////////////////
/////////////////*******************************///////////////
/////////////////**  JadeReactor Contexts  **///////////////
/////////////////*******************************///////////////
///////////////////////////////////////////////////////////////
        ;

/**
 * @brief Constructs an execution context for a two-reactant jade reaction.
 * Extracts shapes, strides, and physical memory pointers from an output jade
 * and two input jades. It automatically attempts to collapse contiguous dimensions
 * via `merge_dims()` to optimize kernel loop limits.
 * @note Assumptions: Ranks (`ndims`) must perfectly match across all three jades.
 * @warning Edge Case: Broadcasting is currently disabled. Shape mismatches will throw a runtime error.
 * @param out The destination jade.
 * @param a The first input jade.
 * @param b The second input jade.
 * @return A configured `JadeReactor` ready for kernel dispatch.
 */
        static JadeReactor react_binary(Jade &out, const Jade &a, const Jade &b);

/**
 * @brief Constructs an execution context for a single-reactant jade reaction.
 * Normalizes strides and memory layouts between an input and output jade.
 * Fuses memory dimensions to treat multi-dimensional contiguous jade as flat 1D arrays
 * where possible, maximizing memory coalescing during kernel execution.
 * @note Assumptions: Exact shape match is required between `out` and `a`.
 * @param out The destination jade.
 * @param a The source jade.
 * @return A configured `JadeReactor`.
 */
        static JadeReactor react_unary(Jade &out, const Jade &a, const double left = 0.f, const double right = 0.f);

        static JadeReactor react_scalar(Jade &out, double Val);

/**
 * @brief Prepares an execution context for General Matrix Multiplication (GEMM).
 * Identifies the contraction dimension (`inner_k`) and configures batching strides.
 * Safely zeros out strides for broadcasted dimensions (where a shape dimension is 1
 * but the output demands >1).
 * @note Assumptions: The last dimension of `a` is strictly treated as the reduction dimension (`K`).
 * @note Explicitly flags the context as non-contiguous, forcing kernels to rely on stride math.
 * @param out Destination jade for the matrix product.
 * @param a Left reactant jade.
 * @param b Right reactant jade.
 * @return A configured `JadeReactor`.
 */
        static JadeReactor react_matmul(Jade &out, const Jade &a, const Jade &b);
    };

/**
 * @brief Type trait utility for member function signature deduction.
 * @specialization Ret (T::*)(Args...)
 */
    template<auto MemberFunc, typename T, typename Ret, typename... Args>
    struct Binder<MemberFunc, Ret (T::*)(Args...)>;

/**
 * @brief Type trait utility for member function signature deduction.
 * @specialization Ret (T::*)(Args...) const
 */
    template<auto MemberFunc, typename T, typename Ret, typename... Args>
    struct Binder<MemberFunc, Ret (T::*)(Args...) const>;

/**
 * @brief Type trait utility for member function signature deduction.
 * @specialization Ret (T::*)(Args...) noexcept
 */
    template<auto MemberFunc, typename T, typename Ret, typename... Args>
    struct Binder<MemberFunc, Ret (T::*)(Args...) noexcept>;

/**
 * @brief Type trait utility for member function signature deduction.
 * @specialization Ret (T::*)(Args...) const noexcept
 */
    template<auto MemberFunc, typename T, typename Ret, typename... Args>
    struct Binder<MemberFunc, Ret (T::*)(Args...) const noexcept>;


}// namespace bm
#include "temp/JadeReactor.tpp"