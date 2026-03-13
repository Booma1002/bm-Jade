#pragma once
#include <memory>
#include <string>
#include <iostream>
#include <iomanip>
#include <cstring>
#include "Storage.hpp"
#include <optional>
#include <vector>
#include <type_traits>

namespace bm {
    ////////////////////////////////////////////////////////////
    /////////////////***************************////////////////
    /////////////////**  Jade Class Assets  **////////////////
    /////////////////***************************////////////////
    ////////////////////////////////////////////////////////////
    struct JadeReactor;
    struct Dispatcher;

    /**
     * @brief Exception thrown when jade dimensions or ranks are incompatible for an reaction.
     * Usually triggered during binary reactions, broadcasting attempts, or matrix multiplications
     * where the underlying algebraic rules or loop bounds are violated.
     */
    class ShapeMismatchException : public std::exception {
        std::string msg;
    public:
        ShapeMismatchException(const std::string &message) : msg(message) {
            bm::Logger::get().shutdown();
        }
        virtual const char *what() const noexcept override { return msg.c_str(); }
    };

    class JadeException : public std::exception {
        std::string msg;
    public:
        JadeException(const std::string &message) : msg(message) {
            bm::Logger::get().shutdown();
        }
        virtual const char *what() const noexcept override { return msg.c_str(); }
    };
    /**
     * @brief Exception thrown when slice boundaries or steps are invalid.
     * Specifically guards against unimplemented negative step sizes or out-of-bounds
     * structural transformations.
     */
    class SlicingException : public std::exception {
        std::string msg;
    public:
        SlicingException(const std::string &message) : msg(message) {
            bm::Logger::get().shutdown();
        }
        virtual const char *what() const noexcept override { return msg.c_str(); }
    };
    /**
     * @brief Exception thrown when explicit padding definitions are malformed.
     * Ensures padding parameter packs exactly match the expected dimensions (e.g., providing
     * both a 'before' and 'after' value for every rank).
     */
    class PaddingException : public std::exception {
        std::string msg;
    public:
        explicit PaddingException(std::string message) : msg(std::move(message)) {
            bm::Logger::get().shutdown();
        }
        virtual const char *what() const noexcept override { return msg.c_str(); }
    };
    /**
     * @brief Exception thrown during failed memory allocations or out-of-bounds physical access.
     */
    class MemoryException : public std::exception {
        std::string msg;
    public:
        explicit MemoryException(std::string message) : msg(std::move(message)) {
            bm::Logger::get().shutdown();
        }
        virtual const char *what() const noexcept override { return msg.c_str(); }
    };
    /**
     * @brief Exception thrown when two jades cannot be logically aligned via NumPy-style broadcasting.
     */
    class BroadcastException : public std::exception {
        std::string msg;
    public:
        explicit BroadcastException(std::string message) : msg(std::move(message)) {
            bm::Logger::get().shutdown();
        }
        virtual const char *what() const noexcept override { return msg.c_str(); }
    };
    /**
     * @brief Represents a Python-like slice interval for jade viewing.
     * Used heavily by the overloaded `operator[]` to compute new offsets and strides
     * without moving underlying physical memory.
     * @note Assumptions: Currently assumes `step` is strictly positive. Negative strides
     * are not fully supported and will throw a `SlicingException`.
     */
    struct Slice {
        long long start, stop, step;
        Slice() : start(0ll), stop(INT64_MAX), step(1ll) {}
        Slice(long long s, long long e, long long st = 1ll) : start(s), stop(e), step(st) {}
        static Slice From(long long s) { return {s, INT64_MAX, 1ll}; }
        static Slice To(long long e) { return {0ll, e, 1ll}; }
        static Slice All() { return {0ll, INT64_MAX, 1ll}; }
        static Slice Range(long long s, long long e) { return {s, e, 1ll}; }
    };
    struct NewAxis_t {};
    inline constexpr NewAxis_t NewAxis{};

    ////////////////////////////////////////////////////////////////////
    /////////////////***********************************////////////////
    /////////////////**  Jade Class Initialization  **////////////////
    /////////////////***********************************////////////////
    ////////////////////////////////////////////////////////////////////
    ;
    /**
     * @brief Core multi-dimensional array_like abstraction.
     * The `Jade` class decouples physical memory (`Storage`) from logical layout
     * (`shape`, `strides`, `offset`). This allows zero-copy transformations like slicing,
     * transposing, and reshaping by simply manipulating the metadata.
     * @warning UB Warning: Modifying the underlying `Storage` of a jade will affect
     * all other jade (views) sharing that same memory pointer.
     */
    class Jade{
    /////////////////////////////////////////////////////////////////
    /////////////////*****************************///////////////////
    /////////////////**  Jade Class Settings  **///////////////////
    /////////////////*****************************///////////////////
    /////////////////////////////////////////////////////////////////

    public:
        using array_like = std::unique_ptr<uint64_t []>;
        array_like shape;
        array_like strides;
        uint64_t ndims;
        uint64_t offset=0;
        DType dtype = DType::NONE;

        template<typename... Dims>
        class JadeBuilder;


    private:
        friend struct JadeReactor;
        std::shared_ptr<Storage> memory;
    public:
    ////////////////////////////////////////////////////////////
    /////////////////***************************////////////////
    /////////////////**  Jade Constructors  **////////////////
    /////////////////***************************////////////////
    ////////////////////////////////////////////////////////////
    ;
        /**
         * @brief Allocates a new jade and fills it with a scalar value.
         * Computes required memory capacity from the variadic `dims` pack, allocates
         * a new `Storage` buffer, and initializes all elements to `Val`.
         * @tparam Dims Variadic integer types representing the size of each dimension.
         * @param Val The scalar double value to fill the physical memory with.
         * @param dims The dimension sizes.
         */
        template <typename... Dims>
        explicit Jade(DType dtype, double Val=0.0f, Dims... dims);

        /**
     * @brief Allocates a new jade from a pre-computed shape array_like.
     * Takes ownership of the logical layout by dynamically copying `shape_ptr` and
     * computing the contiguous strides from first principles.
     * @param Val The scalar double value to fill the memory with.
     * @param shape_ptr Pointer to an array_like containing the dimension sizes.
     * @param ndims The rank of the jade.
     */
        Jade(DType dtype, double Val, uint64_t* shape_ptr, uint64_t ndims);

        /**
     * @brief Allocates a new jade and ingests an existing raw double array_like.
     * Creates a new physical `Storage` buffer and copies the data from the provided
     * raw pointer.
     * @warning Assumes the `data` pointer points to a contiguous array_like of size
     * exactly equal to the product of `dimensions...`. Passing a smaller array_like results in a segfault.
     * @tparam Dims Variadic integer sizes.
     * @param data Reference to the raw double pointer to copy from.
     * @param dimensions The dimension sizes.
     */
        template <typename... Dims>
        explicit Jade(DType dtype, const double *& data, Dims... dimensions);

    /**
     * @brief Zero-copy logical reshape of an existing jade.
     * Creates a new `Jade` that shares the exact same `Storage` backend as `other`,
     * but recalculates metadata for the new `dims`.
     * @warning Throws `ShapeMismatchException` if the total physical size of the new
     * dimensions does not perfectly match the original jade's size.
     * @tparam Dims Variadic integer sizes for the new shape.
     * @param other The source jade to reshape.
     * @param dims The new dimension sizes.
     */
        template <typename... Dims>
        explicit Jade (DType dtype, Jade& other, Dims... dims);

        /**
     * @brief Zero-copy copy constructor (Creates a View).
     * Copies the logical metadata (`shape`, `strides`, `offset`) and shares the
     * underlying `Storage` pointer. Mutating the memory of this jade mutates the original.
     * @param other The source jade to view.
     * @usage Used for creating a `Jade View` from existing jades.
     * @specialization non-const (Jade& other)
     */
        Jade (Jade& other);

        /**
    * @brief Zero-copy copy constructor (Creates a View).
    * Copies the logical metadata (`shape`, `strides`, `offset`) and shares the
    * underlying `Storage` pointer. Mutating the memory of this jade mutates the original.
    * @param other The source jade to view.
    * @usage Used for creating a `Jade View` from existing jades.
    * @specialization const (Jade& other)
    */
        Jade(const Jade& other);

        /**
    * @brief Explicit metadata constructor for creating advanced views.
    * Directly injects pre-calculated metadata arrays.
    * @usage Used internally by slicing
    * and advanced padding reactions.
    * @warning Assumes ownership of the `unique_ptr` metadata arrays.
    * @param new_shape Managed array_like of the new logical shape.
    * @param new_stride Managed array_like of the new logical strides.
    * @param new_ndims The rank of the new view.
    * @param new_off The physical memory offset from the shared storage base.
    * @param new_mem Shared pointer to the physical backend.
    */
        Jade(DType dtype, std::unique_ptr<uint64_t[]> new_shape, std::unique_ptr<uint64_t[]> new_stride,
             uint64_t new_ndims, uint64_t new_off, std::shared_ptr<Storage> new_mem);

    ////////////////////////////////////////////////////////////
    /////////////***********************************////////////
    /////////////**  Jade Operator-Overloading  **////////////
    /////////////***********************************////////////
    ////////////////////////////////////////////////////////////
        ;
        /**
     * @brief Element-wise addition dispatcher.
     * Allocates a new, zero-initialized jade matching the current logical shape,
     * then dispatches `OpCode::ADD` to the hardware registry.
     * @warning Assumes both jades have exactly matching shapes (broadcasting not dynamically resolved here).
     * @param other The right-hand reactant.
     * @return A newly allocated jade containing the element-wise sum.
     */
        Jade operator+(const Jade& other) const;

        /**
    * @brief Element-wise addition dispatcher.
    * Allocates a new, zero-initialized jade, and another val-fill-initialized jade
    * matching the current logical shape, then dispatches `OpCode::ADD` to the hardware registry.
    * @param val The value of the right reactant.
    * @return A newly allocated jade containing the element-wise sum.
    */
        Jade operator+(const double & val) const;

        /**
    * @brief Element-wise subtraction dispatcher.
    * Allocates a new, zero-initialized jade, and another val-fill-initialized jade
    * matching the current logical shape, then dispatches `OpCode::ADD` to the hardware registry.
    * @param val The value of the right reactant.
    * @return A newly allocated jade containing the element-wise difference.
    */
        Jade operator-(const uint64_t & val) const;

        /**
     * @brief Element-wise subtraction dispatcher.
     * Allocates a new, zero-initialized jade matching the current logical shape,
     * then dispatches `OpCode::SUB` to the hardware registry.
     * @warning Assumes both jades have exactly matching shapes.
     * @param other The right-hand reactant to subtract.
     * @return A newly allocated jade containing the element-wise difference.
     */
        Jade operator-(const Jade& other) const;
        Jade operator-(const double & val) const;


        /**
     * @brief Batched Matrix Multiplication (GEMM) dispatcher.
     * Resolves batch dimension broadcasting using NumPy rules. Computes the required output
     * shape (Batch..., M, N), allocates a new zero-initialized jade, and dispatches `OpCode::MATMUL`.
     * @throws ShapeMismatchException if trailing dimensions do not align (A.cols != B.rows) or
     * if ranks are strictly less than 2D.
     * @param other The right-hand jade reactant.
     * @return A newly allocated jade containing the matrix product.
     */
        Jade operator*(const Jade &other) const;
        Jade operator*(const double &val) const;
        Jade matmul(const Jade& other) const;

        /**
     * @brief In-place element-wise multiplication.
     * Reassigns the current jade to point to the result of `*this * other`.
     * @warning UB Warning: This drops the original `Storage` reference. Any views pointing to
     * the original memory will not reflect this multiplication; they will retain the old data.
     * @param other The multiplier jade.
     */
        void operator*=(const Jade& other) ;
        void operator*=(const double & val) ;

        /**
     * @brief In-place element-wise subtraction.
     * Reassigns the current jade to point to the result of `*this - other`.
     * @warning UB Warning: Drops original memory reference. Does not mutate the original physical memory.
     * @param other The subtrahend jade.
     */
        void operator-=(const Jade& other);
        void operator-=(const double & val);

        /**
     * @brief In-place element-wise addition.
     * Reassigns the current jade to point to the result of `*this + other`.
     * @warning UB Warning: Drops original memory reference. Does not mutate the original physical memory.
     * @param other The addend jade.
     */
        void operator+=(Jade& other);
        void operator+=(const double & val);

        /**
     * @brief Generates a zero-copy sub-jade view via variadic slicing.
     * Recursively computes a new physical `offset` and modifies `shape` and `strides`
     * based on the provided `Slice` arguments or exact indices. Memory is NOT duplicated.
     * @warning UB Warning: If the number of slice arguments exceeds `ndims`, a
     * `ShapeMismatchException` is thrown. Passing out-of-bounds discrete indices throws `std::out_of_range`.
     * @tparam Args Variadic parameter pack of integers (exact indices) or `Slice` structs.
     * @param args The slicing parameters per dimension.
     * @return A new `Jade` object acting as a view into the original memory.
     */
        template <typename... Args>
        Jade operator[](Args... args) const;
        template <typename T>
        Jade operator[](std::initializer_list<uint64_t> il) const;

        /**
     * @brief Reassigns the jade to become a view of another jade.
     * Safely drops the current `Storage` reference (triggering cleanup if refcount hits 0)
     * and deeply copies the metadata of `other`, adopting its physical backend.
     * @param other The jade to mirror.
     * @return Reference to the updated jade.
     */
        Jade& operator=(const Jade& val) &;
        Jade& operator=(const Jade& val) &&;
        Jade& operator=(const double val);


    ////////////////////////////////////////////////////
    ///////////////**********************///////////////
    ///////////////**  Jade MathOps  **///////////////
    ///////////////**********************///////////////
    ////////////////////////////////////////////////////
    ;
        static Jade sin(const Jade& input);
        static Jade cos(const Jade& input);
        static Jade tan(const Jade& input);
        static Jade exp(const Jade& input);
        static Jade log(const Jade& input);
        static Jade clip(const Jade& input, const double lower, const double upper);

    ///////////////////////////////////////////////////////
    ///////////////*************************///////////////
    ///////////////**  Jade Reductions  **///////////////
    ///////////////*************************///////////////
    ///////////////////////////////////////////////////////
    ;
        static Jade std(const Jade& input, std::initializer_list<uint64_t> axes = {});
        static Jade var(const Jade& input, std::initializer_list<uint64_t> axes = {});
        static Jade mean(const Jade& input, std::initializer_list<uint64_t> axes = {});
        static Jade max(const Jade& input, std::initializer_list<uint64_t> ilist = {});
        static Jade min(const Jade& input, std::initializer_list<uint64_t> axes = {});
        static Jade argmax(const Jade& input, std::initializer_list<uint64_t> axes = {});
        static Jade argmin(const Jade& input, std::initializer_list<uint64_t> axes ={});
        Jade dot(const Jade& input) const;

        ////////////////////////////////////////////////////////////
    ///////////////******************************///////////////
    ///////////////**  Jade Transformations  **///////////////
    ///////////////******************************///////////////
    ////////////////////////////////////////////////////////////
    ;
        /**
     * @brief Performs a zero-copy transpose.
     * Reverses the logical `shape` and `strides` arrays. The physical memory remains
     * untouched, but future access via `get()` or kernels will read the data transposed.
     * @return A new transposed `Jade` view.
     */
        Jade transpose();

        Jade& flatten();

        template<typename... Dims>
        Jade& reshape(Dims... dims);


        /**
     * @brief Core out-of-place padding engine.
     * Calculates a new expanded shape, allocates fresh contiguous memory, and modifies the
     * offset of a temporary view to logically align with the center of the new memory space.
     * It then deep-copies the original data into this center.
     * @param fill_val The value to populate the expanded margins.
     * @param pads array_like of size `ndims * 2` containing the [before, after] pairs per dimension.
     * @return A newly allocated jade containing the padded data.
     */
        Jade pad(double fill_val, const uint64_t* pads) const;


        void reshape_like(const uint64_t* dims, uint64_t* strides, uint64_t N);

        /**
     * @brief Overwrites physical memory using a strided copy kernel.
     * Dispatches `OpCode::COPY` to overwrite the current jade's memory with `other`'s memory.
     * Safely handles contiguous and non-contiguous layouts via the kernel registry.
     * @warning Assumes both jades share the exact same rank and shape. Mutates underlying `Storage`.
     * @param other The source jade to copy data from.
     */
        void copy_from(const Jade& other);

        /**
     * @brief Performs a deep, physical copy of the jade.
     * Allocates a completely independent `Storage` backend and uses `OpCode::COPY` to
     * pull data from the current layout into a densely packed, C-contiguous format.
     * @return A newly allocated, contiguous jade.
     */
        Jade copy() const;


    ///////////////////////////////////////////////////////
    ///////////////*************************///////////////
    ///////////////**  Jade Factories  **////////////////
    ///////////////*************************///////////////
    ///////////////////////////////////////////////////////
    ;
    private:
        static std::optional<uint64_t> globalSeed, localSeed;

    public:
        struct ArrayBuilder {
            DType dtype;
            std::vector<uint64_t> shape;
            uint64_t nelm;
            ArrayBuilder(DType dt, std::vector<uint64_t> sh, uint64_t n)
                    : dtype(dt), shape(std::move(sh)), nelm(n) {};
            Jade operator=(std::initializer_list<double> data);
        };

        template<typename T>
        static void set_seed(T s);

        template<typename T>
        void seed(T s) ;

        static uint64_t getSeed();

        template<typename... Dims>
        static ArrayBuilder array(DType dType, const Dims... dims);

        template<typename... Dims>
        static Jade zeros(DType dType, const Dims... dims);

        template<typename... Dims>
        static Jade ones(DType dType, const Dims... dims);

        template<typename... Dims>
        static Jade full(DType dType, const double val, const Dims... dims);

        static Jade arange(DType dType, Slice range);

        template<typename... Dims>
        static Jade rand(DType dType, const Dims... dims);

        template<typename... Dims>
        static Jade randn(DType dType, const Dims... dims);

        template<typename... Dims>
        static Jade randint(DType dType, double lo, double hi, const Dims... dims);


    /**
     * @brief Creates an identical uninitialized jade with the same logical layout.
     * Allocates a fresh `Storage` buffer filled with zeros, copying the shape and rank
     * of the reference jade.
     * @param other The reference jade.
     * @return A new physical jade.
     */
        static Jade zeros_like(const Jade& other);
        static Jade ones_like(const Jade& other);

        /**
     * @brief Creates an identical uninitialized jade with the same logical layout.
     * Allocates a fresh `Storage` buffer filled with specified value, copying the shape and rank
     * of the reference jade.
     * @param other The reference jade.
     * @param val The fill value.
     * @return A new physical jade.
     */
        static Jade full_like(const Jade& other, const double val);


    ////////////////////////////////////////////////////////////
    ///////////////****************************/////////////////
    ///////////////**  Jade Encapsulation  **/////////////////
    ///////////////****************************/////////////////
    ////////////////////////////////////////////////////////////
        ;
        /**
     * @brief Fetches the total physical bytes allocated in the backend.
     * Reflects the aligned capacity of the `Storage` object, which may be larger than
     * the logical size due to over-allocation or alignment padding.
     * @return Capacity in bytes.
     */
        [[nodiscard]] uint64_t get_capacity() const;

        /**
     * @brief Fetches the logical element count of the jade.
     * Returns the product of the jade's logical shape dimensions. Does not reflect
     * physical byte capacity.
     * @return Total number of logical elements.
     */
        [[nodiscard]] uint64_t get_numel() const;
        [[nodiscard]] uint64_t get_size_physical() const;
        [[nodiscard]] void* data_ptr() const;

    ////////////////////////////////////////////////////////////
    ///////////////////***********************//////////////////
    ///////////////////**  Jade Indexers  **//////////////////
    ///////////////////***********************//////////////////
    ////////////////////////////////////////////////////////////
    ;

        /**
        * @brief Extracts the raw scalar value from a 1-element Jade tensor.
        * @throws std::runtime_error if the tensor contains more than 1 element.
        */
        template<typename T = double>
        T item() const;

        /**
     * @brief Resolves a logical multi-dimensional coordinate to a physical double value.
     * Multiplies each index by its corresponding dimension stride and adds the base `offset`.
     * @warning UB Warning: No bounds checking is performed on the individual indices
     * for performance reasons. Out-of-bounds indices will yield garbage data or segfaults.
     * @tparam Indices Variadic pack of dimensional coordinates.
     * @param indices The discrete logical coordinates.
     * @return The scalar double stored at that location.
     */
        template <typename... Indices>
        double get(Indices... indices) const;

        /**
     * @brief Mutates a specific physical location via logical coordinates.
     * Computes the strided memory address and overwrites it.
     * @warning UB Warning: Mutates the underlying `Storage`. If this jade is a view,
     * the mutation will be visible to all other views sharing the memory.
     * @tparam Indices Variadic pack of dimensional coordinates.
     * @param val The new scalar value.
     * @param indices The discrete logical coordinates.
     */
        template <typename... Indices>
        void set(double val, Indices... indices);


    ////////////////////////////////////////////////////////////
    /////////////////************************///////////////////
    /////////////////**  Jade Utilities  **///////////////////
    /////////////////************************///////////////////
    ////////////////////////////////////////////////////////////
    ;
        /**
     * @brief Calculates the broadcasted shape of two jades following NumPy rules.
     * Compares dimensions starting from the trailing (inner-most) dimension.
     * Two dimensions are compatible if they are equal, or if one of them is 1.
     * @throws BroadcastException if the dimensions are mathematically incompatible.
     * @param A The left jade.
     * @param B The right jade.
     * @return A managed array_like containing the finalized broadcasted shape.
     */
        static std::unique_ptr<uint64_t[]> broadcast(Jade A, Jade B);

        /**
    * @brief Calculates the broadcasted shape of two jades following NumPy rules.
    * Compares dimensions starting from the trailing (inner-most) dimension.
    * Two dimensions are compatible if they are equal, or if one of them is 1.
    * @throws BroadcastException if the dimensions are mathematically incompatible.
    * @param A_shape Pointer to the left reactant's shape array_like.
    * @param A_ndims Rank of the left reactant.
    * @param B_shape Pointer to the right reactant's shape array_like.
    * @param B_ndims Rank of the right reactant.
    * @return A managed array_like containing the finalized broadcasted shape.
    */
        static std::unique_ptr<uint64_t[]> broadcast(uint64_t* A_shape, uint64_t A_ndims, uint64_t* B_shape, uint64_t B_ndims );

        /**
     * @brief Validates if two jades can undergo General Matrix Multiplication (GEMM).
     * Checks if the batch dimensions (everything except the last two) can broadcast,
     * and verifies the standard inner-matrix dimension matching rule: `A.columns == B.rows`.
     * @param A The left jade.
     * @param B The right jade.
     * @return True if MatMul is valid, false otherwise.
     */
        static bool can_matmul(Jade& A, Jade& B);

    private:

    ////////////////////////////////////////////////////////////
    ///////////////////**********************///////////////////
    ///////////////////**  Jade Helpers  **///////////////////
    ///////////////////**********************///////////////////
    ////////////////////////////////////////////////////////////
        ;

        /**
     * @brief Translates a flat physical index back into multidimensional logical coordinates.
     * @usage Used internally for iterator traversals or debug printing where flat iteration
     * requires understanding of the multi-dimensional position.
     * @param linear_idx The flat memory index.
     * @param cursor Output array_like to store the resolved multi-dimensional indices.
     * @param shape Pointer to the shape array_like.
     * @param ndims Rank of the jade.
     */
        static constexpr void get_cursor(uint64_t linear_idx, uint64_t* cursor, const uint64_t* shape, uint64_t ndims) ;

        /**
     * @brief In-place array_like reversal utility.
     * @usage Used heavily for stride calculation and NumPy broadcast alignment where trailing
     * dimensions must be evaluated first.
     * @param arr Pointer to the array_like to reverse.
     * @param N The number of elements in the array_like.
     * @return Pointer to the modified array_like.
     */
        static uint64_t* reverse(uint64_t* arr, uint64_t N);

        /**
     * @brief Forwards capacity requests to the `Storage` backend.
     * @usage Used by kernels (via `JadeReactor`) to trigger re-allocations prior to in-place mutation.
     * @tparam Args Argument types required by `Storage::ensure_capacity`.
     * @param args Arguments forwarded to the allocator.
     */
        template<typename... Args>
        void ensure_capacity(Args... args) const;
        /**
     * @brief Base case for the recursive variadic slice resolver.
     * When all explicit slice arguments are exhausted, this function maps the remaining
     * original dimensions and strides directly into the new view.
     * @param dim The current dimension index.
     * @param ndim_tracker By-ref tracker for the new view's rank.
     * @param offset_tracker By-ref tracker for the physical memory offset.
     * @param shp_2 Output array_like for the new view's shape.
     * @param str_2 Output array_like for the new view's strides.
     */

        void apply_slice(uint64_t dim, uint64_t& ndim_tracker, uint64_t& offset_tracker,
                         uint64_t* shp_2, uint64_t* str_2) const;
        void apply_slice_from_array(uint64_t dim, uint64_t &ndim_tracker, uint64_t &offset_tracker,
                                    uint64_t *shape_out, uint64_t *stride_out,
                                    const uint64_t* axis, size_t N) const;

        /**
     * @brief Recursively processes slice arguments to calculate new bounds.
     * Called by `operator[]`. For integral indices, it collapses the dimension and
     * advances the offset. For `Slice` objects, it recalculates the shape and stride
     * for that dimension based on the requested step.
     * @tparam T Type of the current slice argument (int or Slice).
     * @tparam Rest Remaining variadic slice arguments.
     * @param dim The current dimension being processed.
     * @param ndim_tracker By-ref tracker for the new jade's rank.
     * @param offset_tracker By-ref tracker for the physical memory offset.
     * @param shape_out Output array_like for the new view's shape.
     * @param stride_out Output array_like for the new view's strides.
     * @param cur The current slice token.
     * @param rest The remaining slice tokens.
     */
        template <typename T, typename... Rest>
        void apply_slice(uint64_t dim, uint64_t& ndim_tracker, uint64_t& offset_tracker,
                         uint64_t* shape_out, uint64_t* stride_out, T cur, Rest... rest) const;

        /**
    * @brief Bootstraps logical layouts and calculates strictly contiguous strides.
    * Standard C-contiguous layout generation. Traverses dimensions right-to-left,
    * multiplying the running product of shapes to determine jump sizes.
    * @param sh Pointer to the shape array_like.
    * @param st Pointer to the stride array_like to populate.
    * @param ndims Rank of the jade.
    */
        static void calc_strides(const uint64_t* sh, uint64_t* st, uint64_t ndims);

    ////////////////////////////////////////////////////////////
    ///////////////*****************************////////////////
    ///////////////**  Jade Infrastructure  **////////////////
    ///////////////*****************************////////////////
    ////////////////////////////////////////////////////////////
        ;
        /**
     * @brief Initializes logical shape and stride metadata from variadic dimensions.
     * Calculates contiguous strides right-to-left.
     * @usage Used during initial jade construction.
     * @tparam Dims Variadic integer parameters.
     * @param dimensions The specific sizes of each axis.
     */
        template <typename... Dims>
        void init_metadata(Dims... dimensions);

        void init_metadata_like(const uint64_t* dimensions);

        /**
     * @brief Deep-copies the metadata arrays from another jade.
     * Allocates new `unique_ptr` arrays for shape and strides to ensure logical independence
     * from the parent when creating views.
     * @param other The jade whose metadata will be cloned.
     */
        void clone_metadata(const Jade& other);

    /**
     * @brief Supply the initial physical backend.
     * Calculates total required elements and instantiates a new `Storage` manager.
     */
        void allocate_storage();


    /////////////////////////////////////////////////////////////////
    /////////////////*****************************///////////////////
    /////////////////**  Jade Interpretation  **///////////////////
    /////////////////*****************************///////////////////
    /////////////////////////////////////////////////////////////////
    public:

        /**
     * @brief Serializes the jade data into a human-readable nested string.
     * Defaults to 2 decimal places of precision.
     * @return Formatted string representation.
     */
        [[nodiscard]] std::string display() const;

        /**
     * @brief Serializes the jade data into a human-readable nested string with custom precision.
     * Recursively traverses dimensions using stride math to construct bracketed `[ ... ]`
     * string representations of the layout.
     * @param round Number of decimal places to print for double values.
     * @return Formatted string representation.
     */
        [[nodiscard]] std::string display(uint64_t round, uint64_t threshold) const;

    private:

        /**
     * @brief Flattens an array_like into a comma-separated string enclosed in parentheses.
     * @param Arr Pointer to the array_like to format.
     * @param len Number of elements to read.
     * @return Formatted string representation.
     */
        static std::string present(uint64_t* Arr, uint64_t len);

        /**
     * @brief Internal recursive string builder for multidimensional formatting.
     * Walks the physical layout via logical stride jumps to correctly represent
     * contiguous, strided, and sliced views.
     * @param dim_tracker Current axis being traversed.
     * @param offset_tracker Current physical memory offset.
     * @param round double precision limit.
     * @return Formatted string representation of the current axis.
     */
        [[nodiscard]] std::string display(uint64_t dim_tracker, uint64_t offset_tracker, uint64_t round, const uint64_t threshold) const;

    public:
    /**
    * @brief Generates a brief signature string for the jade.
    * Includes rank, physical type sizes, allocated byte capacity, and logical shape tuples.
    * @return Signature string
    * @example `{Jade\<"2D", 32-bit; 64Bytes> (4,4)}`
    */
        [[nodiscard]] std::string repr() const;
    };

}// namespace bm
#include "../temp/Jade.tpp";
#include "../temp/JadeTransformations.tpp"
#include "../temp/JadeFactories.tpp"