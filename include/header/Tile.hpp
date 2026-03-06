#pragma once
#include <memory>
#include <string>
#include <iostream>
#include <iomanip>
#include <cstring>
#include "Storage.hpp"
namespace zeza {
    ////////////////////////////////////////////////////////////
    /////////////////***************************////////////////
    /////////////////**  Tile Class Assets  **////////////////
    /////////////////***************************////////////////
    ////////////////////////////////////////////////////////////
    struct TileOperator;
    struct Dispatcher;

    /**
     * @brief Exception thrown when tile dimensions or ranks are incompatible for an operation.
     * Usually triggered during binary operations, broadcasting attempts, or matrix multiplications
     * where the underlying algebraic rules or loop bounds are violated.
     */
    class ShapeMismatchException : public std::exception {
        std::string msg;
    public:
        ShapeMismatchException(const std::string &message) : msg(message) {}
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
        SlicingException(const std::string &message) : msg(message) {}
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
        explicit PaddingException(std::string message) : msg(std::move(message)) {}
        virtual const char *what() const noexcept override { return msg.c_str(); }
    };
    /**
     * @brief Exception thrown during failed memory allocations or out-of-bounds physical access.
     */
    class MemoryException : public std::exception {
        std::string msg;
    public:
        explicit MemoryException(std::string message) : msg(std::move(message)) {}
        virtual const char *what() const noexcept override { return msg.c_str(); }
    };
    /**
     * @brief Exception thrown when two tiles cannot be logically aligned via NumPy-style broadcasting.
     */
    class BroadcastException : public std::exception {
        std::string msg;
    public:
        explicit BroadcastException(std::string message) : msg(std::move(message)) {}
        virtual const char *what() const noexcept override { return msg.c_str(); }
    };
    /**
     * @brief Represents a Python-like slice interval for tile viewing.
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
    /////////////////**  Tile Class Initialization  **////////////////
    /////////////////***********************************////////////////
    ////////////////////////////////////////////////////////////////////
    ;
    /**
     * @brief Core multi-dimensional array abstraction.
     * The `Tile` class decouples physical memory (`Storage`) from logical layout
     * (`shape`, `strides`, `offset`). This allows zero-copy transformations like slicing,
     * transposing, and reshaping by simply manipulating the metadata.
     * @warning UB Warning: Modifying the underlying `Storage` of a tile will affect
     * all other tile (views) sharing that same memory pointer.
     */
    class Tile{
    /////////////////////////////////////////////////////////////////
    /////////////////*****************************///////////////////
    /////////////////**  Tile Class Settings  **///////////////////
    /////////////////*****************************///////////////////
    /////////////////////////////////////////////////////////////////

    public:
        using array = std::unique_ptr<uint64_t []>;
        array shape;
        array strides;
        uint64_t ndims;
        uint64_t offset=0;
        DType dtype = DType::NONE;
    private:
        friend struct TileOperator;
        std::shared_ptr<Storage> memory;
    public:
    ////////////////////////////////////////////////////////////
    /////////////////***************************////////////////
    /////////////////**  Tile Constructors  **////////////////
    /////////////////***************************////////////////
    ////////////////////////////////////////////////////////////
    ;
        /**
         * @brief Allocates a new tile and fills it with a scalar value.
         * Computes required memory capacity from the variadic `dims` pack, allocates
         * a new `Storage` buffer, and initializes all elements to `Val`.
         * @tparam Dims Variadic integer types representing the size of each dimension.
         * @param Val The scalar double value to fill the physical memory with.
         * @param dims The dimension sizes.
         */
        template <typename... Dims>
        explicit Tile(DType dtype, double Val=0.0f, Dims... dims);

        /**
     * @brief Allocates a new tile from a pre-computed shape array.
     * Takes ownership of the logical layout by dynamically copying `shape_ptr` and
     * computing the contiguous strides from first principles.
     * @param Val The scalar double value to fill the memory with.
     * @param shape_ptr Pointer to an array containing the dimension sizes.
     * @param ndims The rank of the tile.
     */
        Tile(DType dtype, double Val, uint64_t* shape_ptr, uint64_t ndims);

        /**
     * @brief Allocates a new tile and ingests an existing raw double array.
     * Creates a new physical `Storage` buffer and copies the data from the provided
     * raw pointer.
     * @warning Assumes the `data` pointer points to a contiguous array of size
     * exactly equal to the product of `dimensions...`. Passing a smaller array results in a segfault.
     * @tparam Dims Variadic integer sizes.
     * @param data Reference to the raw double pointer to copy from.
     * @param dimensions The dimension sizes.
     */
        template <typename... Dims>
        explicit Tile(DType dtype, const double *& data, Dims... dimensions);

    /**
     * @brief Zero-copy logical reshape of an existing tile.
     * Creates a new `Tile` that shares the exact same `Storage` backend as `other`,
     * but recalculates metadata for the new `dims`.
     * @warning Throws `ShapeMismatchException` if the total physical size of the new
     * dimensions does not perfectly match the original tile's size.
     * @tparam Dims Variadic integer sizes for the new shape.
     * @param other The source tile to reshape.
     * @param dims The new dimension sizes.
     */
        template <typename... Dims>
        explicit Tile (DType dtype, Tile& other, Dims... dims);

        /**
     * @brief Zero-copy copy constructor (Creates a View).
     * Copies the logical metadata (`shape`, `strides`, `offset`) and shares the
     * underlying `Storage` pointer. Mutating the memory of this tile mutates the original.
     * @param other The source tile to view.
     * @usage Used for creating a `Tile View` from existing tiles.
     * @specialization non-const (Tile& other)
     */
        Tile (Tile& other);

        /**
    * @brief Zero-copy copy constructor (Creates a View).
    * Copies the logical metadata (`shape`, `strides`, `offset`) and shares the
    * underlying `Storage` pointer. Mutating the memory of this tile mutates the original.
    * @param other The source tile to view.
    * @usage Used for creating a `Tile View` from existing tiles.
    * @specialization const (Tile& other)
    */
        Tile(const Tile& other);

        /**
    * @brief Explicit metadata constructor for creating advanced views.
    * Directly injects pre-calculated metadata arrays.
    * @usage Used internally by slicing
    * and advanced padding operations.
    * @warning Assumes ownership of the `unique_ptr` metadata arrays.
    * @param new_shape Managed array of the new logical shape.
    * @param new_stride Managed array of the new logical strides.
    * @param new_ndims The rank of the new view.
    * @param new_off The physical memory offset from the shared storage base.
    * @param new_mem Shared pointer to the physical backend.
    */
        Tile(DType dtype, std::unique_ptr<uint64_t[]> new_shape, std::unique_ptr<uint64_t[]> new_stride,
             uint64_t new_ndims, uint64_t new_off, std::shared_ptr<Storage> new_mem);

    ////////////////////////////////////////////////////////////
    /////////////***********************************////////////
    /////////////**  Tile Operator-Overloading  **////////////
    /////////////***********************************////////////
    ////////////////////////////////////////////////////////////
        ;
        /**
     * @brief Element-wise addition dispatcher.
     * Allocates a new, zero-initialized tile matching the current logical shape,
     * then dispatches `OpCode::ADD` to the hardware registry.
     * @warning Assumes both tiles have exactly matching shapes (broadcasting not dynamically resolved here).
     * @param other The right-hand operand.
     * @return A newly allocated tile containing the element-wise sum.
     */
        Tile operator+(const Tile& other) const;

        /**
    * @brief Element-wise addition dispatcher.
    * Allocates a new, zero-initialized tile, and another val-fill-initialized tile
    * matching the current logical shape, then dispatches `OpCode::ADD` to the hardware registry.
    * @param val The value of the right operand.
    * @return A newly allocated tile containing the element-wise sum.
    */
        Tile operator+(const double & val) const;

        /**
    * @brief Element-wise subtraction dispatcher.
    * Allocates a new, zero-initialized tile, and another val-fill-initialized tile
    * matching the current logical shape, then dispatches `OpCode::ADD` to the hardware registry.
    * @param val The value of the right operand.
    * @return A newly allocated tile containing the element-wise difference.
    */
        Tile operator-(const uint64_t & val) const;

        /**
     * @brief Element-wise subtraction dispatcher.
     * Allocates a new, zero-initialized tile matching the current logical shape,
     * then dispatches `OpCode::SUB` to the hardware registry.
     * @warning Assumes both tiles have exactly matching shapes.
     * @param other The right-hand operand to subtract.
     * @return A newly allocated tile containing the element-wise difference.
     */
        Tile operator-(const Tile& other) const;
        Tile operator-(const double & val) const;


        /**
     * @brief Batched Matrix Multiplication (GEMM) dispatcher.
     * Resolves batch dimension broadcasting using NumPy rules. Computes the required output
     * shape (Batch..., M, N), allocates a new zero-initialized tile, and dispatches `OpCode::MATMUL`.
     * @throws ShapeMismatchException if trailing dimensions do not align (A.cols != B.rows) or
     * if ranks are strictly less than 2D.
     * @param other The right-hand matrix/tile operand.
     * @return A newly allocated tile containing the matrix product.
     */
        Tile operator*(const Tile &other) const;
        Tile operator*(const double &val) const;

        /**
     * @brief In-place element-wise multiplication.
     * Reassigns the current tile to point to the result of `*this * other`.
     * @warning UB Warning: This drops the original `Storage` reference. Any views pointing to
     * the original memory will not reflect this multiplication; they will retain the old data.
     * @param other The multiplier tile.
     */
        void operator*=(const Tile& other) ;
        void operator*=(const double & val) ;

        /**
     * @brief In-place element-wise subtraction.
     * Reassigns the current tile to point to the result of `*this - other`.
     * @warning UB Warning: Drops original memory reference. Does not mutate the original physical memory.
     * @param other The subtrahend tile.
     */
        void operator-=(const Tile& other);
        void operator-=(const double & val);

        /**
     * @brief In-place element-wise addition.
     * Reassigns the current tile to point to the result of `*this + other`.
     * @warning UB Warning: Drops original memory reference. Does not mutate the original physical memory.
     * @param other The addend tile.
     */
        void operator+=(Tile& other);
        void operator+=(const double & val);

        /**
     * @brief Generates a zero-copy sub-tile view via variadic slicing.
     * Recursively computes a new physical `offset` and modifies `shape` and `strides`
     * based on the provided `Slice` arguments or exact indices. Memory is NOT duplicated.
     * @warning UB Warning: If the number of slice arguments exceeds `ndims`, a
     * `ShapeMismatchException` is thrown. Passing out-of-bounds discrete indices throws `std::out_of_range`.
     * @tparam Args Variadic parameter pack of integers (exact indices) or `Slice` structs.
     * @param args The slicing parameters per dimension.
     * @return A new `Tile` object acting as a view into the original memory.
     */
        template <typename... Args>
        Tile operator[](Args... args) const;

        /**
     * @brief Reassigns the tile to become a view of another tile.
     * Safely drops the current `Storage` reference (triggering cleanup if refcount hits 0)
     * and deeply copies the metadata of `other`, adopting its physical backend.
     * @param other The tile to mirror.
     * @return Reference to the updated tile.
     */
        Tile& operator=(const Tile& val) &;
        Tile& operator=(const Tile& val) &&;
        Tile& operator=(const double val);


    ////////////////////////////////////////////////////
    ///////////////**********************///////////////
    ///////////////**  Tile MathOps  **///////////////
    ///////////////**********************///////////////
    ////////////////////////////////////////////////////
    ;
        static Tile sin(const Tile& input);
        static Tile cos(const Tile& input);
        static Tile tan(const Tile& input);
        static Tile exp(const Tile& input);
        static Tile log(const Tile& input);
        static Tile clip(const Tile& input, const double lower, const double upper);

    ///////////////////////////////////////////////////////
    ///////////////*************************///////////////
    ///////////////**  Tile Reductions  **///////////////
    ///////////////*************************///////////////
    ///////////////////////////////////////////////////////
    ;
        Tile& std(const Tile& input);
        Tile& mean(const Tile& input);
        Tile& max(const Tile& input);
        Tile& min(const Tile& input);
        Tile& argmax(const Tile& input);
        Tile& argmin(const Tile& input);
        Tile& dot(const Tile& input);

    ////////////////////////////////////////////////////////////
    ///////////////******************************///////////////
    ///////////////**  Tile Transformations  **///////////////
    ///////////////******************************///////////////
    ////////////////////////////////////////////////////////////
    ;
        /**
     * @brief Performs a zero-copy transpose.
     * Reverses the logical `shape` and `strides` arrays. The physical memory remains
     * untouched, but future access via `get()` or kernels will read the data transposed.
     * @return A new transposed `Tile` view.
     */
        Tile transpose();


        Tile& flatten();


        template<typename... Dims>
        Tile& reshape(Dims... dims);


        /**
     * @brief Core out-of-place padding engine.
     * Calculates a new expanded shape, allocates fresh contiguous memory, and modifies the
     * offset of a temporary view to logically align with the center of the new memory space.
     * It then deep-copies the original data into this center.
     * @param fill_val The value to populate the expanded margins.
     * @param pads array of size `ndims * 2` containing the [before, after] pairs per dimension.
     * @return A newly allocated tile containing the padded data.
     */
        Tile pad(double fill_val, const uint64_t* pads) const;


        void reshape_like(const uint64_t* dims, uint64_t* strides, uint64_t N);

        /**
     * @brief Overwrites physical memory using a strided copy kernel.
     * Dispatches `OpCode::COPY` to overwrite the current tile's memory with `other`'s memory.
     * Safely handles contiguous and non-contiguous layouts via the kernel registry.
     * @warning Assumes both tiles share the exact same rank and shape. Mutates underlying `Storage`.
     * @param other The source tile to copy data from.
     */
        void copy_from(const Tile& other);

        /**
     * @brief Performs a deep, physical copy of the tile.
     * Allocates a completely independent `Storage` backend and uses `OpCode::COPY` to
     * pull data from the current layout into a densely packed, C-contiguous format.
     * @return A newly allocated, contiguous tile.
     */
        Tile copy();


    ///////////////////////////////////////////////////////
    ///////////////*************************///////////////
    ///////////////**  Tile Factories  **////////////////
    ///////////////*************************///////////////
    ///////////////////////////////////////////////////////
    ;

        template<typename... Dims>
        Tile& zeros(const Dims... dims);

        template<typename... Dims>
        Tile& ones(const Dims... dims);

        Tile& arange(Slice range);

        template<typename... Dims>
        Tile& Array(const Dims... dims);

        template<typename... Dims>
        Tile& rand(const Dims... dims);

        template<typename... Dims>
        Tile& randn(const Dims... dims);

        template<typename... Dims>
        Tile& randint(const Dims... dims);


    /**
     * @brief Creates an identical uninitialized tile with the same logical layout.
     * Allocates a fresh `Storage` buffer filled with zeros, copying the shape and rank
     * of the reference tile.
     * @param other The reference tile.
     * @return A new physical tile.
     */
        static Tile zeros_like(const Tile& other);

        /**
     * @brief Creates an identical uninitialized tile with the same logical layout.
     * Allocates a fresh `Storage` buffer filled with specified value, copying the shape and rank
     * of the reference tile.
     * @param other The reference tile.
     * @param val The fill value.
     * @return A new physical tile.
     */
        static Tile fill_like(const Tile& other, const double val);


    ////////////////////////////////////////////////////////////
    ///////////////****************************/////////////////
    ///////////////**  Tile Encapsulation  **/////////////////
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
     * @brief Fetches the logical element count of the tile.
     * Returns the product of the tile's logical shape dimensions. Does not reflect
     * physical byte capacity.
     * @return Total number of logical elements.
     */
        [[nodiscard]] uint64_t get_size() const;
        [[nodiscard]] uint64_t get_size_physical() const;
        [[nodiscard]] void* data_ptr() const;

    ////////////////////////////////////////////////////////////
    ///////////////////***********************//////////////////
    ///////////////////**  Tile Indexers  **//////////////////
    ///////////////////***********************//////////////////
    ////////////////////////////////////////////////////////////
    ;
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
     * @warning UB Warning: Mutates the underlying `Storage`. If this tile is a view,
     * the mutation will be visible to all other views sharing the memory.
     * @tparam Indices Variadic pack of dimensional coordinates.
     * @param val The new scalar value.
     * @param indices The discrete logical coordinates.
     */
        template <typename... Indices>
        void set(double val, Indices... indices);


    ////////////////////////////////////////////////////////////
    /////////////////************************///////////////////
    /////////////////**  Tile Utilities  **///////////////////
    /////////////////************************///////////////////
    ////////////////////////////////////////////////////////////
    ;
        /**
     * @brief Calculates the broadcasted shape of two tiles following NumPy rules.
     * Compares dimensions starting from the trailing (inner-most) dimension.
     * Two dimensions are compatible if they are equal, or if one of them is 1.
     * @throws BroadcastException if the dimensions are mathematically incompatible.
     * @param A The left tile.
     * @param B The right tile.
     * @return A managed array containing the finalized broadcasted shape.
     */
        static std::unique_ptr<uint64_t[]> broadcast(Tile A, Tile B);

        /**
    * @brief Calculates the broadcasted shape of two tiles following NumPy rules.
    * Compares dimensions starting from the trailing (inner-most) dimension.
    * Two dimensions are compatible if they are equal, or if one of them is 1.
    * @throws BroadcastException if the dimensions are mathematically incompatible.
    * @param A_shape Pointer to the left operand's shape array.
    * @param A_ndims Rank of the left operand.
    * @param B_shape Pointer to the right operand's shape array.
    * @param B_ndims Rank of the right operand.
    * @return A managed array containing the finalized broadcasted shape.
    */
        static std::unique_ptr<uint64_t[]> broadcast(uint64_t* A_shape, uint64_t A_ndims, uint64_t* B_shape, uint64_t B_ndims );

        /**
     * @brief Validates if two tiles can undergo General Matrix Multiplication (GEMM).
     * Checks if the batch dimensions (everything except the last two) can broadcast,
     * and verifies the standard inner-matrix dimension matching rule: `A.columns == B.rows`.
     * @param A The left tile.
     * @param B The right tile.
     * @return True if MatMul is valid, false otherwise.
     */
        static bool can_matmul(Tile& A, Tile& B);

    private:

    ////////////////////////////////////////////////////////////
    ///////////////////**********************///////////////////
    ///////////////////**  Tile Helpers  **///////////////////
    ///////////////////**********************///////////////////
    ////////////////////////////////////////////////////////////
        ;

        /**
     * @brief Translates a flat physical index back into multidimensional logical coordinates.
     * @usage Used internally for iterator traversals or debug printing where flat iteration
     * requires understanding of the multi-dimensional position.
     * @param linear_idx The flat memory index.
     * @param cursor Output array to store the resolved multi-dimensional indices.
     * @param shape Pointer to the shape array.
     * @param ndims Rank of the tile.
     */
        static constexpr void get_cursor(uint64_t linear_idx, uint64_t* cursor, const uint64_t* shape, uint64_t ndims) ;

        /**
     * @brief In-place array reversal utility.
     * @usage Used heavily for stride calculation and NumPy broadcast alignment where trailing
     * dimensions must be evaluated first.
     * @param arr Pointer to the array to reverse.
     * @param N The number of elements in the array.
     * @return Pointer to the modified array.
     */
        static uint64_t* reverse(uint64_t* arr, uint64_t N);

        /**
     * @brief Forwards capacity requests to the `Storage` backend.
     * @usage Used by kernels (via `TileOperator`) to trigger re-allocations prior to in-place mutation.
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
     * @param shp_2 Output array for the new view's shape.
     * @param str_2 Output array for the new view's strides.
     */
        void apply_slice(uint64_t dim, uint64_t& ndim_tracker, uint64_t& offset_tracker,
                         uint64_t* shp_2, uint64_t* str_2) const;

        /**
     * @brief Recursively processes slice arguments to calculate new bounds.
     * Called by `operator[]`. For integral indices, it collapses the dimension and
     * advances the offset. For `Slice` objects, it recalculates the shape and stride
     * for that dimension based on the requested step.
     * @tparam T Type of the current slice argument (int or Slice).
     * @tparam Rest Remaining variadic slice arguments.
     * @param dim The current dimension being processed.
     * @param ndim_tracker By-ref tracker for the new tile's rank.
     * @param offset_tracker By-ref tracker for the physical memory offset.
     * @param shape_out Output array for the new view's shape.
     * @param stride_out Output array for the new view's strides.
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
    * @param sh Pointer to the shape array.
    * @param st Pointer to the stride array to populate.
    * @param ndims Rank of the tile.
    */
        static void calc_strides(const uint64_t* sh, uint64_t* st, uint64_t ndims);

    ////////////////////////////////////////////////////////////
    ///////////////*****************************////////////////
    ///////////////**  Tile Infrastructure  **////////////////
    ///////////////*****************************////////////////
    ////////////////////////////////////////////////////////////
        ;
        /**
     * @brief Initializes logical shape and stride metadata from variadic dimensions.
     * Calculates contiguous strides right-to-left.
     * @usage Used during initial tile construction.
     * @tparam Dims Variadic integer parameters.
     * @param dimensions The specific sizes of each axis.
     */
        template <typename... Dims>
        void init_metadata(Dims... dimensions);

        void init_metadata_like(const uint64_t* dimensions);

        /**
     * @brief Deep-copies the metadata arrays from another tile.
     * Allocates new `unique_ptr` arrays for shape and strides to ensure logical independence
     * from the parent when creating views.
     * @param other The tile whose metadata will be cloned.
     */
        void clone_metadata(const Tile& other);

    /**
     * @brief Supply the initial physical backend.
     * Calculates total required elements and instantiates a new `Storage` manager.
     */
        void allocate_storage();


    /////////////////////////////////////////////////////////////////
    /////////////////*****************************///////////////////
    /////////////////**  Tile Interpretation  **///////////////////
    /////////////////*****************************///////////////////
    /////////////////////////////////////////////////////////////////
    public:

        /**
     * @brief Serializes the tile data into a human-readable nested string.
     * Defaults to 2 decimal places of precision.
     * @return Formatted string representation.
     */
        [[nodiscard]] std::string display() const;

        /**
     * @brief Serializes the tile data into a human-readable nested string with custom precision.
     * Recursively traverses dimensions using stride math to construct bracketed `[ ... ]`
     * string representations of the layout.
     * @param round Number of decimal places to print for double values.
     * @return Formatted string representation.
     */
        [[nodiscard]] std::string display(uint64_t round, uint64_t threshold) const;

    private:

        /**
     * @brief Flattens an array into a comma-separated string enclosed in parentheses.
     * @param Arr Pointer to the array to format.
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
    * @brief Generates a brief signature string for the tile.
    * Includes rank, physical type sizes, allocated byte capacity, and logical shape tuples.
    * @return Signature string
    * @example `{Tile\<"2D", 32-bit; 64Bytes> (4,4)}`
    */
        [[nodiscard]] std::string repr() const;
    };

}// namespace zeza