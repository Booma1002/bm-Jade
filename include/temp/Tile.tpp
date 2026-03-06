#pragma once

namespace zeza {
////////////////////////////////////////////////////////////
/////////////////***************************////////////////
/////////////////**  Tile Constructors  **////////////////
/////////////////***************************////////////////
////////////////////////////////////////////////////////////

    template<typename... Dims>
    Tile::Tile(DType dtype, double Val, Dims... dims) : ndims(sizeof...(Dims)), dtype(dtype) {
        init_metadata(dims...);
        allocate_storage();
        if (Val != 0.0f) {
            *this = Val;
        }
        std::string msg = std::format("New {}{}{}{}", this->repr(), " Filled with ", std::to_string(Val), "'s.");
        LOG_INFO(msg);
    }

    template<typename... Dims>
    Tile::Tile(DType dtype, const double *&data, Dims... dimensions) : ndims(sizeof...(Dims)), dtype(dtype) {
        // Tile data filler constructor.
        init_metadata(dimensions...);
        allocate_storage();
        std::string msg = std::format("New {}{}", this->repr(), ".");
        LOG_INFO(msg);
    }

    template<typename... Dims>
    Tile::Tile(DType dtype, Tile &other, Dims... dims):
            ndims(sizeof...(Dims)), memory(other.memory), dtype(dtype) {
        uint64_t sz = 1;
        ((sz *= dims), ...); // check size match
        if (other.get_size() != sz) {
            std::string msg = "Cannot reshape Tile into the given dims.";
            LOG_ERR(msg);
            throw ShapeMismatchException(msg);
        }
        init_metadata(dims...); // initialize new tile
        std::string msg = std::format("Reshaped {}{}{}", other.repr(), " Into ", this->repr());
        LOG_INFO(msg);
    }

//////////////////////////////////////////////////////
/////////////******************************///////////
/////////////**  Tile Transformations  **///////////
/////////////******************************///////////
//////////////////////////////////////////////////////




//////////////////////////////////////////////////////
/////////////*****************************////////////
/////////////**  Tile Infrastructure  **////////////
/////////////*****************************////////////
//////////////////////////////////////////////////////

    template<typename... Args>
    void Tile::ensure_capacity(Args... args) const {
        this->memory->ensure_capacity(args...);
    }


    template<typename... Dims>
    void Tile::init_metadata(Dims... dimensions) {
        shape = std::make_unique<uint64_t[]>(ndims);
        strides = std::make_unique<uint64_t[]>(ndims);
        uint64_t shape_array[] = {static_cast<uint64_t>(dimensions)...};
        for (long i = ndims - 1; i >= 0; --i) {
            shape[i] = shape_array[i];
        }
        calc_strides(shape.get(), strides.get(), ndims);
    }

////////////////////////////////////////////////
/////////////***********************////////////
/////////////**  Tile Indexers  **////////////
/////////////***********************////////////
////////////////////////////////////////////////
    template<typename... Args>
    Tile Tile::operator[](Args... args) const {

        int constexpr newaxes = (0 + ... + (std::is_same_v<Args, NewAxis_t> ? 1 : 0));
        if (sizeof...(Args) - newaxes != ndims) {
            std::string msg = "Telemetry dimensions arguments count doesn't match tile's ndims.";
            LOG_ERR(msg);
            throw ShapeMismatchException(msg);
        }

        auto new_shape = std::make_unique<uint64_t[]>(ndims + newaxes);
        auto new_strides = std::make_unique<uint64_t[]>(ndims + newaxes);
        uint64_t new_offset = this->offset;
        uint64_t new_ndims = 0;

        apply_slice(0, new_ndims, new_offset, new_shape.get(), new_strides.get(), args...);

        Tile view(*this);
        view.ndims = new_ndims;
        view.offset = new_offset;
        view.shape = std::move(new_shape);
        view.strides = std::move(new_strides);
        return view;
    }

    template<typename... Indices>
    double Tile::get(Indices... indices) const {
        if (sizeof...(Indices) != ndims) {
            std::string msg = "Number of indices must match Tile rank.";
            LOG_ERR(msg);
            throw ShapeMismatchException(msg);
        }
        uint64_t IDX = 0;
        size_t id = 0;
        ((IDX += indices * strides[id++]), ...);
        return memory->get<float>(this->offset + IDX);
    }

    template<typename... Indices>
    void Tile::set(const double val, Indices... indices) {
        if (sizeof...(Indices) != ndims) {
            std::string msg = "Number of indices must match Tile rank.";
            LOG_ERR(msg);
            throw ShapeMismatchException(msg);
        }
        uint64_t IDX = 0;
        size_t id = 0;
        ((IDX += indices * strides[id++]), ...);
        memory->set<float>(this->offset + IDX, val);
    }

    template<typename T, typename... Rest>
    void Tile::apply_slice(uint64_t dim, uint64_t &ndim_tracker, uint64_t &offset_tracker,
                           uint64_t *shape_out, uint64_t *stride_out, T cur, Rest... rest) const {
        if constexpr (std::is_same_v<T, NewAxis_t>) {
            shape_out[ndim_tracker] = 1;
            stride_out[ndim_tracker] = 0;
            ndim_tracker++;
            apply_slice(dim, ndim_tracker, offset_tracker, shape_out, stride_out, rest...);
        }
        if constexpr (std::is_integral_v<T>) {
            auto i = static_cast<long long>(cur);
            if (i < 0) i += shape[dim];
            if (i < 0 || static_cast<uint64_t>(i) >= shape[dim]) {
                std::string msg = "Tile index out of range.";
                LOG_ERR(msg);
                throw SlicingException(msg);
            }
            offset_tracker += i * strides[dim];
            apply_slice(dim + 1, ndim_tracker, offset_tracker, shape_out, stride_out, rest...);
        }
            // case slicing a range Slice
        else if constexpr (std::is_same_v<T, Slice>) {
            long long l = cur.start;
            long long r = cur.stop;
            long long jmp = cur.step;
            uint64_t size = shape[dim];
            if (jmp > 0) {
                if (l < 0) l += size;
                if (r < 0) r += size;
                if (l < 0) l = 0;
                if (r > static_cast<long long>(size)) r = size;
                if (l > r) l = r;
            } else {
                std::string msg = "Negative stride slicing not yet fully implemented :)";
                LOG_ERR(msg);
                throw SlicingException(msg);
            }

            uint64_t len = 0;
            if (jmp > 0) len = (r - l + jmp - 1) / jmp;
            offset_tracker += l * strides[dim];
            shape_out[ndim_tracker] = len;
            stride_out[ndim_tracker] = strides[dim] * jmp;
            ndim_tracker++;
            apply_slice(dim + 1, ndim_tracker, offset_tracker, shape_out, stride_out, rest...);
        }
    }

}