#pragma once

namespace bm {
    ;


////////////////////////////////////////////////////////////
/////////////////***************************////////////////
/////////////////**  Jade Constructors  **////////////////
/////////////////***************************////////////////
////////////////////////////////////////////////////////////

    template<typename... Dims>
    Jade::Jade(DType dtype, double Val, Dims... dims) : ndims(sizeof...(Dims)), dtype(dtype) {
        init_metadata(dims...);
        allocate_storage();
        if (Val != 0.0f) {
            *this = Val;
        }
        std::string msg = std::format("New {}{}{}{}", this->repr(), " Filled with ", std::to_string(Val), "'s.");
        LOG_INFO(msg);
    }

    template<typename... Dims>
    Jade::Jade(DType dtype, const double *&data, Dims... dimensions) : ndims(sizeof...(Dims)), dtype(dtype) {
        // Jade data filler constructor.
        init_metadata(dimensions...);
        allocate_storage();
        std::string msg = std::format("New {}{}", this->repr(), ".");
        LOG_INFO(msg);
    }

    template<typename... Dims>
    Jade::Jade(DType dtype, Jade &other, Dims... dims):
            ndims(sizeof...(Dims)), memory(other.memory), dtype(dtype) {
        uint64_t sz = 1;
        ((sz *= dims), ...); // check size match
        if (other.get_numel() != sz) {
            std::string msg = "Cannot reshape Jade into the given dims.";
            LOG_ERR(msg);
            throw ShapeMismatchException(msg);
        }
        init_metadata(dims...); // initialize new jade
        std::string msg = std::format("Reshaped {}{}{}", other.repr(), " Into ", this->repr());
        LOG_INFO(msg);
    }

//////////////////////////////////////////////////////
/////////////******************************///////////
/////////////**  Jade Transformations  **///////////
/////////////******************************///////////
//////////////////////////////////////////////////////




//////////////////////////////////////////////////////
/////////////*****************************////////////
/////////////**  Jade Infrastructure  **////////////
/////////////*****************************////////////
//////////////////////////////////////////////////////

    template<typename... Args>
    void Jade::ensure_capacity(Args... args) const {
        this->memory->ensure_capacity(args...);
    }


    template<typename... Dims>
    void Jade::init_metadata(Dims... dimensions) {
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
/////////////**  Jade Indexers  **////////////
/////////////***********************////////////
////////////////////////////////////////////////
    template<typename... Args>
    Jade Jade::operator[](Args... args) const {

        int constexpr newaxes = (0 + ... + (std::is_same_v<Args, NewAxis_t> ? 1 : 0));
        if (sizeof...(Args) - newaxes != ndims) {
            std::string msg = "Telemetry dimensions arguments count doesn't match jade's ndims.";
            LOG_ERR(msg);
            throw ShapeMismatchException(msg);
        }

        auto new_shape = std::make_unique<uint64_t[]>(ndims + newaxes);
        auto new_strides = std::make_unique<uint64_t[]>(ndims + newaxes);
        uint64_t new_offset = this->offset;
        uint64_t new_ndims = 0;

        apply_slice(0, new_ndims, new_offset, new_shape.get(), new_strides.get(), args...);

        Jade view(*this);
        view.ndims = new_ndims;
        view.offset = new_offset;
        view.shape = std::move(new_shape);
        view.strides = std::move(new_strides);
        return view;
    }

    template<typename T>
    Jade Jade::operator[](std::initializer_list<uint64_t> il) const {
        auto new_shape = std::make_unique<uint64_t[]>(ndims);
        auto new_strides = std::make_unique<uint64_t[]>(ndims);
        uint64_t new_offset = this->offset;
        uint64_t new_ndims = 0;

        apply_slice_from_array(0, new_ndims, new_offset, new_shape.get(), new_strides.get(), il.begin(), il.size());

        Jade view(*this);
        view.ndims = new_ndims;
        view.offset = new_offset;
        view.shape = std::move(new_shape);
        view.strides = std::move(new_strides);
        return view;
    }



    template<typename T>
    T Jade::item() const {
        if (get_numel() <= 0) {
            std::string msg = "[Jade] Cannot call .item() on a jade with " + std::to_string(get_numel()) + " elements. Must be >= 1.";
            LOG_ERR(msg);
            throw std::runtime_error(msg);
        }
        switch(this->dtype) {
            case DType::FLOAT32: return static_cast<T>(static_cast<float*>(data_ptr())[0]);
            case DType::FLOAT64: return static_cast<T>(static_cast<double*>(data_ptr())[0]);
            case DType::INT32:   return static_cast<T>(static_cast<int32_t*>(data_ptr())[0]);
            case DType::UINT8:   return static_cast<T>(static_cast<uint8_t*>(data_ptr())[0]);
            case DType::UINT16 : return static_cast<T>(static_cast<int16_t*>(data_ptr())[0]);
            case DType::UINT32:  return static_cast<T>(static_cast<uint32_t*>(data_ptr())[0]);
            case DType::INT16:   return static_cast<T>(static_cast<int16_t*>(data_ptr())[0]);
            case DType::UINT64:  return static_cast<T>(static_cast<uint64_t*>(data_ptr())[0]);
            case DType::INT64:   return static_cast<T>(static_cast<int64_t*>(data_ptr())[0]);
            default: return static_cast<T>(0);;
        }
    }

    template<typename... Indices>
    double Jade::get(Indices... indices) const {
        uint64_t IDX = 0;
        size_t id = 0;
        ((IDX += indices * strides[id++]), ...);
        return static_cast<double*>(data_ptr())[IDX];
    }

    template<typename... Indices>
    void Jade::set(const double val, Indices... indices) {
        uint64_t IDX = 0;
        size_t id = 0;
        ((IDX += indices * strides[id++]), ...);
        static_cast<double*>(data_ptr())[IDX] = val;
    }

    template<typename T, typename... Rest>
    void Jade::apply_slice(uint64_t dim, uint64_t &ndim_tracker, uint64_t &offset_tracker,
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
                std::string msg = "Jade index out of range.";
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