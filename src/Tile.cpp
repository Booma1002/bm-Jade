#include "header/Tile.hpp"
#include "header/Dispatcher.hpp"
#include "header/Logger.hpp"
using namespace zeza;
////////////////////////////////////////////////////////////
/////////////////***************************////////////////
/////////////////**  Tile Constructors  **////////////////
/////////////////***************************////////////////
////////////////////////////////////////////////////////////
;
Tile::Tile(DType dtype, double Val, uint64_t* shape_ptr, uint64_t ndims)
        : ndims(ndims), dtype(dtype) {
    shape = std::make_unique<uint64_t[]>(ndims);
    strides = std::make_unique<uint64_t[]>(ndims);
    std::memcpy(shape.get(), shape_ptr, ndims * sizeof(uint64_t));
    calc_strides(this->shape.get(), strides.get(), ndims);
    allocate_storage();
    if(Val!=0.f){
        *this = Val;
    }
    std::string msg = std::format("New {}{}{}{}", this->repr(), " Filled with ", std::to_string(Val), "'s.");
    LOG_INFO(msg);
}

Tile::Tile(Tile& other):
    ndims(other.ndims), offset(other.offset), memory(other.memory), dtype(other.dtype) {
    clone_metadata(other);
    std::string msg = std::format("Copied {}{}{}", other.repr(), " Into ", this->repr());
    LOG_INFO(msg);
}

Tile::Tile(const Tile& other):
    ndims(other.ndims), offset(other.offset), memory(other.memory), dtype(other.dtype) {
    clone_metadata(other);
    std::string msg = std::format("Copied {}{}{}", other.repr(),  " Into ", this->repr());
    LOG_INFO(msg);
}

Tile::Tile (DType dtype, std::unique_ptr<uint64_t[]> new_shape, std::unique_ptr<uint64_t[]> new_stride,
            uint64_t new_ndims, uint64_t new_off, std::shared_ptr<Storage> new_mem)
    : shape(std::move(new_shape)), strides(std::move(new_stride)),
    ndims(new_ndims), offset(new_off), memory(std::move(new_mem)), dtype(dtype) {
    std::string msg = std::format("Updated metadata to {}{}", this->repr(), ".");
    LOG_INFO(msg);
}



////////////////////////////////////////////////////////////
///////////////****************************/////////////////
///////////////**  Tile Encapsulation  **/////////////////
///////////////****************************/////////////////
////////////////////////////////////////////////////////////

uint64_t Tile::get_capacity() const{
    return memory->capacity();
}

uint64_t Tile::get_size() const{
    uint64_t sz = 1;
    for (size_t i = 0; i < ndims; ++i) sz *= shape[i];
    return sz;
}

uint64_t Tile::get_size_physical() const {
    return memory->size();
}

void* Tile::data_ptr() const {
    return memory->get_offset_ptr(this->offset);
}

////////////////////////////////////////////////////////////
/////////////////************************///////////////////
/////////////////**  Tile Utilities  **///////////////////
/////////////////************************///////////////////
////////////////////////////////////////////////////////////

std::unique_ptr<uint64_t[]> Tile::broadcast(Tile A, Tile B){
    return Tile::broadcast(A.shape.get(), A.ndims, B.shape.get(), B.ndims);
}

std::unique_ptr<uint64_t[]> Tile::broadcast(uint64_t* A_shape, uint64_t A_ndims, uint64_t* B_shape, uint64_t B_ndims ) {
    auto shape1 = std::make_unique<uint64_t[]>(A_ndims);
    auto shape2 = std::make_unique<uint64_t[]>(B_ndims);
    std::memcpy(shape1.get(), A_shape, A_ndims * sizeof(uint64_t));
    std::memcpy(shape2.get(), B_shape, B_ndims * sizeof(uint64_t));
    Tile::reverse(shape1.get(), A_ndims);
    Tile::reverse(shape2.get(), B_ndims);

    auto max_dims = std::max(A_ndims, B_ndims);
    auto shape_out = std::make_unique<uint64_t[]>(max_dims);
    uint64_t i =0;
    for (uint64_t dim = 0; dim <max_dims; ++dim){
        auto a = (i < A_ndims)?shape1[i] : 1;
        auto b = (i < B_ndims)?shape2[i] : 1;

        if(a==b) shape_out[i++] = a;
        else if(a==1) shape_out[i++] = b;
        else if(b==1) shape_out[i++] = a;
        else throw BroadcastException("Can't Broadcast The Given Tiles.");
    }
    reverse(shape_out.get(), max_dims);
    return shape_out;
}

bool Tile::can_matmul(Tile& A, Tile& B){
    if (A.ndims == 0 || B.ndims == 0) return false;
    long long a = (A.ndims>=0)?(long long) A.ndims-2:0;
    long long b = (B.ndims>=0)?(long long) B.ndims-2:0;
    if(a>0 and b>0){
        try{
            Tile::broadcast(A.shape.get(), a, B.shape.get(), b);
        }
        catch (...){
            return false;
        }
    }
    auto a2 = A.shape[A.ndims-1];
    auto b2 = (B.ndims ==1)? 1 : B.shape[B.ndims-2];
    return a2 == b2;
}

////////////////////////////////////////////////////////////
///////////////////**********************///////////////////
///////////////////**  Tile Helpers  **///////////////////
///////////////////**********************///////////////////
////////////////////////////////////////////////////////////

constexpr void Tile::get_cursor(uint64_t linear_idx, uint64_t* cursor, const uint64_t* shape, uint64_t ndims){
    for (uint64_t dim2 = 0; dim2 <ndims; ++dim2) {
        uint64_t dim = ndims - dim2 - 1;
        cursor[dim] = linear_idx % shape[dim];
        linear_idx /= shape[dim];
    }
}

uint64_t* Tile::reverse(uint64_t* arr, const uint64_t N){
    for (size_t i =0; i < N / 2; ++i){
        uint64_t temp = arr[i];
        arr[i] = arr[N - i - 1];
        arr[N - i - 1] = temp;
    }
    return arr;
}

void Tile::apply_slice(uint64_t dim, uint64_t& ndim_tracker, uint64_t& offset_tracker,
                       uint64_t* shp_2, uint64_t* str_2) const {
    for (uint64_t i = dim; i < ndims; ++i) {
        shp_2[ndim_tracker] = shape[i];
        str_2[ndim_tracker] = strides[i];
        ndim_tracker++;
    }
}

void Tile::calc_strides(const uint64_t* sh, uint64_t* st, const uint64_t ndims) {
    uint64_t cur = 1;
    for (long i = ndims - 1; i >= 0; --i) {
        st[i] = cur;
        cur *= sh[i];
    }
}


void Tile::reshape_like(const uint64_t* dims, uint64_t* stride, uint64_t N){
//    uint64_t sz = 1;
//    for (int i = 0; i < N; ++i) {
//        sz*= dims[i];
//    }
//    if (get_size() != sz)
//        throw ShapeMismatchException("Cannot reshape Tile into the given dims.");
    std::string repr1 = repr();
    ndims = N;
    init_metadata_like(dims); // initialize new tile
    for (int i = 0; i < N; ++i) {
        strides[i] = stride[i];
    }
    std::cout << "Reshaped Tile from " << repr1 << " Into " << repr() << std::endl;
}



////////////////////////////////////////////////////////////
///////////////*****************************////////////////
///////////////**  Tile Infrastructure  **////////////////
///////////////*****************************////////////////
////////////////////////////////////////////////////////////

void Tile::clone_metadata(const Tile& other) {
    shape = std::make_unique<uint64_t[]>(ndims);
    strides = std::make_unique<uint64_t[]>(ndims);
    std::memcpy(shape.get(), other.shape.get(), ndims * sizeof(uint64_t));
    std::memcpy(strides.get(), other.strides.get(), ndims * sizeof(uint64_t));
}


void Tile::init_metadata_like(const uint64_t* dimensions) {
    shape = std::make_unique<uint64_t[]>(ndims);
    strides = std::make_unique<uint64_t[]>(ndims);
    for (long i = ndims - 1; i >= 0; --i) {
        shape[i] = dimensions[i];
    }
    calc_strides(shape.get(), strides.get(), ndims);
}


void Tile::allocate_storage() {
    uint64_t size = 1;
    for (size_t i = 0; i < ndims; ++i) size *= shape[i];
    memory = std::make_shared<Storage>(size, get_dtype_size(this->dtype));
    ;
}

/////////////////////////////////////////////////////////////////
/////////////////*****************************///////////////////
/////////////////**  Tile Interpretation  **///////////////////
/////////////////*****************************///////////////////
/////////////////////////////////////////////////////////////////

std::string Tile::display() const{ return display(2, 6);}

std::string Tile::display(uint64_t round, uint64_t threshold) const {
    std::string str = display(0, this->offset, round, threshold);
    std::cout << repr() << std::endl;
    std::cout << str << std::endl;
    return str;
}

std::string Tile::present(uint64_t* Arr, uint64_t len) {
    std::string ret = "(";
    for (size_t i =0; i < len; ++ i) ret+= std::to_string(Arr[i]) + ((i!=len-1)?", ":"");
    ret +=")";
    return ret;
}
std::string Tile::display(const uint64_t dim_tracker, const uint64_t offset_tracker,
                          const uint64_t round, const uint64_t threshold) const {
    auto fetch_scalar = [&](uint64_t idx) {
        double val = 0.0;
        void* ptr = memory->get_offset_ptr(idx);
        switch(this->dtype) {
            case DType::FLOAT32: val = static_cast<double>(*static_cast<float*>(ptr)); break;
            case DType::FLOAT64: val = static_cast<double>(*static_cast<double*>(ptr)); break;
            case DType::INT32:   val = static_cast<double>(*static_cast<int32_t*>(ptr)); break;
            case DType::UINT8:   val = static_cast<double>(*static_cast<uint8_t*>(ptr)); break;
            case DType::UINT16 : val = static_cast<double>(*static_cast<uint16_t*>(ptr)); break;
            case DType::UINT32:  val = static_cast<double>(*static_cast<uint32_t*>(ptr)); break;
            case DType::INT16:   val = static_cast<double>(*static_cast<int16_t*>(ptr)); break;
            case DType::UINT64:  val = static_cast<double>(*static_cast<uint64_t*>(ptr)); break;
            case DType::INT64:   val = static_cast<double>(*static_cast<int64_t*>(ptr)); break;
            default: throw std::runtime_error("Unsupported DType in display");
        }
        std::ostringstream out;
        out << std::fixed << std::setprecision(round) << val;
        return out.str();
    };

    std::string placeholder = "[";
    uint64_t stride = this->strides[dim_tracker];
    bool over = shape[dim_tracker] > threshold;

    uint64_t start_dim = over ? 3 : shape[dim_tracker];
    for (uint64_t i = 0; i < start_dim; ++i) {
        uint64_t IDX = offset_tracker + (i * stride);
        if (dim_tracker < ndims - 1) {
            placeholder += display(dim_tracker + 1, IDX, round, threshold);
        }
        else placeholder += fetch_scalar(IDX);
        if (i < start_dim - 1 || over) {
            placeholder += ", ";
            if (dim_tracker < ndims - 1) placeholder += "\n";
        }
    }
    if (over) {
        placeholder += "..., ";
        if (dim_tracker < ndims - 1) placeholder += "\n";

        uint64_t end_count = 3;
        for (uint64_t i = shape[dim_tracker] - end_count; i < shape[dim_tracker]; ++i) {
            uint64_t IDX = offset_tracker + (i * stride);
            if (dim_tracker < ndims - 1) {
                placeholder += display(dim_tracker + 1, IDX, round, threshold);
            }
            else placeholder += fetch_scalar(IDX);
            if (i < shape[dim_tracker] - 1) {
                placeholder += ", ";
                if (dim_tracker < ndims - 1) placeholder += "\n";
            }
        }
    }

    placeholder += "]";
    return placeholder;
}

std::string Tile::repr() const{
    std::string str = "{Tile<\"" + std::to_string(ndims) + "D\", ";
    str += std::to_string(memory->get_item_size()*8) + "-bit; "
       + std::to_string(get_capacity())+"Bytes> (";
    for (size_t i =0; i < ndims; ++i)
        str += std::to_string((shape[i])) +((i != ndims - 1) ? "," : "");
    str+= ")}";
    return str;
}


