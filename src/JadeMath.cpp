#include "header/Jade.hpp"
#include "header/Dispatcher.hpp"
using namespace bm;
Jade Jade::operator+(const Jade& other) const {
    uint64_t max_dims = std::max(this->ndims, other.ndims);
    auto out_shape = Jade::broadcast(*this, other);
    Jade view(this->dtype, 0.0f, out_shape.get(), max_dims);

    Dispatcher::execute_binary(OpCode::ADD, view, *this, other);
    return view;
}


Jade Jade::operator-(const Jade& other) const {
    uint64_t max_dims = std::max(this->ndims, other.ndims);
    auto out_shape = Jade::broadcast(*this, other);
    Jade view(this->dtype, 0.0f, out_shape.get(), max_dims);
    Dispatcher::execute_binary(OpCode::SUB, view, *this, other);
    return view;
}

void Jade::operator+=(Jade& other) {
    *this = *this + other;
}


void Jade::operator-=(const Jade& other) {
    *this = *this - other;
}

void Jade::operator*=(const Jade& other) {
    *this = *this * other;
}

Jade& Jade::operator=(const Jade& other) &{
    if (this != &other) {
        this->ndims = other.ndims;
        this->offset = other.offset;
        this->shape = std::make_unique<uint64_t[]>(ndims);
        this->strides = std::make_unique<uint64_t[]>(ndims);
        std::memcpy(this->shape.get(), other.shape.get(), ndims * sizeof(uint64_t));
        std::memcpy(this->strides.get(), other.strides.get(), ndims * sizeof(uint64_t));
        this->memory = other.memory;
    }
    return *this;
}
Jade& Jade::operator=(const Jade& other) &&{
    this->copy_from(other);
    return *this;
}

Jade Jade::operator*(const Jade &other) const{
    if(!Jade::can_matmul(const_cast<Jade &>(*this), const_cast<Jade &>(other))){
        std::string msg = "Cannot Apply MatMul: " + this->repr() + " @ " + other.repr();
        LOG_ERR(msg);
        throw ShapeMismatchException(msg);
    }

    Jade a = *this;
    Jade b = other;
    std::unique_ptr<uint64_t[]> sh = std::make_unique<uint64_t[]>(ndims);
    if (this->ndims == 1){
        a.reshape(1, shape[0]);
    }
    if (other.ndims == 1){
        b.reshape(shape[0], 1);
    }

    uint64_t M = (a.ndims > 1) ? a.shape[a.ndims - 2] : 1;
    uint64_t N = b.shape[b.ndims - 1];
    uint64_t NA = a.ndims - 2;
    uint64_t NB = b.ndims - 2;
    std::unique_ptr<uint64_t[]> batch_shape;
    uint64_t res = 0;

    if (NA > 0 || NB > 0) {
        batch_shape = Jade::broadcast(a.shape.get(), NA, b.shape.get(), NB);
        res = std::max(NA, NB);
    }
    uint64_t N_OUT = res + 2;
    auto output_shape = std::make_unique<uint64_t[]>(N_OUT);
    if (res > 0) std::memcpy(output_shape.get(), batch_shape.get(), res * sizeof(uint64_t));

    output_shape[N_OUT - 2] = M;
    output_shape[N_OUT - 1] = N;

    Jade view(this->dtype, 0.0f, output_shape.get(), N_OUT);
    Dispatcher::execute_binary(OpCode::MATMUL, view, *this, other);
    view.init_metadata_like(output_shape.get());
    return view;
}



Jade Jade::operator+(const double & val) const {
    Jade other = Jade::full_like(*this, val);
    Jade view = Jade(*this);
    Dispatcher::execute_binary(OpCode::ADD, view, *this, other);
    return view;
}

Jade Jade::operator-(const double & val) const {
    Jade other = Jade::full_like(*this, val);
    Jade view = Jade(*this);
    Dispatcher::execute_binary(OpCode::SUB, view, *this, other);
    return view;
}

void Jade::operator+=(const double & val) {
    *this = *this + val;
}

void Jade::operator-=(const double & val) {
    *this = *this - val;
}


void Jade::operator*=(const double & val) {
    *this = *this * val;
}


Jade& Jade::operator=(const double val) {
    Dispatcher::execute_scalar(OpCode::FILL, *this, val);
    return *this;
}

Jade Jade::operator*(const double & val) const {
    Jade other = Jade::full_like(*this, val);
    Jade view = Jade(*this);
    Dispatcher::execute_binary(OpCode::MUL, view, *this, other);
    return view;
}


Jade Jade::sin(const Jade& input){
    Jade view(input.dtype, 0.0f, input.shape.get(), input.ndims);
    Dispatcher::execute_unary(OpCode::SIN, view, input);
    return view;
};
Jade Jade::cos(const Jade& input){
    Jade view(input.dtype, 0.0f, input.shape.get(), input.ndims);
    Dispatcher::execute_unary(OpCode::COS, view, input);
    return view;
};
Jade Jade::tan(const Jade& input){
    Jade view(input.dtype, 0.0f, input.shape.get(), input.ndims);
    Dispatcher::execute_unary(OpCode::TAN, view, input);
    return view;
};
Jade Jade::exp(const Jade& input){
    Jade view(input.dtype, 0.0f, input.shape.get(), input.ndims);
    Dispatcher::execute_unary(OpCode::EXP, view, input);
    return view;
};
Jade Jade::log(const Jade& input){
    Jade view(input.dtype, 0.0f, input.shape.get(), input.ndims);
    Dispatcher::execute_unary(OpCode::LOG, view, input);
    return view;
};

Jade Jade::clip(const Jade& input, double lower, double upper){
    if(lower > upper) std::swap(lower, upper);
    Jade view(input.dtype, 0.0, input.shape.get(), input.ndims);
    Dispatcher::execute_unary(OpCode::CLIP, view, input, lower, upper);
    return view;
}

