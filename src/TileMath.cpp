#include "header/Tile.hpp"
#include "header/Dispatcher.hpp"
using namespace zeza;
Tile Tile::operator+(const Tile& other) const {
    uint64_t max_dims = std::max(this->ndims, other.ndims);
    auto out_shape = Tile::broadcast(*this, other);
    Tile view(this->dtype, 0.0f, out_shape.get(), max_dims);

    Dispatcher::execute_binary(OpCode::ADD, view, *this, other);
    return view;
}


Tile Tile::operator-(const Tile& other) const {
    uint64_t max_dims = std::max(this->ndims, other.ndims);
    auto out_shape = Tile::broadcast(*this, other);
    Tile view(this->dtype, 0.0f, out_shape.get(), max_dims);
    Dispatcher::execute_binary(OpCode::SUB, view, *this, other);
    return view;
}

void Tile::operator+=(Tile& other) {
    *this = *this + other;
}


void Tile::operator-=(const Tile& other) {
    *this = *this - other;
}

void Tile::operator*=(const Tile& other) {
    *this = *this * other;
}

Tile& Tile::operator=(const Tile& other) &{
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
Tile& Tile::operator=(const Tile& other) &&{
    this->copy_from(other);
    return *this;
}

Tile Tile::operator*(const Tile &other) const{
    if(!Tile::can_matmul(const_cast<Tile &>(*this), const_cast<Tile &>(other))){
        std::string msg = "Cannot Apply MatMul: " + this->repr() + " @ " + other.repr();
        LOG_ERR(msg);
        throw ShapeMismatchException(msg);
    }

    uint64_t M = (this->ndims > 1) ? this->shape[this->ndims - 2] : 1;
    if (this->ndims < 2 || other.ndims < 2){
        std::string msg = "Strict MatMul requires at least 2D tile for now.";
        LOG_ERR(msg);
        throw ShapeMismatchException(msg);
    }

    uint64_t N = other.shape[other.ndims - 1];
    uint64_t NA = this->ndims - 2;
    uint64_t NB = other.ndims - 2;
    std::unique_ptr<uint64_t[]> batch_shape;
    uint64_t res = 0;

    if (NA > 0 || NB > 0) {
        batch_shape = Tile::broadcast(this->shape.get(), NA, other.shape.get(), NB);
        res = std::max(NA, NB);
    }
    uint64_t N_OUT = res + 2;
    auto output_shape = std::make_unique<uint64_t[]>(N_OUT);
    if (res > 0) std::memcpy(output_shape.get(), batch_shape.get(), res * sizeof(uint64_t));

    output_shape[N_OUT - 2] = M;
    output_shape[N_OUT - 1] = N;

    Tile view(this->dtype, 0.0f, output_shape.get(), N_OUT);
    Dispatcher::execute_binary(OpCode::MATMUL, view, *this, other);
    return view;
}



Tile Tile::operator+(const double & val) const {
    Tile other = Tile::fill_like(*this, val);
    Tile view = Tile(*this);
    Dispatcher::execute_binary(OpCode::ADD, view, *this, other);
    return view;
}

Tile Tile::operator-(const double & val) const {
    Tile other = Tile::fill_like(*this, val);
    Tile view = Tile(*this);
    Dispatcher::execute_binary(OpCode::SUB, view, *this, other);
    return view;
}

void Tile::operator+=(const double & val) {
    *this = *this + val;
}

void Tile::operator-=(const double & val) {
    *this = *this - val;
}


void Tile::operator*=(const double & val) {
    *this = *this * val;
}


Tile& Tile::operator=(const double val) {
    Dispatcher::execute_scalar(OpCode::FILL, *this, val);
    return *this;
}

Tile Tile::operator*(const double & val) const {
    Tile other = Tile::fill_like(*this, val);
    Tile view = Tile(*this);
    Dispatcher::execute_binary(OpCode::MUL, view, *this, other);
    return view;
}


Tile Tile::sin(const Tile& input){
    Tile view(input.dtype, 0.0f, input.shape.get(), input.ndims);
    Dispatcher::execute_unary(OpCode::SIN, view, input);
    return view;
};
Tile Tile::cos(const Tile& input){
    Tile view(input.dtype, 0.0f, input.shape.get(), input.ndims);
    Dispatcher::execute_unary(OpCode::COS, view, input);
    return view;
};
Tile Tile::tan(const Tile& input){
    Tile view(input.dtype, 0.0f, input.shape.get(), input.ndims);
    Dispatcher::execute_unary(OpCode::TAN, view, input);
    return view;
};
Tile Tile::exp(const Tile& input){
    Tile view(input.dtype, 0.0f, input.shape.get(), input.ndims);
    Dispatcher::execute_unary(OpCode::EXP, view, input);
    return view;
};
Tile Tile::log(const Tile& input){
    Tile view(input.dtype, 0.0f, input.shape.get(), input.ndims);
    Dispatcher::execute_unary(OpCode::LOG, view, input);
    return view;
};

Tile Tile::clip(const Tile& input, double lower, double upper){
    if(lower > upper) std::swap(lower, upper);
    Tile view(input.dtype, 0.0, input.shape.get(), input.ndims);
    Dispatcher::execute_unary(OpCode::CLIP, view, input, lower, upper);
    return view;
}

