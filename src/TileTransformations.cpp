#include "header/Tile.hpp"
#include "header/Dispatcher.hpp"
using namespace zeza;
Tile  Tile::transpose() {
    Tile newTile(*this);
    array arr = std::make_unique<uint64_t[]>(ndims);
    array strd = std::make_unique<uint64_t[]>(ndims);
    std::memcpy(arr.get(), shape.get(), ndims * sizeof(uint64_t));
    std::memcpy(strd.get(), strides.get(), ndims * sizeof(uint64_t));
    reverse(arr.get(), ndims);
    reverse(strd.get(), ndims);
    std::memcpy(newTile.shape.get(), arr.get(), ndims * sizeof(uint64_t));
    std::memcpy(newTile.strides.get(), strd.get(), ndims * sizeof(uint64_t));
//        std::cout << "Transposed " << this->repr() <<" Into " << newTile.repr() << std::endl;
    return newTile;
}

Tile Tile::zeros_like(const Tile& other) {
    Tile output(other.dtype, 0.0f, other.shape.get(), other.ndims);
    return output;
}

Tile Tile::fill_like(const Tile& other, const double val){
    Tile output(other.dtype, val, other.shape.get(), other.ndims);
    return output;
}

Tile Tile::pad(double fill_val, const uint64_t* pads) const {
    // TODO : UNFINISHED;
    // TODO : NEED TESTING;
    auto new_shape = std::make_unique<uint64_t[]>(ndims);
    for(size_t i=0; i<ndims; ++i)
        new_shape[i] = shape[i] + pads[i*2] + pads[i*2+1];
    Tile output(this->dtype, fill_val, new_shape.get(), ndims);
    Tile view (output);
    for(size_t i=0; i<ndims; ++i) {
        view.offset += pads[i*2] * output.strides[i];
        view.shape[i] = this-> shape[i];
    }

    view.copy_from(*this);
    return output;
}

void Tile::copy_from(const Tile& other) {
    Dispatcher::execute_unary(OpCode::COPY, *this, other);
}

Tile Tile::copy(){
    Tile view = Tile(*this);
    Dispatcher::execute_unary(OpCode::COPY, view, *this);
    return view;
}


Tile& Tile::flatten(){};



