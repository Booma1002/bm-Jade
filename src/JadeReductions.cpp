#include "header/Jade.hpp"
#include "header/Dispatcher.hpp"
using namespace bm;

Jade Jade::std(const Jade& input, std::initializer_list<uint64_t> axes) {
    Jade output(input.dtype, 0.0, 1ULL);
    if(axes .size() == 0)
        Dispatcher::execute_reduction(OpCode::STD, output, input);
    else Dispatcher::execute_reduction(OpCode::STD, output, input[axes]);
    return output;
}

Jade Jade::var(const Jade& input, std::initializer_list<uint64_t> axes) {
    Jade output(input.dtype, 0.0, 1ULL);
    if(axes .size() == 0)
        Dispatcher::execute_reduction(OpCode::VAR, output, input);
    else Dispatcher::execute_reduction(OpCode::VAR, output, input[axes]);
    return output;
}

Jade Jade::max(const Jade& input, std::initializer_list<uint64_t> axes) {
    Jade output(input.dtype, 0.0, 1ULL);
    if(axes .size() == 0)
        Dispatcher::execute_reduction(OpCode::MAX, output, input);
    else Dispatcher::execute_reduction(OpCode::MAX, output, input[axes]);
    return output;
}

Jade Jade::min(const Jade& input, std::initializer_list<uint64_t> axes) {
    Jade output(input.dtype, 0.0, 1ULL);
    if(axes .size() == 0)
        Dispatcher::execute_reduction(OpCode::MIN, output, input);
    else Dispatcher::execute_reduction(OpCode::MIN, output, input[axes]);
    return output;
}

Jade Jade::mean(const Jade& input, std::initializer_list<uint64_t> axes) {
    Jade output(input.dtype, 0.0, 1ULL);
    if(axes .size() == 0)
        Dispatcher::execute_reduction(OpCode::MEAN, output, input);
    else Dispatcher::execute_reduction(OpCode::MEAN, output, input[axes]);
    return output;
}

Jade Jade::dot(const Jade& other) const {
    if (this->ndims != 1 || other.ndims != 1 || this->shape[0] != other.shape[0]) {
        throw ShapeMismatchException("Dot product requires two 1D tensors of identical size.");
    }
    Jade view(this->dtype, 0.0f, static_cast<uint64_t*>(nullptr), static_cast<uint64_t>(0));
    Dispatcher::execute_binary(OpCode::DOT, view, *this, other);
    return view;
}

Jade Jade::argmax(const Jade& input, std::initializer_list<uint64_t> axes) {
    // indexes are uint64_t
    Jade output(DType::UINT64, 0.0, 1ULL);
    if(axes .size() == 0)
        Dispatcher::execute_reduction(OpCode::ARGMAX, output, input);
    else Dispatcher::execute_reduction(OpCode::ARGMAX, output, input[axes]);
    return output;
}

Jade Jade::argmin(const Jade& input, std::initializer_list<uint64_t> axes) {
    Jade output(DType::UINT64, 0.0, 1ULL);
    if(axes .size() == 0)
        Dispatcher::execute_reduction(OpCode::ARGMIN, output, input);
    else Dispatcher::execute_reduction(OpCode::ARGMIN, output, input[axes]);
    return output;
}