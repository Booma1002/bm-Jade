#include "header/Jade.hpp"
#include "header/Dispatcher.hpp"
using namespace bm;
Jade Jade::arange(DType dtype, Slice range) {
    long long start = range.start;
    long long stop = range.stop;
    long long step = range.step;

    if (step == 0) {
        std::string msg = "[JadeFactory] Arange step cannot be zero.";
        LOG_ERR(msg);
        throw std::invalid_argument(msg);
    }
    uint64_t len = 0;
    if (step > 0 && stop > start) {
        len = (stop - start + step - 1) / step;
    } else if (step < 0 && stop < start) {
        len = (start - stop - step - 1) / -step;
    }
    Jade output(dtype, 0.0, len);

    if (len == 0) return output;
  double arg_start = static_cast<double>(start);
    double arg_step  = static_cast<double>(step);

    Dispatcher::execute_unary(OpCode::ARANGE, output, output, arg_start, arg_step);

    return output;
}
