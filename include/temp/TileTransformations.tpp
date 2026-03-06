#include "header/Tile.hpp"
namespace zeza {
    template<typename... Dims>
    Tile &Tile::reshape(Dims... dims) {
        uint64_t sz = 1;
        ((sz *= dims), ...); // check size match
        uint64_t n = sizeof...(Dims);
        if (get_size() != sz) {
            LOG_ERR("[Tile] Cannot reshape Tile into the given dims.");
            throw ShapeMismatchException("Cannot reshape Tile into the given dims.");
        }
        std::string repr1 = repr();
        ndims = n;
        init_metadata(dims...); // initialize tile
        std::string msg;
        msg+= std::format("Reshaped Tile from {} Into {}", repr1 ,repr());
        LOG_INFO(msg);
        return *this;
    }
}