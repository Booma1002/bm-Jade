#pragma once
#include "Jade.hpp"
#include <vector>
#include <memory>
#include <functional>
#include <unordered_set>

namespace bm {
    struct Vein {
        bool is_checkpointed = false;
        bool requires_grad = false;
        Jade grad;
        std::shared_ptr<Vein> parents[3];
        uint8_t num_parents = 0;
        std::function<void()> backward_op;
    };

} // namespace bm