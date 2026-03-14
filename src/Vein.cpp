#include "header/Jade.hpp"
#include "header/Vein.hpp"

namespace bm {

    Jade Jade::grad() const {
        if (!vein) {
            std::string msg = "Cannot fetch grad: Jade requires_grad is false.";
            LOG_ERR(msg);
            throw JadeException(msg);
        }
        return vein->grad;
    }

    void Jade::form_veining(std::vector<std::shared_ptr<Vein>>& veining,
                            std::unordered_set<std::shared_ptr<Vein>>& traversed,
                            const std::shared_ptr<Vein>& vein) const {

        if (vein && traversed.find(vein) == traversed.end()) {
            traversed.insert(vein);
            for (uint8_t i = 0; i < vein->num_parents; ++i) {
                // form in-place dag cascade
                form_veining(veining, traversed, vein->parents[i]);
            }
            veining.push_back(vein);
        }
    }

    void Jade::backward() {
        if (!vein || !vein->requires_grad) {
            LOG_WARN("Called backward() on a tensor that does not require gradients.");
            return;
        }

        // initial d0/d0 = 1.0
        vein->grad = Jade::ones_like(*this);

        std::vector<std::shared_ptr<Vein>> veining;
        std::unordered_set<std::shared_ptr<Vein>> traversed;
        form_veining(veining, traversed, this->vein);

        // reverse resolution
        for (auto it = veining.rbegin(); it != veining.rend(); ++it) {
            if ((*it)->backward_op) (*it)->backward_op();
        }
    }

} // namespace bm