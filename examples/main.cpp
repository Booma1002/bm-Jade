#include "header/Engine.hpp"
using namespace bm;

std::atomic_uint64_t watching;
void see(Jade& t, std::string msg =""){
    std::cout << std::format("\n\n{:x}) {}:-\n",
             watching.load(std::memory_order::relaxed), msg);
    atomic_fetch_add(&watching, 1);
    t.display(2, 1);
}

int main(){
    LOG_INFO("[Engine] Initiating Bare Metal (BM) Ignition Sequence...");
    watching.store(0, std::memory_order_relaxed);
    Jade a(DType::UINT64, 7000, 100, 10, 10);
    bm::Jade b =a;
    see(b);
    b = bm::Jade(a);
    see(b, "SIN");
    b = bm::Jade(a);
    see(b, "COS");
    b = Jade(a);
    see(b, "TAN");
    Jade c = Jade::clip(a, -10, 10);
    b = Jade(c);
    see(b, "LOG");
    b = Jade(a);
    see(b);
    b = Jade::clip(a, -103, 105);
    see(b, "CLIP");
    a = 10000;
    b = Jade::clip(a, -103, 105);
    see(b, "CLIP");
    double y = 7000;
    uint64_t z;
    auto x = static_cast<decltype(z)>(y);
    std::cout << x;

}