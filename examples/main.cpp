#include "header/Engine.hpp"
using namespace zeza;

std::atomic_uint64_t watching;
void see(Tile& t, std::string msg =""){
    std::cout << std::format("\n\n{:x}) {}:-\n",
             watching.load(std::memory_order::relaxed), msg);
    atomic_fetch_add(&watching, 1);
    t.display(2, 1);
}

int main(){
    watching.store(0, std::memory_order_relaxed);
    Tile a(DType::UINT64, 7000, 100, 10, 10);
    zeza::Tile b =a;
    see(b);
    b = Tile::sin(a);
    see(b, "SIN");
    b = Tile::cos(a);
    see(b, "COS");
    b = Tile::tan(a);
    see(b, "TAN");
    Tile c = Tile::clip(a, -10, 10);
    b = Tile::log(c);
    see(b, "LOG");
    b = Tile(a);
    see(b);
    b = Tile::clip(a, -103, 105);
    see(b, "CLIP");
    a = 10000;
    b = Tile::clip(a, -103, 105);
    see(b, "CLIP");
    double y = 7000;
    uint64_t z;
    auto x = static_cast<decltype(z)>(y);
    std::cout << x;
}