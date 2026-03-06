#include <iostream>
#include <vector>
#include <cmath>
#include "header/Tile.hpp"
#include "header/Dispatcher.hpp"

void assert_eq(float a, float b, std::string msg) {
    if (std::abs(a - b) > 0.001f) {
        throw std::runtime_error("FAIL: " + msg + " | Expected " + std::to_string(b) + ", Got " + std::to_string(a));
    }
}
/*
int main() {
    try {
        std::cout << "\n=== TEST 3: The Contiguous Trap ===\n";
        // 2x2 Tile. Memory: [1, 2, 3, 4]
        Tile t(0.0f, 2, 2);
        t.set(1.0f, 0,0); t.set(2.0f, 0,1);
        t.set(3.0f, 1,0); t.set(4.0f, 1,1);

        // Pad Inner Dim (1) ONLY. Outer Dim (0) is unpadded.
        // This tricks your sort into treating Dim 0 as a contiguous tile.
        std::cout << "Padding Inner Dimension (1) by 1...\n";
        t.pad_inplace(0.0f, 0, 1);
        t.display();

        // Expected Row 0: [0, 1, 2, 0]
        assert_eq(t.get(0,1), 1.0f, "Row 0 Col 1 (Data)");
        assert_eq(t.get(0,2), 2.0f, "Row 0 Col 2 (Data)");

        // Expected Row 1: [0, 3, 4, 0]
        assert_eq(t.get(1,1), 3.0f, "Row 1 Col 1 (Data)");
        assert_eq(t.get(1,2), 4.0f, "Row 1 Col 2 (Data)");

        std::cout << "[PASS] Logic Correct.\n";
    } catch (const std::exception& e) {
        std::cerr << "\n[FAILURE] " << e.what() << "\n";
        return 1;
    }
    return 0;
}
 */