#include "header/Engine.hpp"
#include <iostream>
#include <cmath>
#include <chrono>

using namespace bm;

void chk(bool c, const char* file = __builtin_FILE(), int line = __builtin_LINE()) {
    if (!c) {
        LOG_ERR_LN("!C; FATAL", file, line);
        throw JadeException("!C; FATAL");
    }
}
#define chk(...) chk(__VA_ARGS__, __FILE__, __LINE__)

void chk_v(double a, double b, const char* file = __builtin_FILE(), int line = __builtin_LINE()) {
    if (std::abs(a - b) > 1e-3) {
        LOG_ERR_LN("A!= B; FATAL", file, line);
        throw JadeException("A!=B; FATAL");
    }
}
#define chk_v(...) chk_v(__VA_ARGS__, __FILE__, __LINE__)

void phase_factories() {
    std::cout << "PHASE 1\n";
    Jade::set_seed(42);

    Jade a = Jade::zeros(DType::FLOAT64, 10, 10, 10);
    chk_v(a.get_numel(), 1000);
    chk_v(Jade::max(a).item<double>(), 0.0);

    Jade b = Jade::ones(DType::FLOAT64, 5, 5);
    chk_v(b.get_numel(), 25);
    chk_v(Jade::mean(b).item<double>(), 1.0);

    Jade c = Jade::arange(DType::FLOAT64, Slice(0, 100, 2));
    chk_v(c.get_numel(), 50);
    chk_v(c.get(49), 98.0);

    Jade d = Jade::randn(DType::FLOAT64, 100, 100);
    chk_v(d.get_numel(), 10000);

    Jade e = Jade::rand(DType::FLOAT64, 50, 50);
    chk(Jade::max(e).item<double>() <= 1.0);
    chk(Jade::min(e).item<double>() >= 0.0);

    Jade f = Jade::randint(DType::FLOAT64, -10, 10, 1000);
    chk(Jade::max(f).item<double>() <= 10.0);
    chk(Jade::min(f).item<double>() >= -10.0);

    Jade g = Jade::full(DType::FLOAT64, 7.5, 4, 4);
    chk_v(Jade::mean(g).item<double>(), 7.5);

    Jade h = Jade::zeros_like(e);
    chk_v(h.get_numel(), 2500);
    chk_v(Jade::max(h).item<double>(), 0.0);

    Jade i = Jade::ones_like(d);
    chk_v(Jade::mean(i).item<double>(), 1.0);

    Jade j = Jade::full_like(c, 3.14);
    chk_v(Jade::max(j).item<double>(), 3.14);

    Jade k = Jade::array(DType::FLOAT64, 2, 3) = {1, 2, 3, 4, 5, 6};
    chk_v(k.get(1, 2), 6.0);
}

void phase_math() {
    std::cout << "PHASE 2\n";
    Jade a = Jade::array(DType::FLOAT64, 4) = {1, 2, 3, 4};
    Jade b = Jade::array(DType::FLOAT64, 4) = {10, 20, 30, 40};

    // ping
    Jade c = a + b;
    chk_v(c.get(3), 44.0);
    // ping
    Jade d = b - a;
    chk_v(d.get(0), 9.0);

    // ping
    Jade e = a.transpose() * b;
    a.display();
    std::cout << "a\n";
    b.display();
    std::cout << "b\n";
    e.display();
    std::cout << "E\n";
    chk_v(e.get(2), 90.0);

    // ping
    Jade f = a + 5.0;
    chk_v(f.get(0), 6.0);

    Jade g = b - 10.0;
    chk_v(g.get(1), 10.0);

    Jade h = a * 2.0;
    chk_v(h.get(3), 8.0);

    Jade i = Jade::sin(a);
    chk_v(i.get(0), std::sin(1.0));

    Jade j = Jade::cos(a);
    chk_v(j.get(0), std::cos(1.0));

    Jade k = Jade::tan(a);
    chk_v(k.get(0), std::tan(1.0));

    Jade l = Jade::exp(a);
    chk_v(l.get(0), std::exp(1.0));

    Jade m = Jade::log(a);
    chk_v(m.get(0), std::log(1.0));

    Jade n = Jade::clip(b, 15.0, 35.0);
    chk_v(n.get(0), 15.0);
    chk_v(n.get(1), 20.0);
    chk_v(n.get(3), 35.0);
}

void phase_reductions() {
    std::cout << "PHASE 3\n";
    Jade a = Jade::array(DType::FLOAT64, 2, 4) = {
            10, 20, 30, 40,
            50, 60, 70, 80
    };

    chk_v(Jade::max(a).item<double>(), 80.0);
    chk_v(Jade::min(a).item<double>(), 10.0);
    chk_v(Jade::mean(a).item<double>(), 45.0);

    double v = Jade::var(a).item<double>();
    chk_v(v, 600.0);

    double s = Jade::std(a).item<double>();
    chk_v(s, std::sqrt(600.0));

    uint64_t mx_idx = Jade::argmax(a).item<uint64_t>();
    chk(mx_idx == 7);

    uint64_t mn_idx = Jade::argmin(a).item<uint64_t>();
    chk(mn_idx == 0);

    chk_v(Jade::max(a, {0}).item<double>(), 40.0);
    chk_v(Jade::max(a, {1}).item<double>(), 80.0);
    chk_v(Jade::mean(a, {0}).item<double>(), 25.0);
    chk_v(Jade::mean(a, {1}).item<double>(), 65.0);
}

void phase_broadcasting() {
    std::cout << "PHASE 4\n";
    Jade a = Jade::ones(DType::FLOAT64, 4, 4, 4);
    Jade b = Jade::full(DType::FLOAT64, 5.0, 4);

    Jade c = a + b;
    chk_v(Jade::max(c).item<double>(), 6.0);
    chk_v(Jade::min(c).item<double>(), 6.0);

    Jade d = Jade::full(DType::FLOAT64, 10.0, 4, 1, 4);
    Jade e = c + d;
    chk_v(Jade::max(e).item<double>(), 16.0);

    Jade f = a * b;
    chk_v(Jade::mean(f).item<double>(), 5.0);
}

void phase_matmul() {
    std::cout << "PHASE 5\n";
    Jade a = Jade::array(DType::FLOAT64, 2, 3) = {
            1, 2, 3,
            4, 5, 6
    };

    Jade b = Jade::array(DType::FLOAT64, 3, 2) = {
            7, 8,
            9, 10,
            11, 12
    };

    Jade c = a.dot(b);
    chk(c.shape[0] == 2 && c.shape[1] == 2);
    chk_v(c.get(0, 0), 58.0);
    chk_v(c.get(0, 1), 64.0);
    chk_v(c.get(1, 0), 139.0);
    chk_v(c.get(1, 1), 154.0);

    Jade d = Jade::ones(DType::FLOAT64, 10, 5, 4);
    Jade e = Jade::ones(DType::FLOAT64, 4, 6);
    Jade f = d.dot(e);

    chk(f.ndims == 3);
    chk(f.shape[0] == 10 && f.shape[1] == 5 && f.shape[2] == 6);
    chk_v(f.get(9, 4, 5), 4.0);
}

void phase_transformations() {
    std::cout << "PHASE 6\n";
    Jade a = Jade::arange(DType::FLOAT64, Slice(0, 24));
    a.reshape(2, 3, 4);

    chk(a.ndims == 3);
    chk(a.shape[0] == 2 && a.shape[1] == 3 && a.shape[2] == 4);
    chk_v(a.get(1, 2, 3), 23.0);

    Jade b = a.transpose();
    chk(b.shape[0] == 4 && b.shape[1] == 3 && b.shape[2] == 2);
    chk_v(b.get(3, 2, 1), 23.0);

    Jade c = a.copy();
    chk(c.data_ptr() != a.data_ptr());
    chk_v(c.get(1, 2, 3), 23.0);

    c.set(999.0, 1, 2, 3);
    chk_v(c.get(1, 2, 3), 999.0);
    chk_v(a.get(1, 2, 3), 23.0);

    a.flatten();
    chk(a.ndims == 1);
    chk(a.shape[0] == 24);

    uint64_t p[] = {1, 1};
    Jade x = Jade::ones(DType::FLOAT64, 2);
    Jade y = x.pad(0.0, p);
    chk(y.shape[0] == 4);
    chk_v(y.get(0), 0.0);
    chk_v(y.get(1), 1.0);
    chk_v(y.get(2), 1.0);
    chk_v(y.get(3), 0.0);
}

void phase_mutation() {
    std::cout << "PHASE 7\n";
    Jade a = Jade::ones(DType::FLOAT64, 10, 10);
    Jade b = a[Slice(0, 5), Slice(0, 5)];

    b += 9.0;
    chk_v(a.get(0, 0), 10.0);
    chk_v(a.get(9, 9), 1.0);

    b *= 2.0;
    chk_v(a.get(0, 0), 20.0);

    b -= 5.0;
    chk_v(a.get(0, 0), 15.0);
}

void phase_stress() {
    std::cout << "PHASE 8\n";
    Jade seq = Jade::arange(DType::FLOAT64, Slice(0, 1000000));
    seq.reshape(10, 100, 1000);

    Jade w = Jade::ones(DType::FLOAT64, 1000, 50);
    Jade out = seq.dot(w);

    chk(out.shape[0] == 10 && out.shape[1] == 100 && out.shape[2] == 50);

    out += 5.5;
    Jade red = Jade::mean(out);
    chk(red.item<double>() > 0.0);

    Jade view = out.transpose();
    view.flatten();
    uint64_t idx = Jade::argmax(view).item<uint64_t>();
    chk(idx < 500000);
}

int main() {
    std::cout << "IGNITION\n";
    auto t1 = std::chrono::high_resolution_clock::now();

    phase_factories();
    phase_math();
    phase_reductions();
    phase_broadcasting();
    phase_matmul();
    phase_transformations();
    phase_mutation();
    phase_stress();

    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = t2 - t1;
    std::cout << "TERMINATED " << diff.count() << "s\n";


    return 0;
}