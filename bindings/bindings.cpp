#include "header/Tile.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

py::array_t<float> py_convolve(py::array_t<float> input_array, py::array_t<float> kernel_array, int stride) {

    py::buffer_info img_buf = input_array.request();
    float* img_ptr = static_cast<float*>(img_buf.ptr);
    int c = (img_buf.ndim == 3)? img_buf.shape[0] : 1;
    int h = (img_buf.ndim == 3)? img_buf.shape[1] : img_buf.shape[0];
    int w = (img_buf.ndim == 3)? img_buf.shape[2] : img_buf.shape[1];
    Tile image(img_ptr, c, h, w);

    py::buffer_info ker_buf = kernel_array.request();
    float* ker_ptr = static_cast<float*>(ker_buf.ptr);
    int n_filters = (ker_buf.ndim == 4) ? ker_buf.shape[0] : 1;
    int k_in_c = (ker_buf.ndim == 4) ? ker_buf.shape[1] : ((ker_buf.ndim == 3) ? ker_buf.shape[0] : 1);
    int total_kernel_channels = n_filters * k_in_c;
    int kh = ker_buf.shape[ker_buf.ndim - 2];
    int kw = ker_buf.shape[ker_buf.ndim - 1];
    Tile kernel(ker_ptr, total_kernel_channels, kh, kw);

    Tile result = convolve(image, kernel, n_filters, stride);
    auto result_array = py::array_t<float>({(int)result.channels, (int)result.height, (int)result.width});

    py::buffer_info res_buf = result_array.request();
    float* res_ptr = static_cast<float*>(res_buf.ptr);

    result.clone_data(res_ptr);
    return result_array;
}

PYBIND11_MODULE(my_engine, m) {
m.doc() = "C++ Convolution Engine Wrapper 🔥🔥";
m.def("Hazem_Convolution", &py_convolve, "cool stuff. benchmark won scipy 🔥");
}
