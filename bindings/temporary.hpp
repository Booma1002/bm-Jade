#include <unordered_map>
#include <string>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstring>
#include <memory>
#include <chrono>
#include "header/Tile.hpp"


Tensor convolve(Tensor& input, Tensor& kernel, uint64_t n_filters, uint64_t step){
    uint64_t in_h = input.shape[1];
    uint64_t in_w = input.width;
    auto pad_w = (kernel.width - 1) / 2;
    auto pad_h = (kernel.height - 1) / 2;
    auto in_channels = input.channels;
    PaddedTensor padded_inp = padding(std::move(input), pad_w, pad_h);
    uint64_t out_h = (padded_inp.height - kernel.height) / step + 1;
    uint64_t out_w = (padded_inp.width - kernel.width) / step + 1;

    auto output = Tensor(n_filters, in_h / step, in_w / step);

    auto kh = kernel.height;
    auto kw = kernel.width;
    auto img_stride = padded_inp.stride;
    auto ker_stride = kernel.stride;

    for (size_t k = 0; k < n_filters; ++k)
        for (size_t i = 0; i < out_h; ++i)
            for (size_t j = 0; j < out_w; ++j){

                float sum = 0.0f;
                size_t ii = i * step; // projected i
                size_t jj = j * step; // projected j

                for (size_t c = 0; c < in_channels; ++c) {
                    size_t filter_idx = (k * in_channels) + c; // the filter number for kernel

                    // pointers to the start of the 2d plane:
                    const float* ker_plane_start = &kernel.at(filter_idx, 0, 0);
                    const float* img_plane_start = &padded_inp.at(c, 0, 0);

                    for (size_t u = 0; u < kh; ++u) {
                        // pointer arithmetics faster than at()
                        const float* img_row = img_plane_start + (ii + u) * (img_stride) + jj;
                        const float* ker_row = ker_plane_start + u * ker_stride;

                        for (size_t v = 0; v < kw; ++v) {
                            sum += img_row[v] * ker_row[v];
                        }
                    }
                }
                output.at(k, i , j) = sum;
            }
    return output;
}
