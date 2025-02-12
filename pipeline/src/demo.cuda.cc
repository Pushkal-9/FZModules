#ifndef FZMOD_DRIVER
#define FZMOD_DRIVER

// regular includes
#include <cuda_runtime.h>
#include <iostream>

// include all modules
#include "fzg_class.hh"
#include "hfclass.hh"
#include "hist.hh"
#include "lorenzo.hh"
#include "spline.hh"

// include type
#include "type.h"
#include "context.h"
#include "utils/io.hh"
#include "fzmod.h"

namespace utils = _portable::utils;

void fzmod_compress_demo(std::string fname, fzmod_settings settings) {
    float *d_uncomp_ptr, *h_uncomp_ptr;
    float *d_decomp_ptr, *h_decomp_ptr;
    size_t data_len = settings.x * settings.y * settings.z;
    size_t compressed_len{0};
    size_t original_bytes = data_len * sizeof(float);
    uint8_t *compressed_data;

    // allocate device memory, copy data to device, and create stream (cuda stuff)
    cudaMalloc(&d_uncomp_ptr, original_bytes), cudaMallocHost(&h_uncomp_ptr, original_bytes);
    cudaMalloc(&d_decomp_ptr, original_bytes), cudaMallocHost(&h_decomp_ptr, original_bytes);

    // put file data into host memory
    utils::fromfile(fname, h_uncomp_ptr, data_len);

    cudaMemcpy(d_uncomp_ptr, h_uncomp_ptr, original_bytes, cudaMemcpyHostToDevice);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    settings.stream = stream;

    // compress and decompress
    fzmod_compress<float>(settings, h_uncomp_ptr, d_decomp_ptr);

    // clean up
    cudaFree(compressed_data);
    cudaFree(d_uncomp_ptr), cudaFreeHost(h_uncomp_ptr);
    cudaFree(d_decomp_ptr), cudaFreeHost(h_decomp_ptr);
    cudaStreamDestroy(stream);
}

// int fz_module_decompress(std::string fname, fzmod_len3 const len3) {
//   return 0;
// }

int main(int argc, char** argv) {

    auto fname = std::string(argv[1]);

    // get dimensions of the data as len1xlen2xlen3
    fzmod_len3 len3 = {0, 0, 0};
    if (argc == 5) {
        len3.x = std::stoi(argv[2]);
        len3.y = std::stoi(argv[3]);
        len3.z = std::stoi(argv[4]);
    } else {
        std::cerr << "Usage: " << argv[0] << " <filename> <len1> <len2> <len3>" << std::endl;
        return 1;
    }

    // set settings
    fzmod_settings settings;
    settings.predictor = fzmod_predtype::Lorenzo;
    settings.histogram = fzmod_histogramtype::HistogramSparse;
    settings.codec = fzmod_codectype::Huffman;
    settings.mode = fzmod_mode::Abs;
    settings.eb = 1e-4;
    settings.x = len3.x;
    settings.y = len3.y;
    settings.z = len3.z;

    // compress the data
    fzmod_compress_demo(fname, settings);


    std::cout << "Hello, World!" << std::endl;
    return 0;
}

#endif // FZMOD_DRIVER