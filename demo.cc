#ifndef FZMOD_DRIVER
#define FZMOD_DRIVER

#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>

#include "fzmod_api.hh"
namespace utils = _portable::utils;



int main(int argc, char **argv) {

    // ensure 5 args
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <filename> <len1> <len2> <len3>" << std::endl;
        return 1;
    }

    // get filename
    auto fname = std::string(argv[1]);

    // get dimensions of the data as len1xlen2xlen3
    size_t len1 = std::stoi(argv[2]);
    size_t len2 = std::stoi(argv[3]);
    size_t len3 = std::stoi(argv[4]);
    std::vector<size_t> dims({len1, len2, len3});

    fz::Config conf(len1, len2, len3);
    conf.eb = 1e-4;
    conf.eb_type = fz::EB_TYPE::EB_ABS;
    conf.algo = fz::ALGO::ALGO_LORENZO;
    conf.precision = fz::PRECISION::PRECISION_FLOAT;
    conf.codec = fz::CODEC::CODEC_HUFFMAN;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

    float* input_data_d, * input_data_h;
    uint8_t* compressed;

    cudaMallocHost(&input_data_h, conf.orig_size);
    cudaMalloc(&input_data_d, conf.orig_size);
    utils::fromfile(fname, input_data_h, conf.orig_size);
    cudaMemcpy(input_data_d, input_data_h, conf.orig_size, cudaMemcpyHostToDevice);

    fz::Compressor<float> compressor(conf);

    std::cout << "Compressing...\n" << std::endl;

    compressor.compress(input_data_d, &compressed, stream);

    
    // add decompression

    std::cout << "compressed!\n" << std::endl;
    std::cout << "original size: " << conf.orig_size << std::endl;
    std::cout << "compressed size: " << conf.comp_size << std::endl;
    std::cout << "compression ratio: " << (float)conf.orig_size / (float)conf.comp_size << std::endl;

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
    cudaFreeHost(input_data_h);
    cudaFree(input_data_d);
    cudaStreamDestroy(stream);

    return 0;
}

#endif /* FZMOD_DRIVER */