#include <cuda/experimental/stf.cuh>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>

#include "fzmod_api.hh"

#include "proto_lorenzo_1d.cu"
#include "hist_generic.cu"
#include "huffman_class.hh"

//? can I freeze data after prediction (outliers/quant codes) and then call another task?

namespace utils = _portable::utils;
using namespace cuda::experimental::stf;

int main(int argc, char **argv) {

  // ~~~~~~~~~~~~~~~~~~~~~~~ check args ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

  if (argc != 5) {
      std::cerr << "Usage: " << argv[0] << " <filename> <len1> <len2> <len3> <eb>" << std::endl;
      return 1;
  }
  auto fname = std::string(argv[1]);
  size_t len1 = std::stoi(argv[2]);
  size_t len2 = std::stoi(argv[3]);
  size_t len3 = std::stoi(argv[4]);

  printf("fname: %s, len1: %zu, len2: %zu, len3: %zu\n", fname.c_str(), len1, len2, len3);

  // ~~~~~~~~~~~~~~~~~~~~~~~ Setup ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

  size_t data_len = len1 * len2 * len3;
  double eb = 2e-4;

  int sublen, pardeg;
  uint8_t* codec_comp_output{nullptr};
  size_t codec_comp_output_len{0};

  // get dataset
  float* input_data_host;
  cudaMallocHost(&input_data_host, data_len * sizeof(float));
  utils::fromfile(fname, input_data_host, data_len);

  // make the cudastream
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // ~~~~~~~~~~~~~~~~~~~~~~~ kernel optimizations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

  // Prediction kernel optimization

  auto divide_3 = [](dim3 len, dim3 sublen) {
    return dim3((len.x - 1) / sublen.x + 1, (len.y - 1) / sublen.y + 1,
                (len.z - 1) / sublen.z + 1);
  };

  constexpr auto Tile1D = dim3(256, 1, 1);
  constexpr auto Block1D = dim3(256, 1, 1);
  auto Grid1D = divide_3(dim3(data_len, 1, 1), Tile1D);

  // Histogram kernel optimization

  uint32_t out_num[1];
  int hist_grid_d, hist_block_d, hist_shmem_use, hist_repeat;
  histogram_optimizer(data_len, 1024, hist_grid_d, hist_block_d, hist_shmem_use, hist_repeat);

  // huffman kernel optimizer
  capi_phf_coarse_tune(data_len, &sublen, &pardeg);

  // ~~~~~~~~~~~~~~~~~~~~~~~ CUDASTF Setup ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  context ctx;

  // logical uncompressed data
  auto l_u = ctx.logical_data(input_data_host, {data_len});

  auto quant_codes = ctx.logical_data(shape_of<slice<uint16_t>>(data_len));

  auto o_vals = ctx.logical_data(shape_of<slice<float>>(data_len));
  auto o_idxs = ctx.logical_data(shape_of<slice<uint32_t>>(data_len));
  auto o_num = ctx.logical_data(out_num);

  auto l_hist = ctx.logical_data(shape_of<slice<uint32_t>>(1024));

  HuffmanCodecSTF codec_hf(data_len, pardeg, ctx);

  // ~~~~~~~~~~~~~~~~~~~~~~~ Tasks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

  //! 1D Prototype Lorenzo Kernel
  ctx.task(l_u.rw(), quant_codes.write(), o_vals.write(), o_idxs.write(), o_num.write())->*[&](
    cudaStream_t s, 
    slice<float> l_u,
    slice<uint16_t> q_c,
    slice<float> o_v,
    slice<uint32_t> o_i,
    slice<uint32_t> o_n) {
      kernel_lorenzo_1d<<<Grid1D, Block1D, 0, s>>>(l_u, data_len, q_c, o_v, o_i, o_n, eb*2, 1/(eb*2)); 
  };

  //! Generic 2013 Histogram Kernel
  ctx.task(quant_codes.read(), l_hist.write())->*[&](cudaStream_t s, auto q_c, auto l_h) {
      kernel_hist_generic<<<hist_grid_d, hist_block_d, hist_shmem_use, s>>>(q_c, data_len, l_h, 1024, hist_repeat);
  };

  //! Huffman Kernel CPU Buildbook
  codec_hf.buildbook(l_hist, 1024, ctx);

  //! Huffman Encode GPU Kernel
  // ctx.task(quant_codes.rw())->*[&](cudaStream_t s, auto q_c) {
    
  // };

  //! Finalize Compression
  // ctx.host_launch()->*[&]() {
    
  // };

  //! Launch the tasks
  ctx.finalize();

  // ~~~~~~~~~~~~~~~~~~~~~~~ cleanup / report ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

  printf("out_num = %u\n", out_num[0]);

  // free memory
  cudaFreeHost(input_data_host);
  
  return 0;
}