#pragma once

#include <cuda_runtime.h>
#include <sys/time.h>
#include <cuda/experimental/stf.cuh> // cudastf
#include <fstream>
#include <iostream>
#include <vector>

#include "fzmod_api.hh" // FZMod API for helper methods
#include "ibuffer.hh" // internal buffer for STF
#include "hist_generic_stf.cu" // histogram generic kernel
#include "hist_sparse_stf.cu" // histogram sparse kernel
#include "huffman_class.hh" // HuffmanCodecSTF class
#include "lorenzo_1d.cu" // lorenzo_1d kernel
#include "spvn_stf.cu" // decomp outlier_scatter

using namespace util = _portable::utils; // portable utils namespace
using namespace cuda::experimental::stf; // STF namespace

