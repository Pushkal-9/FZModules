#ifndef FZMOD_H
#define FZMOD_H

// #ifdef __cplusplus
// extern "C" {
// #endif

#include "context.h"
#include "type.h"
#include "header.h"

namespace fz {

template <typename DType>
class Compressor {
  private:

};

}

template <typename T>
fzmoderror fzmod_compress(fzmod_settings settings, T* input_data, T* output_data);

// #ifdef __cplusplus
// }
// #endif

#endif