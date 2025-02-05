#ifndef FZ_MODULE_HIST_HH
#define FZ_MODULE_HIST_HH

#include "type.h"

namespace fz::module {

template <typename E>
int SEQ_histogram_generic(E* in_data, size_t const data_len, uint32_t* out_hist,
                          uint16_t const hist_len, float* milliseconds);

template <typename E>
int GPU_histogram_generic(E* in_data, size_t const data_len, uint32_t* out_hist,
                          uint16_t const hist_len, float* milliseconds,
                          void* stream);

template <typename E>
int SEQ_histogram_Cauchy_v2(E* in_data, size_t const data_len,
                            uint32_t* out_hist, uint16_t const hist_len,
                            float* milliseconds);

template <typename E>
int GPU_histogram_Cauchy(E* in_data, size_t const data_len, uint32_t* out_hist,
                         uint16_t const hist_len, float* milliseconds,
                         void* stream);

}  // namespace fz::module

#endif /* FZ_MODULE_HIST_HH */
