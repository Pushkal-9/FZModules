/**
 * @file l23.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-11-01
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef FZ_MODULE_LRZ_GPU_HH
#define FZ_MODULE_LRZ_GPU_HH

#include <array>
#include <cstdint>

#include "type.h"

#define PROPER_EB f8

using stdlen3 = std::array<size_t, 3>;

namespace fz::module {

template <typename T, bool UseZigZag, typename Eq>
pszerror GPU_c_lorenzo_nd_with_outlier(T* const in_data,
                                       stdlen3 const data_len3,
                                       Eq* const out_eq, void* out_outlier,
                                       PROPER_EB const eb,
                                       uint16_t const radius, void* stream);

template <typename T, bool UseZigZag, typename Eq>
pszerror GPU_x_lorenzo_nd(Eq* const in_eq, T* const in_outlier,
                          T* const out_data, stdlen3 const data_len3,
                          PROPER_EB const eb, uint16_t const radius,
                          void* stream);

template <typename TIN, typename TOUT, bool ReverseProcess>
pszerror GPU_lorenzo_prequant(TIN* const in, size_t const len,
                              PROPER_EB const eb, TOUT* const out,
                              float* time_elapsed, void* _stream);

}  // namespace fz::module

#endif /* FZ_MODULE_LRZ_GPU_HH */