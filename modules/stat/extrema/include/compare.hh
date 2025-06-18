/**
 * @file compare.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-10-09
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef CE05A256_23CB_4243_8839_B1FDA9C540D2
#define CE05A256_23CB_4243_8839_B1FDA9C540D2

#include <stdint.h>
#include <stdlib.h>


namespace fz::module {
// bool GPU_identical(void* d1, void* d2, size_t sizeof_T, size_t const len, void* stream = nullptr);
template <typename T> void GPU_extrema(T* d_ptr, size_t len, T res[4]);
template <typename T> void GPU_calculate_errors(T* d_odata, T odata_avg, T* d_xdata, T xdata_avg, size_t len, T h_err[4]);
template <typename T> void GPU_find_max_error(T* a, T* b, size_t const len, T& maxval, size_t& maxloc, void* stream = nullptr);
}

namespace fz::analysis {

template <typename T1, typename T2 = T1>
void GPU_probe_extrema(T1* in, size_t len, T2& max_value, T2& min_value, T2& range)
{
  T1 result[4];
  module::GPU_extrema(in, len, result);
  min_value = result[0];
  max_value = result[1];
  range = max_value - min_value;
}


}  // namespace psz::analysis

#endif /* CE05A256_23CB_4243_8839_B1FDA9C540D2 */
