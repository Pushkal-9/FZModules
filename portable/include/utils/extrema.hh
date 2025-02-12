#ifndef CE05A256_23CB_4243_8839_B1FDA9C540D2
#define CE05A256_23CB_4243_8839_B1FDA9C540D2

#include <stdint.h>
#include <stdlib.h>

#include "busyheader.hh"
#include "type.h"

namespace _portable::utils {

template <typename T> void GPU_extrema(T* d_ptr, size_t len, T res[4]);

template <typename T1, typename T2 = T1>
void GPU_probe_extrema(T1* in, size_t len, T2& max_value, T2& min_value, T2& range) {
  T1 result[4];
  GPU_extrema(in, len, result);
  min_value = result[0];
  max_value = result[1];
  range = max_value - min_value;
}


} // namespace _portable::utils

#endif /* CE05A256_23CB_4243_8839_B1FDA9C540D2 */