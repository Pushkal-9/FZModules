/**
 * @file hist.seq.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-07-26
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#include "hist.hh"
#include "timer.hh"

namespace fz::module {

template <typename E>
int SEQ_histogram_generic(E* in, size_t const inlen, 
                          uint32_t* out_hist, float* milliseconds) {
  auto t1 = hires::now();
  for (size_t i = 0; i < inlen; i++) {
    auto n = in[i];
    out_hist[(int)n] += 1;
  }
  auto t2 = hires::now();
  *milliseconds = static_cast<duration_t>(t2 - t1).count() * 1000;

  return 0;
}

}  // namespace fz::module

#define INIT_HIST_SEQ(E)                                                    \
  template int fz::module::SEQ_histogram_generic(                          \
      E* in, size_t const inlen, uint32_t* out_hist, float* milliseconds);

INIT_HIST_SEQ(uint8_t);
INIT_HIST_SEQ(uint16_t);
INIT_HIST_SEQ(uint32_t);
INIT_HIST_SEQ(float);

#undef INIT_HIST_SEQ
