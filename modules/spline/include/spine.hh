#include <array>
#include <cstddef>
#include <cstdint>

#include "type.h"

using stdlen3 = std::array<size_t, 3>;

namespace psz::module {

template <typename T, typename E, typename FP = T>
int GPU_predict_spline(T* in_data, stdlen3 const data_len3,       //
                       E* out_ectrl, stdlen3 const ectrl_len3,    //
                       T* out_anchor, stdlen3 const anchor_len3,  //
                       void* out_outlier,                         //
                       double eb, uint32_t radius, void* stream);

template <typename T, typename E, typename FP = T>
int GPU_reverse_predict_spline(E* in_ectrl, stdlen3 const ectrl_len3,    //
                               T* in_anchor, stdlen3 const anchor_len3,  //
                               T* out_xdata, stdlen3 const xdata_len3,   //
                               double eb, uint32_t radius, void* stream);

};  // namespace psz::module
