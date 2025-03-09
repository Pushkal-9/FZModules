#ifndef A54D2009_1D4F_4113_9E26_9695A3669224
#define A54D2009_1D4F_4113_9E26_9695A3669224

#include <cstdint>

namespace fz {

// template <typename T, typename M = uint32_t>
// void spv_gather(T* in, size_t const in_len, T* d_val, M* d_idx, int* nnz,
//                 float* milliseconds, void* stream);

// template <typename T, typename M = uint32_t>
// void spv_scatter(T* d_val, M* d_idx, int const nnz, T* decoded,
//                  float* milliseconds, void* stream);

// template <typename T, typename Criterion, typename M = uint32_t>
// void spv_gather_naive(T* in, size_t const in_len, int const radius, T* cval,
//                       M* cidx, int* cn, Criterion c, float* milliseconds,
//                       void* stream);

template <typename T, typename M = uint32_t>
void spv_scatter_naive(T* d_val, M* d_idx, int const nnz, T* decoded,
                       float* milliseconds, void* stream = nullptr);

}  // namespace psz

#endif /* A54D2009_1D4F_4113_9E26_9695A3669224 */
