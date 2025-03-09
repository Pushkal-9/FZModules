#include "spv.hh"
#include "err.hh"
#include "timer.hh"

namespace fz {

template <typename T, typename Criterion, typename M = uint32_t>
__global__ void KERNEL_CUHIP_spvn_gather(
    T* in, size_t const in_len, int const radius, T* cval, M* cidx, int* cn,
    Criterion criteria)
{
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < in_len) {
    auto d = in[tid];
    auto quantizable = criteria(d, radius);

    if (not quantizable) {
      auto cur_idx = atomicAdd(cn, 1);
      cidx[cur_idx] = tid;
      cval[cur_idx] = d;
      in[tid] = 0;
    }
  }
}

template <typename T, typename M = uint32_t>
__global__ void KERNEL_CUHIP_spvn_scatter(
    T* val, M* idx, int const nnz, T* out)
{
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < nnz) {
    int dst_idx = idx[tid];
    out[dst_idx] = val[tid];
  }
}

}  // namespace psz

// TODO only 1 CUHIP
// #define SPVN_GATHER(T, Criterion, M)                                            \
//   template <>                                                                   \
//   void fz::spv_gather_naive<T, Criterion, M>(                            \
//       T * in, size_t const in_len, int const radius, T* cval, M* cidx, int* cn, \
//       Criterion criteria, float* milliseconds, void* stream)                       \
//   {                                                                             \
//     auto grid_dim = (in_len - 1) / 128 + 1;                                     \
//     CREATE_GPUEVENT_PAIR;                                                       \
//     START_GPUEVENT_RECORDING(stream);                                           \
//     fz::KERNEL_CUHIP_spvn_gather<<<grid_dim, 128, 0, (cudaStream_t)stream>>>(  \
//         in, in_len, radius, cval, cidx, cn, criteria);                          \
//     STOP_GPUEVENT_RECORDING(stream);                                            \
//     CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));                     \
//     TIME_ELAPSED_GPUEVENT(milliseconds);                                        \
//     DESTROY_GPUEVENT_PAIR;                                                      \
//   }

#define SPVN_SCATTER(T, M)                                                     \
  template <>                                                                  \
  void fz::spv_scatter_naive<T, M>(                                     \
      T * val, M * idx, int const nnz, T* out, float* milliseconds, void* stream) \
  {                                                                            \
    auto grid_dim = (nnz - 1) / 128 + 1;                                       \
    CREATE_GPUEVENT_PAIR;                                                      \
    START_GPUEVENT_RECORDING(stream);                                          \
    fz::KERNEL_CUHIP_spvn_scatter<T, M>                                       \
        <<<grid_dim, 128, 0, (cudaStream_t)stream>>>(val, idx, nnz, out);      \
    STOP_GPUEVENT_RECORDING(stream);                                           \
    CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));                    \
    TIME_ELAPSED_GPUEVENT(milliseconds);                                       \
    DESTROY_GPUEVENT_PAIR;                                                     \
  }

SPVN_SCATTER(float, uint32_t)
SPVN_SCATTER(double, uint32_t)

// SPVN_GATHER(float, fz::criterion::gpu::eq<float>, uint32_t)
// SPVN_GATHER(float, fz::criterion::gpu::in_ball<float>, uint32_t)
// SPVN_GATHER(float, fz::criterion::gpu::in_ball_shifted<float>, uint32_t)
// SPVN_GATHER(double, fz::criterion::gpu::eq<double>, uint32_t)
// SPVN_GATHER(double, fz::criterion::gpu::in_ball<double>, uint32_t)
// SPVN_GATHER(double, fz::criterion::gpu::in_ball_shifted<double>, uint32_t)

#undef SPVN_SCATTER