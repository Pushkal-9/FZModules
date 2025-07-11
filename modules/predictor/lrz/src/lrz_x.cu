/**
 * @file l23_x.cuhip.inl
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2022-12-22
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */


#include "lorenzo.hh"
#include "err.hh"
#include "timer.hh"
// #include "subr.cu_hip.inl"
// #include "mem/cxx_backends.h"
#include "detail/wave32.cu"
#include "detail/zigzag.hh"

#define SETUP_ZIGZAG                    \
  using ZigZag = fz::ZigZag<Eq>;       \
  using EqUInt = typename ZigZag::UInt; \
  using EqSInt = typename ZigZag::SInt;

#define Z(LEN3) LEN3[2]
#define Y(LEN3) LEN3[1]
#define X(LEN3) LEN3[0]
#define TO_DIM3(LEN3) dim3(X(LEN3), Y(LEN3), Z(LEN3))

namespace fz {

template <
    typename T, int TileDim, int Seq, bool UseZigZag, typename Eq = uint16_t, typename Fp = T>
__global__ void KERNEL_CUHIP_x_lorenzo_1d1l(
    Eq* const in_eq, T* const in_outlier, T* const out_data, dim3 const data_len3,
    dim3 const data_leap3, uint16_t const radius, Fp const ebx2)
{
  SETUP_ZIGZAG
  constexpr auto NTHREAD = TileDim / Seq;  // equiv. to blockDim.x

  __shared__ T scratch[TileDim];  // for data and in_outlier
  __shared__ Eq s_eq[TileDim];
  __shared__ T exch_in[NTHREAD / 32];
  __shared__ T exch_out[NTHREAD / 32];

  T thp_data[Seq];

  auto id_base = blockIdx.x * TileDim;

  auto load_fuse_1d = [&]() {
#pragma unroll
    for (auto i = 0; i < Seq; i++) {
      auto local_id = threadIdx.x + i * NTHREAD;
      auto id = id_base + local_id;
      if (id < data_len3.x) {
        // fuse outlier and error-quant
        if constexpr (not UseZigZag) {
          scratch[local_id] = in_outlier[id] + static_cast<T>(in_eq[id]) - radius;
        }
        else {
          auto e = in_eq[id];
          scratch[local_id] =
              in_outlier[id] + static_cast<T>(ZigZag::decode(static_cast<EqUInt>(e)));
        }
      }
    }
    __syncthreads();

#pragma unroll
    for (auto i = 0; i < Seq; i++) thp_data[i] = scratch[threadIdx.x * Seq + i];
    __syncthreads();
  };

  auto block_scan_1d = [&]() {
    fz::SUBR_CUHIP_WAVE32_intrawarp_inclscan_1d<T, Seq>(thp_data);
    fz::SUBR_CUHIP_WAVE32_intrablock_exclscan_1d<T, Seq, NTHREAD>(thp_data, exch_in, exch_out);

    // put back to shmem
#pragma unroll
    for (auto i = 0; i < Seq; i++) scratch[threadIdx.x * Seq + i] = thp_data[i] * ebx2;
    __syncthreads();
  };

  auto write_1d = [&]() {
#pragma unroll
    for (auto i = 0; i < Seq; i++) {
      auto local_id = threadIdx.x + i * NTHREAD;
      auto id = id_base + local_id;
      if (id < data_len3.x) out_data[id] = scratch[local_id];
    }
  };

  /*-----------*/

  load_fuse_1d();
  block_scan_1d();
  write_1d();
}

//   2D partial sum: memory layout
//
//       ------> gix (x)
//
//   |   t(0,0)       t(0,1)       t(0,2)       t(0,3)       ... t(0,f)
//   |
//   |   thp(0,0)[0]  thp(0,0)[0]  thp(0,0)[0]  thp(0,0)[0]
//  giy  thp(0,0)[1]  thp(0,0)[1]  thp(0,0)[1]  thp(0,0)[1]
//  (y)  |            |            |            |
//       thp(0,0)[7]  thp(0,0)[7]  thp(0,0)[7]  thp(0,0)[7]
//
//   |   t(1,0)       t(1,1)       t(1,2)       t(1,3)       ... t(1,f)
//   |
//   |   thp(1,0)[0]  thp(1,0)[0]  thp(1,0)[0]  thp(1,0)[0]
//  giy  thp(1,0)[1]  thp(1,0)[1]  thp(1,0)[1]  thp(1,0)[1]
//  (y)  |            |            |            |
//       thp(1,0)[7]  thp(1,0)[7]  thp(1,0)[7]  thp(1,0)[7]

template <typename T, bool UseZigZag, typename Eq = uint16_t, typename Fp = T>
__global__ [[deprecated]] void KERNEL_CUHIP_x_lorenzo_2d1l(  //
    Eq* const in_eq, T* const in_outlier, T* const out_data, dim3 const data_len3,
    dim3 const data_leap3, uint16_t const radius, Fp const ebx2)
{
  SETUP_ZIGZAG;
  constexpr auto TileDim = 16;
  constexpr auto YSEQ = TileDim / 2;  // sequentiality in y direction
  static_assert(TileDim == 16, "In one case, we need TileDim for 2D == 16");

  __shared__ T scratch[TileDim];  // TODO use warp shuffle to eliminate this
  T thp_data[YSEQ] = {0};

  auto gix = blockIdx.x * TileDim + threadIdx.x;
  auto giy_base = blockIdx.y * TileDim + threadIdx.y * YSEQ;  // BDY * YSEQ = TileDim == 16

  auto get_gid = [&](auto i) { return (giy_base + i) * data_leap3.y + gix; };

  auto load_fuse_2d = [&]() {

#pragma unroll
    for (auto i = 0; i < YSEQ; i++) {
      auto gid = get_gid(i);
      if (gix < data_len3.x and (giy_base + i) < data_len3.y) {
        // fuse outlier and error-quant
        if constexpr (not UseZigZag) {
          thp_data[i] = in_outlier[gid] + static_cast<T>(in_eq[gid]) - radius;
        }
        else {
          auto e = in_eq[gid];
          thp_data[i] = in_outlier[gid] + static_cast<T>(ZigZag::decode(static_cast<EqUInt>(e)));
        }
      }
    }
  };

  // partial-sum along y-axis, sequantially
  // then, in-warp partial-sum along x-axis
  auto block_scan_2d = [&]() {
    for (auto i = 1; i < YSEQ; i++) thp_data[i] += thp_data[i - 1];
    // two-pass: store for cross-thread-private update
    // TODO shuffle up by 16 in the same warp
    if (threadIdx.y == 0) scratch[threadIdx.x] = thp_data[YSEQ - 1];
    __syncthreads();
    // broadcast the partial-sum result from a previous segment
    if (threadIdx.y == 1) {
      auto tmp = scratch[threadIdx.x];
#pragma unroll
      for (auto i = 0; i < YSEQ; i++) thp_data[i] += tmp;  // regression as pointer
    }
    // implicit sync as there is half-warp divergence

#pragma unroll
    for (auto i = 0; i < YSEQ; i++) {
      for (auto d = 1; d < TileDim; d *= 2) {
        T n = __shfl_up_sync(0xffffffff, thp_data[i], d, 16);  // half-warp shuffle
        if (threadIdx.x >= d) thp_data[i] += n;
      }
      thp_data[i] *= ebx2;  // scale accordingly
    }
  };

  auto decomp_write_2d = [&]() {
#pragma unroll
    for (auto i = 0; i < YSEQ; i++) {
      auto gid = get_gid(i);
      if (gix < data_len3.x and (giy_base + i) < data_len3.y) out_data[gid] = thp_data[i];
    }
  };

  /*-----------*/

  load_fuse_2d();
  block_scan_2d();
  decomp_write_2d();
}

template <typename T, bool UseZigZag, typename Eq = uint16_t, typename Fp = T>
__global__ void KERNEL_CUHIP_x_lorenzo_2d1l__32x32(  //
    Eq* const in_eq, T* const in_outlier, T* const out_data, dim3 const data_len3,
    dim3 const data_leap3, uint16_t const radius, Fp const ebx2)
{
  SETUP_ZIGZAG;
  constexpr auto TileDim = 32;
  constexpr auto NumWarps = 4;
  constexpr auto YSEQ = TileDim / NumWarps;

  __shared__ T scratch[NumWarps - 1][TileDim + 1];
  T thp_data[YSEQ] = {0};

  auto gix = blockIdx.x * TileDim + threadIdx.x;
  auto giy_base = blockIdx.y * TileDim + threadIdx.y * YSEQ;
  auto get_gid = [&](auto i) { return (giy_base + i) * data_leap3.y + gix; };

  auto load_fuse_2d = [&]() {
#pragma unroll
    for (auto i = 0; i < YSEQ; i++) {
      auto gid = get_gid(i);
      if (gix < data_len3.x and (giy_base + i) < data_len3.y) {
        // fuse outlier and error-quant
        if constexpr (not UseZigZag) {
          thp_data[i] = in_outlier[gid] + static_cast<T>(in_eq[gid]) - radius;
        }
        else {
          auto e = in_eq[gid];
          thp_data[i] = in_outlier[gid] + static_cast<T>(ZigZag::decode(static_cast<EqUInt>(e)));
        }
      }
    }
  };

  auto block_scan_2d = [&]() {
    for (auto i = 1; i < YSEQ; i++) thp_data[i] += thp_data[i - 1];

    // 0, 1, 2
    if (threadIdx.y < NumWarps - 1) scratch[threadIdx.y][threadIdx.x] = thp_data[YSEQ - 1];
    __syncthreads();

    // cross-wrap scan

    if (threadIdx.y == 0) {
      T warp_accum[NumWarps - 1];  // 0, 1, 2
#pragma unroll
      for (auto i = 0; i < NumWarps - 1; i++) {  // load thp_data[YSEQ - 1] from each warp
        warp_accum[i] = scratch[i][threadIdx.x];
      }
#pragma unroll
      for (auto i = 1; i < NumWarps - 1; i++) {  // exclusive scan
        warp_accum[i] += warp_accum[i - 1];
      }
#pragma unroll
      for (auto i = 1; i < NumWarps - 1; i++) {  // determine the final addends
        scratch[i][threadIdx.x] = warp_accum[i];
      }
    }
    __syncthreads();

    if (threadIdx.y > 0) {
      auto addend = scratch[threadIdx.y - 1][threadIdx.x];
#pragma unroll
      for (auto i = 0; i < YSEQ; i++) thp_data[i] += addend;  // regression as pointer
    }
    __syncthreads();

#pragma unroll
    for (auto i = 0; i < YSEQ; i++) {
      for (auto d = 1; d < TileDim; d *= 2) {
        T n = __shfl_up_sync(0xffffffff, thp_data[i], d, 32);  // full-warp shuffle
        if (threadIdx.x >= d) thp_data[i] += n;
      }
      thp_data[i] *= ebx2;  // scale accordingly
    }
  };

  auto decomp_write_2d = [&]() {
#pragma unroll
    for (auto i = 0; i < YSEQ; i++) {
      auto gid = get_gid(i);
      if (gix < data_len3.x and (giy_base + i) < data_len3.y) out_data[gid] = thp_data[i];
    }
  };

  /*-----------*/

  load_fuse_2d();
  block_scan_2d();
  decomp_write_2d();
}

// 32x8x8 data block maps to 32x1x8 thread block
template <typename T, bool UseZigZag, typename Eq = uint16_t, typename Fp = T>
__global__ void KERNEL_CUHIP_x_lorenzo_3d1l(  //
    Eq* const in_eq, T* const in_outlier, T* const out_data, dim3 const data_len3,
    dim3 const data_leap3, uint16_t const radius, Fp const ebx2)
{
  SETUP_ZIGZAG
  constexpr auto TileDim = 8;
  constexpr auto YSEQ = TileDim;
  static_assert(TileDim == 8, "In one case, we need TileDim for 3D == 8");

  __shared__ T scratch[TileDim][4][8];
  T thread_private[YSEQ] = {0};

  auto seg_id = threadIdx.x / 8;
  auto seg_tix = threadIdx.x % 8;

  auto gix = blockIdx.x * (4 * TileDim) + threadIdx.x;
  auto giy_base = blockIdx.y * TileDim;
  auto giy = [&](auto y) { return giy_base + y; };
  auto giz = blockIdx.z * TileDim + threadIdx.z;
  auto gid = [&](auto y) { return giz * data_leap3.z + (giy_base + y) * data_leap3.y + gix; };

  auto load_fuse_3d = [&]() {
  // load to thread-private array (fuse at the same time)
#pragma unroll
    for (auto y = 0; y < YSEQ; y++) {
      if (gix < data_len3.x and giy_base + y < data_len3.y and giz < data_len3.z) {
        // fuse outlier and error-quant
        if constexpr (not UseZigZag) {
          thread_private[y] = in_outlier[gid(y)] + static_cast<T>(in_eq[gid(y)]) - radius;
        }
        else {
          auto e = in_eq[gid(y)];
          thread_private[y] =
              in_outlier[gid(y)] + static_cast<T>(ZigZag::decode(static_cast<EqUInt>(e)));
        }
      }
    }
  };

  auto block_scan_3d = [&]() {
    // partial-sum along y-axis, sequentially
    for (auto y = 1; y < YSEQ; y++) thread_private[y] += thread_private[y - 1];

#pragma unroll
    for (auto i = 0; i < TileDim; i++) {
      // ND partial-sums along x- and z-axis
      // in-warp shuffle used: in order to perform, it's transposed after
      // X-partial sum
      T val = thread_private[i];

      for (auto dist = 1; dist < TileDim; dist *= 2) {
        auto addend = __shfl_up_sync(0xffffffff, val, dist, 8);
        if (seg_tix >= dist) val += addend;
      }

      // x-z transpose
      scratch[threadIdx.z][seg_id][seg_tix] = val;
      __syncthreads();
      val = scratch[seg_tix][seg_id][threadIdx.z];
      __syncthreads();

      for (auto dist = 1; dist < TileDim; dist *= 2) {
        auto addend = __shfl_up_sync(0xffffffff, val, dist, 8);
        if (seg_tix >= dist) val += addend;
      }

      scratch[threadIdx.z][seg_id][seg_tix] = val;
      __syncthreads();
      val = scratch[seg_tix][seg_id][threadIdx.z];
      __syncthreads();

      thread_private[i] = val;
    }
  };

  auto decomp_write_3d = [&]() {
#pragma unroll
    for (auto y = 0; y < YSEQ; y++)
      if (gix < data_len3.x and giy(y) < data_len3.y and giz < data_len3.z)
        out_data[gid(y)] = thread_private[y] * ebx2;
  };

  ////////////////////////////////////////////////////////////////////////////
  load_fuse_3d();
  block_scan_3d();
  decomp_write_3d();
}

}  // namespace psz

namespace fz::module {

template <typename T, bool UseZigZag, typename Eq>
int GPU_x_lorenzo_nd(
    Eq* const in_eq, T* const in_outlier, T* const out_data, stdlen3 const _data_len3,
    double const ebx2, double const ebx2_r, uint16_t const radius, void* stream)
{
  using namespace fz::kernelconfig;

  // error bound
  auto data_len3 = TO_DIM3(_data_len3);
  auto data_leap3 = dim3(1, data_len3.x, data_len3.x * data_len3.y);
  auto d = lorenzo_utils::ndim(data_len3);

  if (d == 1)
    fz::KERNEL_CUHIP_x_lorenzo_1d1l<T, x_lorenzo<1>::tile.x, x_lorenzo<1>::sequentiality.x, UseZigZag, Eq>
        <<<x_lorenzo<1>::thread_grid(data_len3), x_lorenzo<1>::thread_block, 0, (cudaStream_t)stream>>>(
            in_eq, in_outlier, out_data, data_len3, data_leap3, radius, (T)ebx2);
  else if (d == 2) {
    // psz::KERNEL_CUHIP_x_lorenzo_2d1l<T, UseZigZag, Eq>
    //     <<<x_lorenzo<2>::thread_grid(data_len3), x_lorenzo<2>::thread_block, 0,
    //        (cudaStream_t)stream>>>(
    //         in_eq, in_outlier, out_data, data_len3, data_leap3, radius, (T)ebx2);
    fz::KERNEL_CUHIP_x_lorenzo_2d1l__32x32<T, UseZigZag, Eq>
        <<<x_lorenzo<2, 32>::thread_grid(data_len3), x_lorenzo<2, 32>::thread_block, 0,
           (cudaStream_t)stream>>>(
            in_eq, in_outlier, out_data, data_len3, data_leap3, radius, (T)ebx2);
  }
  else if (d == 3)
    fz::KERNEL_CUHIP_x_lorenzo_3d1l<T, UseZigZag, Eq>
        <<<x_lorenzo<3>::thread_grid(data_len3), x_lorenzo<3>::thread_block, 0,
           (cudaStream_t)stream>>>(
            in_eq, in_outlier, out_data, data_len3, data_leap3, radius, (T)ebx2);
  else
    return -1;

  return 0;
}

}  // namespace fz::module

#define INSTANTIATE_GPU_L23X_3params(T, USE_ZIGZAG, Eq)                                 \
  template int fz::module::GPU_x_lorenzo_nd<T, USE_ZIGZAG, Eq>(                        \
      Eq* const in_eq, T* const in_outlier, T* const out_data, stdlen3 const data_len3, \
      double const ebx2, double const ebx2_r, uint16_t const radius, void* stream);

#define INSTANTIATE_GPU_L23X_2params(T, Eq)   \
  INSTANTIATE_GPU_L23X_3params(T, false, Eq); \
  INSTANTIATE_GPU_L23X_3params(T, true, Eq);

#define INSTANTIATE_GPU_L23X_1param(T) \
  INSTANTIATE_GPU_L23X_2params(T, uint8_t); \
  INSTANTIATE_GPU_L23X_2params(T, uint16_t);

INSTANTIATE_GPU_L23X_1param(float);
INSTANTIATE_GPU_L23X_1param(double);

#undef INSTANTIATE_GPU_L23X_1param
#undef INSTANTIATE_GPU_L23X_2params
#undef INSTANTIATE_GPU_L23X_3params