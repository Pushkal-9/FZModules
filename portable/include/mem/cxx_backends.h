#ifndef _PORTABLE_MEM_CXX_BACKENDS_H
#define _PORTABLE_MEM_CXX_BACKENDS_H

#define ALIGN_128(len) (((len) + 127) & ~127)
#define ALIGN_256(len) (((len) + 255) & ~255)
#define ALIGN_512(len) (((len) + 511) & ~511)
#define ALIGN_1Ki(len) (((len) + 1023) & ~1023)
#define ALIGN_2Ki(len) (((len) + 2047) & ~2047)
#define ALIGN_4Ki(len) (((len) + 4095) & ~4095)
#define ALIGN_8Ki(len) (((len) + 8191) & ~8191)

#include <cstring>
#include <stdexcept>

#include <cuda_runtime.h>

#include "c_type.h"
#include "cxx_mem_ops.h"
#include "cxx_smart_ptr.h"

#define MAKE_STDLEN3(X, Y, Z) \
  std::array<size_t, 3> { X, Y, Z }

#define XYZ_TO_DIM3(LEN3) dim3(X(LEN3), Y(LEN3), Z(LEN3))
#define STDLEN3_TO_DIM3(LEN3) dim3(LEN3[0], LEN3[1], LEN3[2])
#define STDLEN3_TO_STRIDE3(LEN3) dim3(1, LEN3[0], LEN3[0] * LEN3[1])

#define GPULEN3 dim3
#define MAKE_GPULEN3(X, Y, Z) dim3(X, Y, Z)
#define GPU_BACKEND_SPECIFIC_STREAM cudaStream_t
#define GPU_EVENT cudaEvent_t
#define GPU_EVENT_CREATE(e) cudaEventCreate(e);

#define event_create_pair(...)                   \
  ([]() -> std::pair<cudaEvent_t, cudaEvent_t> { \
    cudaEvent_t a, b;                            \
    cudaEventCreate(&a);                         \
    cudaEventCreate(&b);                         \
    return {a, b};                               \
  })(__VA_ARGS__);
#define event_destroy_pair(a, b) \
  cudaEventDestroy(a);           \
  cudaEventDestroy(b);
#define event_recording_start(E1, STREAM) cudaEventRecord(E1, (cudaStream_t)STREAM);
#define event_recording_stop(E2, STREAM)     \
  cudaEventRecord(E2, (cudaStream_t)STREAM); \
  cudaEventSynchronize(E2);
#define event_time_elapsed(start, end, p_millisec) cudaEventElapsedTime(p_millisec, start, end);

#define create_stream(...)     \
  ([]() -> cudaStream_t {      \
    cudaStream_t stream;       \
    cudaStreamCreate(&stream); \
    return stream;             \
  })(__VA_ARGS__);
#define destroy_stream(stream) ([](void* s) { cudaStreamDestroy((cudaStream_t)s); })(stream);

#define sync_by_stream(stream) cudaStreamSynchronize((cudaStream_t)stream);
#define sync_device cudaDeviceSynchronize();


#endif /* _PORTABLE_MEM_CXX_BACKENDS_H */
