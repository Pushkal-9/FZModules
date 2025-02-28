#ifndef UTILS_TIMER_HH
#define UTILS_TIMER_HH

// Jiannan Tian
// (created) 2019-08-26
// (update) 2021-01-05, 2021-12-23, 2022-10-31, 2024-12-22

#include <chrono>
#include <utility>

using hires = std::chrono::high_resolution_clock;
using duration_t = std::chrono::duration<double>;
using hires_clock_t = std::chrono::time_point<hires>;

#define CREATE_CPU_TIMER                                    \
  std::chrono::time_point<std::chrono::steady_clock> a_ct1; \
  std::chrono::time_point<std::chrono::steady_clock> b_ct1;
#define START_CPU_TIMER a_ct1 = std::chrono::steady_clock::now();
#define STOP_CPU_TIMER b_ct1 = std::chrono::steady_clock::now();
#define TIME_ELAPSED_CPU_TIMER(PTR_MILLISEC) \
  ms = std::chrono::duration<float, std::milli>(b_ct1 - a_ct1).count();


#define CREATE_GPUEVENT_PAIR \
  cudaEvent_t a, b;          \
  cudaEventCreate(&a);       \
  cudaEventCreate(&b);
#define DESTROY_GPUEVENT_PAIR \
  cudaEventDestroy(a);        \
  cudaEventDestroy(b);
#define START_GPUEVENT_RECORDING(STREAM) \
  cudaEventRecord(a, (cudaStream_t)STREAM);
#define STOP_GPUEVENT_RECORDING(STREAM)     \
  cudaEventRecord(b, (cudaStream_t)STREAM); \
  cudaEventSynchronize(b);
#define TIME_ELAPSED_GPUEVENT(PTR_MILLISEC) \
  cudaEventElapsedTime(PTR_MILLISEC, a, b);

#endif /* UTILS_TIMER_HH */
