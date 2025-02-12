/**
 * @file type.h
 * @author Jiannan Tian
 * @brief C-complient type definitions; no methods in this header.
 * @version 0.3
 * @date 2022-04-29
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef FZMOD_TYPE_H
#define FZMOD_TYPE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "c_type.h"

// typedef _portable_device fzmod_device;
// typedef _portable_runtime fzmod_runtime;
// typedef _portable_runtime fzmod_backend;
// typedef _portable_toolkit fzmod_toolkit;

// typedef _portable_stream_t fzmod_stream_t;
// typedef _portable_mem_control fzmod_mem_control;
typedef _portable_dtype fzmod_dtype;
typedef _portable_len3 fzmod_len3;
// typedef _portable_size3 fzmod_size3;
// typedef _portable_data_summary fzmod_data_summary;

typedef enum {
  FZMOD_SUCCESS,
  FZMOD_GENERAL_GPU_FAILURE,
  FZMOD_FAIL_ONDISK_FILE_ERROR,
  FZMOD_FAIL_DATA_NOT_READY,
  FZMOD_TYPE_UNSUPPORTED,
  FZMOD_ERROR_GPU_GENERAL,
  FZMOD_ERROR_OUTLIER_OVERFLOW,
  FZMOD_ERROR_IO,
  // specify error when calling CUDA API
  FZMOD_FAIL_GPU_MALLOC,
  FZMOD_FAIL_GPU_MEMCPY,
  FZMOD_FAIL_GPU_ILLEGAL_ACCESS,
  // specify error related to our own memory manager
  FZMOD_FAIL_GPU_OUT_OF_MEMORY,
  // when compression is useless
  FZMOD_FAIL_INCOMPRESSIABLE,
  // TODO component related error
  FZMOD_FAIL_UNSUPPORTED_DATATYPE,
  FZMOD_FAIL_UNSUPPORTED_QUANTTYPE,
  FZMOD_FAIL_UNSUPPORTED_PRECISION,
  FZMOD_FAIL_UNSUPPORTED_PIPELINE,
  // not-implemented error
  FZMOD_NOT_IMPLEMENTED,
  // too many outliers
  FZMOD_OUTLIER_TOO_MANY,
  // specified wrong timer
  FZMOD_WRONG_TIMER_SPECIFIED,
} fzmod_error_status;
typedef fzmod_error_status fzmoderror;

typedef uint8_t byte_t;
typedef size_t szt;

// typedef enum {
//   FZgpu,
//   Histogram,
//   Huffman,
//   Lorenzo,
//   Spline,
// } fzmod_module;

typedef enum { Abs, Rel } fzmod_mode;
typedef enum { Lorenzo, LorenzoZigZag, LorenzoProto, Spline } fzmod_predtype;

typedef enum {
  Huffman,
  FZGPUCodec,
} fzmod_codectype;

typedef enum {
  HistogramGeneric,
  HistogramSparse,
  HistogramNull,
} fzmod_histogramtype;

// typedef enum {
//   FP64toFP32,
//   LogTransform,
//   ShiftedLogTransform,
//   Binning2x2,
//   Binning2x1,
//   Binning1x2,
// } fzmod_preprocestype;

// typedef enum {
//   STAGE_PREDICT = 0,
//   STAGE_HISTOGRM = 1,
//   STAGE_BOOK = 3,
//   STAGE_HUFFMAN = 4,
//   STAGE_OUTLIER = 5,
//   STAGE_END = 6
// } fzmod_time_stage;

struct fzmod_context;
typedef struct fzmod_settings fzmod_settings;

struct fzmod_header;
typedef struct fzmod_header fzmod_header;

// typedef struct fzmod_compressor {
//   void* compressor;
//   fzmod_ctx* ctx;
//   fzmod_header* header;
//   fzmod_dtype type;
//   fzmod_error_status last_error;
//   float stage_time[STAGE_END];
// } fzmod_compressor;

// // nested struct object (rather than ptr) results in Swig creating a `__get`,
// // which can be breaking. Used `prefix_` instead.
// typedef struct fzmod_statistics {
//   fzmod_data_summary odata, xdata;
//   f8 score_PSNR, score_MSE, score_NRMSE, score_coeff;
//   f8 max_err_abs, max_err_rel, max_err_pwrrel;
//   size_t max_err_idx;
//   f8 autocor_lag_one, autocor_lag_two;
//   f8 user_eb;
//   size_t len;
// } fzmod_statistics;

// typedef struct fzmod_capi_array {
//   void* const buf;
//   fzmod_len3 const len3;
//   fzmod_dtype dtype;
// } fzmod_carray;

// typedef fzmod_carray fzmod_data_input;
// typedef fzmod_carray fzmod_input;
// typedef fzmod_carray fzmod_in;
// typedef fzmod_carray* fzmodarray_mutable;

// typedef struct fzmod_rettype_archive {
//   u1* compressed;
//   size_t* comp_bytes;
//   fzmod_header* header;
// } fzmod_archive;

// typedef fzmod_archive fzmod_data_output;
// typedef fzmod_archive fzmod_output;
// typedef fzmod_archive fzmod_out;

// /**
//  * @brief This is an archive description of compaction array rather than
//  * runtime one, which deals with host-device residency status.
//  *
//  */
// typedef struct fzmod_capi_compact {
//   void* const val;
//   uint32_t* idx;
//   uint32_t* num;
//   uint32_t reserved_len;
//   fzmod_dtype const dtype;
// } fzmod_capi_compact;

// typedef fzmod_capi_compact fzmod_capi_outlier;
// typedef fzmod_capi_compact fzmod_compact;
// typedef fzmod_capi_compact fzmod_outlier;
// typedef fzmod_outlier* fzmod_outlier_mutable;

// typedef struct fzmod_runtime_config {
//   double eb;
//   int radius;
// } fzmod_runtime_config;
// typedef fzmod_runtime_config fzmod_rc;

// // forward
// struct fzmod_profiling;

// typedef enum fzmod_timing_mode {
//   SYNC_BY_STREAM,
//   CPU_BARRIER,
//   GPU_AUTOMONY
// } fzmod_timing_mode;

#ifdef __cplusplus
}
#endif

#endif
