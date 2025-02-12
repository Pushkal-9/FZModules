#ifndef FZMOD_CONTEXT_H
#define FZMOD_CONTEXT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include "type.h"

typedef struct fzmod_settings {
  fzmod_predtype predictor;
  fzmod_histogramtype histogram;
  fzmod_codectype codec;
  fzmod_mode mode;
  double eb;
  uint32_t x,y,z;

  // context related
  uint16_t radius = 512;
  uint16_t dict_size = 1024;
  size_t data_len;
  void* stream;
} fzmod_settings;

#ifdef __cplusplus
}
#endif

#endif