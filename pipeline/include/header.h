#ifndef FZMOD_HEADER_H
#define FZMOD_HEADER_H

#ifdef __cplusplus
extern "C" {
#endif

#include "type.h"

#define FZMODHEADER_HEADER 0
#define FZMODHEADER_ANCHOR 1
#define FZMODHEADER_ENCODED 2
#define FZMODHEADER_SPFMT 3
#define FZMODHEADER_END 4

typedef struct fzmod_header {

  union {
    struct {
      fzmod_dtype dtype;
      fzmod_predtype pred_type;
      fzmod_histogramtype hist_type;
      fzmod_codectype codec1_type;
      fzmod_codectype _future_codec2_type;

      // pipeline config
      fzmod_mode mode;
      double eb;
      uint16_t radius;

      // codec config (coarse-HF)
      int vle_sublen;
      int vle_pardeg;

      uint32_t entry[FZMODHEADER_END + 1];  // segment entries

      // runtime sizes
      uint32_t x, y, z, w;
      size_t splen;

      // internal loggin
      double user_input_eb;
      double logging_min, logging_max;
    };
    
    struct {
      uint8_t __[128];
    };
  };

} fzmod_header;

#ifdef __cplusplus
}
#endif

#endif