#include "fzmod.h"
#include "type.h"
#include "context.h"
#include "utils/extrema.hh"



template <typename T>
fzmoderror fzmod_compress(fzmod_settings settings, T* input_data, T* output_data) {

  // setup header from user settings
  fzmod_header header;
  header.dtype = F4; // todo: allow future support for other types
  header.pred_type = settings.predictor;
  header.hist_type = settings.histogram;
  header.codec1_type = settings.codec;
  header.mode = settings.mode;
  header.eb = settings.eb;
  header.radius = settings.radius;
  header.x = settings.x;
  header.y = settings.y;
  header.z = settings.z;
  header.w = 1;

  settings.data_len = settings.x * settings.y * settings.z;




  return FZMOD_SUCCESS;
}

template fzmoderror fzmod_compress<float>(fzmod_settings settings, float* input_data, float* output_data);