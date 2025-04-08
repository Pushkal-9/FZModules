#ifndef FZMOD_CONFIG_HH
#define FZMOD_CONFIG_HH

namespace fz {

  typedef enum EB_TYPE {EB_REL, EB_ABS} EB_TYPE;
  typedef enum ALGO {ALGO_LORENZO, ALGO_SPLINE} ALGO;
  typedef enum PRECISION {PRECISION_FLOAT, PRECISION_DOUBLE} PRECISION;
  typedef enum CODEC {CODEC_HUFFMAN, CODEC_FZG} CODEC;
  typedef enum SECONDARY_CODEC {NONE, GZIP, LSTD} SECONDARY_CODEC;

  class Config {
    public:

      bool toFile = true;
      std::string fname;

      bool report = true;
      bool compare = false;
      bool dump = false;
      bool comp = true;

      double eb = 1e-3;
      EB_TYPE eb_type = EB_ABS;
      ALGO algo = ALGO_LORENZO;
      PRECISION precision = PRECISION_FLOAT;
      CODEC codec = CODEC_HUFFMAN;
      SECONDARY_CODEC secondary_codec = NONE;

      // relative error bounds
      double logging_max, logging_min;
      
      uint32_t x;
      uint32_t y;
      uint32_t z;
      size_t len;
      size_t splen;

      size_t orig_size;
      size_t comp_size;

      // histogram
      uint16_t radius = 512;
      bool use_histogram_sparse = true;
      bool use_histogram_generic = false;

      // huffman
      int sublen;
      int pardeg;

      // lorenzo
      bool use_lorenzo_regular = true;
      bool use_lorenzo_zigzag = false;

      //spline
      size_t const anchor512_len;
      constexpr static size_t BLK = 8;

      static size_t _div(size_t _l, size_t _subl) { return (_l - 1) / _subl + 1; };

      Config(size_t _x, size_t _y = 0, size_t _z = 0) : x(_x), y(_y), z(_z),
        anchor512_len(_div(_x, BLK) * _div(_y, BLK) * _div(_z, BLK)) {
        len = x * y * z;
        orig_size = len * sizeof(float);
        comp_size = 0;
      }

      std::array<size_t, 3> anchor_len3() const {
        return std::array<size_t, 3>{_div(x, BLK), _div(y, BLK), _div(z, BLK)};
      }

  };

  struct fzmod_metrics {
    
    // timing data
    double end_to_end_comp_time = 0;
    double preprocessing_time = 0;
    double prediction_time = 0;
    double hist_time = 0;
    double encoder_time = 0;
    double file_io_time = 0;

    double end_to_end_decomp_time = 0;
    double decoding_time = 0;
    double prediction_reversing_time = 0;
    double decomp_file_io_time = 0;
    double comparison_time = 0;

    // data metrics
    double min = 0;
    double max = 0;
    double range = 0;
    double mean = 0;
    double stddev = 0;

    double decomp_min = 0;
    double decomp_max = 0;
    double decomp_range = 0;
    double decomp_mean = 0;
    double decomp_stddev = 0;

    double max_err = 0;
    size_t max_err_idx = 0;
    double max_abserr = 0;

    // compression metrics
    double compression_ratio = 0;
    double final_eb = 0;

    uint64_t num_outliers = 0;

    uint64_t orig_bytes = 0;
    uint64_t comp_bytes = 0;

    double bitrate = 0;
    double nrmse = 0;
    double coeff = 0;
    double psnr = 0;

  };

} // namespace fz


#endif /* FZMOD_CONFIG_HH */