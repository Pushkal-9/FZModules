#ifndef FZMOD_API_HH
#define FZMOD_API_HH

#include "codec.hh"
#include "predictor.hh"
#include "stat.hh"

#include "io.hh"
#include "mem/buffer.hh"
#include "config.hh"
#include "compare.hh"

// #define MAKE_STDLEN3(X, Y, Z) \
//   std::array<size_t, 3> { X, Y, Z }

namespace fz {

template <typename T>
struct Compressor {

  Config* conf;
  InternalBuffer<T>* ibuffer;

  using CodecHF = phf::HuffmanCodec<uint16_t>;
  using CodecFZG = fz::FzgCodec;
  CodecHF* codec_hf;
  CodecFZG* codec_fzg;

  // histogram
  int hist_generic_grid_dim, hist_generic_block_dim, hist_generic_shmem_use, hist_generic_repeat;

  static const int HEADER = 0;
  static const int ANCHOR = 1;
  static const int ENCODED = 2;
  static const int SPFMT = 3;
  static const int END = 4;

  Compressor(Config& config) : conf(&config) {

    ibuffer = new InternalBuffer<T>(conf, conf->x, conf->y, conf->z);

    capi_phf_coarse_tune(conf->len, &conf->sublen, &conf->pardeg);
    fz::module::GPU_histogram_generic_optimizer_on_initialization<uint16_t>(
      conf->len, conf->radius*2, hist_generic_grid_dim, hist_generic_block_dim, hist_generic_shmem_use,
      hist_generic_repeat);

    codec_hf = new CodecHF(conf->len, conf->pardeg);
    codec_fzg = new CodecFZG(conf->len);

    std::cout << "Compressor Created" << std::endl;
  }

  ~Compressor() {
    delete codec_hf;
    delete codec_fzg;
    delete ibuffer;
    std::cout << "Compressor Destroyed" << std::endl;
  }

  void compress(T* input_data, uint8_t** compressed_out, cudaStream_t stream) {
    
    // extrema check if relative eb
    if (conf->eb_type == EB_TYPE::EB_REL) {
      double _max_val, _min_val, _range;
      analysis::GPU_probe_extrema<T>(input_data, conf->len, _max_val, _min_val, _range);
      conf->eb = _range;
      conf->logging_max = _max_val;
      conf->logging_min = _min_val;

      std::cout << "Max: " << conf->logging_max << std::endl;
      std::cout << "Min: " << conf->logging_min << std::endl;
      std::cout << "Range: " << _range << std::endl;

      std::cout << "Relative eb: " << conf->eb << std::endl;
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    auto len3 = MAKE_STDLEN3(conf->x, conf->y, conf->z);
    double eb = conf->eb;
    double eb_r = 1/eb;
    double ebx2 = eb*2;
    double ebx2_r = 1/ebx2;

    // PREDICTOR
    if (conf->algo == ALGO::ALGO_LORENZO) {
      if (conf->use_lorenzo_regular) {

        fz::module::GPU_c_lorenzo_nd_with_outlier<T, false, uint16_t>(
          input_data, 
          len3, 
          ibuffer->ectrl(), 
          (void*)ibuffer->outlier(), 
          ibuffer->top1(),
          ebx2, 
          ebx2_r, 
          conf->radius, 
          stream);
        

      } else if (conf->use_lorenzo_zigzag) {
        fz::module::GPU_c_lorenzo_nd_with_outlier<T, true, uint16_t>(
          input_data, len3, ibuffer->ectrl(), (void*)ibuffer->outlier(), ibuffer->top1(),
          ebx2, ebx2_r, conf->radius, stream);
      }
    } else if (conf->algo == ALGO::ALGO_SPLINE) {
      fz::module::GPU_predict_spline(
        input_data, len3, ibuffer->ectrl(), ibuffer->ectrl_len3(), ibuffer->anchor(), 
        conf->anchor_len3(), (void*)ibuffer->compact, ebx2, eb_r, conf->radius, stream);
    }

    cudaStreamSynchronize((cudaStream_t)stream);
    // make outlier seen on host
    conf->splen = ibuffer->compact->num_outliers();

    std::cout << "Predictor Finished..." << std::endl;

    // STATISTICS
    if (conf->codec == CODEC_HUFFMAN) {
      if (conf->use_histogram_sparse) {
        fz::module::GPU_histogram_Cauchy<uint16_t>(
          ibuffer->ectrl(),
          ibuffer->len,
          ibuffer->hist(),
          (conf->radius * 2),
          stream);
      } else if (conf->use_histogram_generic) {
        fz::module::GPU_histogram_generic<uint16_t>(
          ibuffer->ectrl(),
          ibuffer->len,
          ibuffer->hist(),
          static_cast<uint16_t>(conf->radius * 2),
          hist_generic_grid_dim, hist_generic_block_dim, hist_generic_shmem_use,
          hist_generic_repeat, stream);
      }
      std::cout << "Statistics Finished..." << std::endl;
    } // end if not huffman

    // ENCODING
    if (conf->codec == CODEC::CODEC_HUFFMAN) {
      codec_hf->buildbook(
        ibuffer->hist(),
        (conf->radius * 2),
        stream
      )->encode(
        ibuffer->ectrl(),
        ibuffer->len,
        &ibuffer->codec_comp_output,
        &ibuffer->codec_comp_output_len,
        stream
      );
    } else if (conf->codec == CODEC::CODEC_FZG) {
      codec_fzg->encode(
        ibuffer->ectrl(),
        ibuffer->len,
        &ibuffer->codec_comp_output,
        &ibuffer->codec_comp_output_len,
        stream
      );
    }

    std::cout << "Encoding Finished...\n" << std::endl;

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    // HEADER STUFF

    uint32_t nbyte[END];
    nbyte[HEADER] = 128;
    nbyte[ENCODED] = ibuffer->codec_comp_output_len * sizeof(uint8_t);
    nbyte[ANCHOR] = conf->algo == ALGO::ALGO_SPLINE ? conf->anchor512_len * sizeof(float) : 0;
    nbyte[SPFMT] = conf->splen * (sizeof(uint32_t) + sizeof(float));

    std::cout << "Header Byte Info " << std::endl;
    std::cout << nbyte[HEADER] << " " << nbyte[ENCODED] << " " << nbyte[ANCHOR] << " " << nbyte[SPFMT] << std::endl;

    uint32_t entry[END + 1];
    entry[0] = 0;
    for (auto i = 1; i < END + 1; i++) entry[i] = nbyte[i-1];
    for (auto i = 1; i < END + 1; i++) entry[i] += entry[i-1];

    #define DST(FIELD, OFFSET) ((void*)(ibuffer->compressed() + entry[FIELD] + OFFSET))

    #define CONCAT_ON_DEVICE(dst, src, nbyte, stream) \
    if (nbyte != 0) cudaMemcpyAsync(dst, src, nbyte, cudaMemcpyDeviceToDevice, (cudaStream_t)stream);

    CONCAT_ON_DEVICE(DST(ANCHOR, 0), ibuffer->anchor(), nbyte[ANCHOR], stream);
    CONCAT_ON_DEVICE(DST(ENCODED, 0), ibuffer->codec_comp_output, nbyte[ENCODED], stream);
    CONCAT_ON_DEVICE(DST(SPFMT, 0), ibuffer->compact_val(), conf->splen * sizeof(float), stream);
    CONCAT_ON_DEVICE(DST(SPFMT, conf->splen * sizeof(float)), ibuffer->compact_idx(), conf->splen * sizeof(uint32_t), stream);

    // output compression
    *compressed_out = ibuffer->compressed();
    
    int END = sizeof(entry) / sizeof(entry[0]);
    conf->comp_size = entry[END - 1];
    
  }

};

} // namespace fz

// int compress(Config conf, float* input_data, uint8_t* compressed);


#endif /* FZMOD_API_HH */