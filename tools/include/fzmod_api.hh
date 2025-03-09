#ifndef FZMOD_API_HH
#define FZMOD_API_HH

#include "codec.hh"
#include "predictor.hh"
#include "stat.hh"

#include "io.hh"
#include "timer.hh"
#include "mem/buffer.hh"
#include "config.hh"

#define HEADER_SIZE 128

namespace utils = _portable::utils;

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

  uint32_t entries[END + 1];

  Compressor(Config& config, bool is_comp = true) : conf(&config) {

    ibuffer = new InternalBuffer<T>(conf, conf->x, conf->y, conf->z, is_comp);

    if (is_comp) {
      capi_phf_coarse_tune(conf->len, &conf->sublen, &conf->pardeg);
      fz::module::GPU_histogram_generic_optimizer_on_initialization<uint16_t>(
        conf->len, conf->radius*2, hist_generic_grid_dim, hist_generic_block_dim, hist_generic_shmem_use,
        hist_generic_repeat);
    }
    
    codec_hf = new CodecHF(conf->len, conf->pardeg);
    codec_fzg = new CodecFZG(conf->len);

    // std::cout << "Compressor Created" << std::endl;
  }

  Compressor(std::string fname, bool toFile = true) {
    
    std::cout << "Preparing Compressor object for Decompression...\n" << std::endl;
    
    uint8_t* header;
    cudaMallocHost(&header, HEADER_SIZE);
    utils::fromfile(fname, header, HEADER_SIZE);

    std::string basename = fname.substr(0, fname.rfind("."));
    
    // read in file header (128 bytes)
    PRECISION p;
    ALGO a;
    uint32_t hist_type;
    CODEC c;
    uint32_t future_codec_spot;
    EB_TYPE e;
    double eb;
    uint16_t radius;
    int sublen;
    int pardeg;
    uint32_t entries_file[END + 1];
    uint32_t x, y, z, w;
    size_t splen;
    double user_input_eb;
    double logging_max;
    double logging_min;

    const uint8_t* ptr = header;
    size_t offset = 0;

    auto readField = [&ptr, &offset](void* data, size_t dataSize) {
      if (offset + dataSize > HEADER_SIZE) {
        std::cerr << "Buffer overrun detected!" << std::endl;
        std::abort();
      }
      std::memcpy(data, ptr + offset, dataSize);
      offset += dataSize;
    };

    readField(&p, sizeof(PRECISION));
    readField(&a, sizeof(ALGO));
    readField(&hist_type, sizeof(uint32_t));
    readField(&c, sizeof(CODEC));
    readField(&future_codec_spot, sizeof(uint32_t));
    readField(&e, sizeof(EB_TYPE));
    readField(&eb, sizeof(double));
    readField(&radius, sizeof(uint16_t));
    readField(&sublen, sizeof(int));
    readField(&pardeg, sizeof(int));
    readField(&entries_file, sizeof(entries_file));
    readField(&x, sizeof(uint32_t));
    readField(&y, sizeof(uint32_t));
    readField(&z, sizeof(uint32_t));
    readField(&w, sizeof(uint32_t));
    readField(&splen, sizeof(size_t));
    readField(&user_input_eb, sizeof(double));
    readField(&logging_max, sizeof(double));
    readField(&logging_min, sizeof(double));

    // make new config
    conf = new Config(x, y, z);
    conf->eb = eb;
    conf->eb_type = e;
    conf->algo = a;
    conf->precision = p;
    conf->codec = c;
    conf->fname = basename;
    conf->splen = splen;
    conf->radius = radius;
    conf->sublen = sublen;
    conf->pardeg = pardeg;
    conf->logging_max = logging_max;
    conf->logging_min = logging_min;
    conf->comp_size = entries_file[END];

    conf-> toFile = toFile;

    for (auto i = 0; i < END + 1; i++) entries[i] = entries_file[i];

    std::cout << "Config options: " << std::endl;
    std::cout << "x, y, z: " << x << " " << y << " " << z << std::endl;
    std::cout << "compsize " << conf->comp_size << std::endl;

    ibuffer = new InternalBuffer<T>(conf, conf->x, conf->y, conf->z, false);

    codec_hf = new CodecHF(conf->len, conf->pardeg);
    codec_fzg = new CodecFZG(conf->len);

    std::cout << "Compressor Created" << std::endl;

    // free header
    cudaFreeHost(header);
  }

  ~Compressor() {
    // std::cout << "Compressor Destroying..." << std::endl;
    if (codec_hf) delete codec_hf;
    if (codec_fzg) delete codec_fzg;
    delete ibuffer;
    // std::cout << "Compressor Destroyed" << std::endl;
  }

  void compress(T* input_data, uint8_t** compressed_out, cudaStream_t stream) {

    CREATE_CPU_TIMER;
    START_CPU_TIMER;
    
    // extrema check if relative eb
    if (conf->eb_type == EB_TYPE::EB_REL) {
      double _max_val, _min_val, _range;
      fz::analysis::GPU_probe_extrema<T>(input_data, conf->len, _max_val, _min_val, _range);
      conf->eb *= _range;
      conf->logging_max = _max_val;
      conf->logging_min = _min_val;

      std::cout << "Max: " << conf->logging_max << std::endl;
      std::cout << "Min: " << conf->logging_min << std::endl;
      std::cout << "Range: " << _range << std::endl;

      std::cout << "Relative eb: " << conf->eb << std::endl;
      std::cout << std::endl;
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

    // print num outliers
    std::cout << "Num Outliers: " << conf->splen << std::endl;

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
    nbyte[HEADER] = HEADER_SIZE;
    nbyte[ENCODED] = ibuffer->codec_comp_output_len * sizeof(uint8_t);
    nbyte[ANCHOR] = conf->algo == ALGO::ALGO_SPLINE ? conf->anchor512_len * sizeof(float) : 0;
    nbyte[SPFMT] = conf->splen * (sizeof(uint32_t) + sizeof(float));

    std::cout << "Header Byte Info " << std::endl;
    std::cout << nbyte[HEADER] << " " << nbyte[ANCHOR] << " " << nbyte[ENCODED] << " " << nbyte[SPFMT] << std::endl;

    uint32_t entry[END + 1];
    entry[0] = 0;
    for (auto i = 1; i < END + 1; i++) entry[i] = nbyte[i-1];
    for (auto i = 1; i < END + 1; i++) entry[i] += entry[i-1];

    // std::cout << "Entry Info " << std::endl;
    // for (auto i = 0; i < END + 1; i++) std::cout << entry[i] << " ";
    // std::cout << std::endl;

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

    STOP_CPU_TIMER;
    float ms;
    TIME_ELAPSED_CPU_TIMER(ms);

    printf("Compression time: %f ms\n", ms);

    if (conf->toFile) {
      auto compressed_fname = conf->fname + ".fzmod";
      auto file = MAKE_UNIQUE_HOST(uint8_t, conf->comp_size);

      // copy header info to beginning of file
      PRECISION p = conf->precision;
      ALGO a = conf->algo;
      uint32_t hist_type = conf->use_histogram_sparse ? 0 : 1;
      CODEC c = conf->codec;
      uint32_t future_codec_spot = 0;
      EB_TYPE e = conf->eb_type;
      double eb = conf->eb;
      uint16_t radius = conf->radius;
      int sublen = conf->sublen;
      int pardeg = conf->pardeg;
      uint32_t entries_file[END];
      for (auto i = 0; i < END; i++) entries_file[i] = entry[i];
      uint32_t x, y, z;
      x = conf->x;
      y = conf->y;
      z = conf->z;
      uint32_t w = 1;
      size_t splen = conf->splen;
      double user_input_eb = conf->eb;
      double logging_max = conf->logging_max;
      double logging_min = conf->logging_min;

      std::array<uint8_t, HEADER_SIZE> buffer{};
      uint8_t* ptr = buffer.data();
      size_t offset = 0;

      auto copyField = [&ptr, &offset](const void* data, size_t dataSize) {
        if (offset + dataSize <= HEADER_SIZE) {
          std::memcpy(ptr + offset, data, dataSize);
          offset += dataSize;
        }
      };

      copyField(&p, sizeof(PRECISION));
      copyField(&a, sizeof(ALGO));
      copyField(&hist_type, sizeof(uint32_t));
      copyField(&c, sizeof(CODEC));
      copyField(&future_codec_spot, sizeof(uint32_t));
      copyField(&e, sizeof(EB_TYPE));
      copyField(&eb, sizeof(double));
      copyField(&radius, sizeof(uint16_t));
      copyField(&sublen, sizeof(int));
      copyField(&pardeg, sizeof(int));
      copyField(&entries_file, sizeof(entries_file));
      copyField(&x, sizeof(uint32_t));
      copyField(&y, sizeof(uint32_t));
      copyField(&z, sizeof(uint32_t));
      copyField(&w, sizeof(uint32_t));
      copyField(&splen, sizeof(size_t));
      copyField(&user_input_eb, sizeof(double));
      copyField(&logging_max, sizeof(double));
      copyField(&logging_min, sizeof(double));

      cudaMemcpy(file.get(), ibuffer->compressed(), conf->comp_size, cudaMemcpyDeviceToHost);
      std::memcpy(file.get(), buffer.data(), HEADER_SIZE);
      utils::tofile(compressed_fname.c_str(), file.get(), conf->comp_size);

      std::cout << "Compressed file written to: " << compressed_fname << std::endl;
    }


    
  } // end compress

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

  void decompress(uint8_t* in_data, T* out_data, cudaStream_t stream) {
    
    CREATE_CPU_TIMER;
    START_CPU_TIMER;

    auto access = [&](int FIELD, size_t offset_nbyte = 0) {
      return (void*)(in_data + entries[FIELD] + offset_nbyte);
    };

    auto d_anchor = (T*)access(ANCHOR);
    auto d_spval = (T*)access(SPFMT);
    auto d_spidx = (uint32_t*)access(SPFMT, conf->splen * sizeof(T));
    auto d_space = out_data, d_xdata = out_data; // aliases
    auto len3_std = MAKE_STDLEN3(conf->x, conf->y, conf->z);

    double eb = conf->eb;
    double eb_r = 1/eb, ebx2 = eb*2, ebx2_r = 1/ebx2;


    if (conf->splen != 0) {
      fz::spv_scatter_naive<T, uint32_t>(d_spval, d_spidx, conf->splen, d_space, nullptr, stream);
    }

    std::cout << "Decompressing..." << std::endl;

    std::cout << "Decoding..." << std::endl;

    if (conf->codec == CODEC::CODEC_HUFFMAN) {
      codec_hf->decode((uint8_t*)access(ENCODED), ibuffer->ectrl(), stream);
    } else if (conf->codec == CODEC::CODEC_FZG) {
      codec_fzg->decode((uint8_t*)access(ENCODED), conf->comp_size, ibuffer->ectrl(), conf->len, stream);
    }

    std::cout << "Prediction Reversing..." << std::endl;

    if (conf->algo == ALGO::ALGO_LORENZO) {
      if (conf->use_lorenzo_regular) {
        fz::module::GPU_x_lorenzo_nd<T, false, uint16_t>(
          ibuffer->ectrl(), d_space, d_xdata, len3_std, ebx2, ebx2_r, conf->radius, stream);
      } else if (conf->use_lorenzo_zigzag) {
        fz::module::GPU_x_lorenzo_nd<T, true, uint16_t>(
          ibuffer->ectrl(), d_space, d_xdata, len3_std, ebx2, ebx2_r, conf->radius, stream);
      }
    } else if (conf->algo == ALGO::ALGO_SPLINE) {
      fz::module::GPU_reverse_predict_spline(
        ibuffer->ectrl(), ibuffer->ectrl_len3(), d_anchor, conf->anchor_len3(),
        d_xdata, len3_std, ebx2, eb_r, conf->radius, stream);
    }

    STOP_CPU_TIMER;
    float ms;
    TIME_ELAPSED_CPU_TIMER(ms);

    printf("Decompression time: %f ms\n", ms);

    if (conf->toFile) {
      auto decompressed_fname = conf->fname + ".fzmodx";
      auto decompressed_file_data_host = MAKE_UNIQUE_HOST(T, conf->len);
      cudaMemcpy(decompressed_file_data_host.get(), d_xdata, conf->orig_size, cudaMemcpyDeviceToHost);
      utils::tofile(decompressed_fname.c_str(), decompressed_file_data_host.get(), conf->orig_size);
      std::cout << "Decompressed file written to: " << decompressed_fname << std::endl;
    }

    std::cout << "Decompression Finished...\n\n\n" << std::endl;

  }

  void decompress(std::string fname, T* out_data, cudaStream_t stream) {

    uint8_t* compressed_data, * compressed_data_device;
    cudaMallocHost(&compressed_data, conf->comp_size);
    utils::fromfile(fname, compressed_data, conf->comp_size);
    cudaMalloc(&compressed_data_device, conf->comp_size);
    cudaMemcpy(compressed_data_device, compressed_data, conf->comp_size, cudaMemcpyHostToDevice);

    decompress(compressed_data_device, out_data, stream);

    cudaFree(compressed_data_device);
    cudaFreeHost(compressed_data);
  }

};


} // namespace fz

// int compress(Config conf, float* input_data, uint8_t* compressed);


#endif /* FZMOD_API_HH */