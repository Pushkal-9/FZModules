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

  fz::fzmod_metrics* metrics;

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

  // take in config and create a compressor object
  Compressor(Config& config, bool is_comp = true) : conf(&config) {

    ibuffer = new InternalBuffer<T>(conf, conf->x, conf->y, conf->z, is_comp);
    metrics = new fz::fzmod_metrics();

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

  // read in file header from a file and create a compressor object
  Compressor(std::string fname, bool toFile = true) {
    
    // std::cout << "Preparing Compressor object for Decompression...\n" << std::endl;
    metrics = new fz::fzmod_metrics();

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
    conf->comp = false;
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

    // std::cout << "Config options: " << std::endl;
    // std::cout << "x, y, z: " << x << " " << y << " " << z << std::endl;
    // std::cout << "compsize " << conf->comp_size << std::endl;

    ibuffer = new InternalBuffer<T>(conf, conf->x, conf->y, conf->z, false);

    codec_hf = new CodecHF(conf->len, conf->pardeg);
    codec_fzg = new CodecFZG(conf->len);

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

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

  // compress the data
  void compress(T* input_data, uint8_t** compressed_out, cudaStream_t stream) {

    // make timer for end-to-end time
    std::chrono::time_point<std::chrono::steady_clock> total_start, total_end;
    total_start = std::chrono::steady_clock::now();

    //~~~~~~~~~~~~~~~~~~~~~~~ Preprocessor ~~~~~~~~~~~~~~~~~~~~~~~~ //

    float ms;
    CREATE_CPU_TIMER;
    START_CPU_TIMER;

    // extrema check if relative eb
    if (conf->eb_type == EB_TYPE::EB_REL) {
      double _max_val, _min_val, _range;
      fz::analysis::GPU_probe_extrema<T>(input_data, conf->len, _max_val, _min_val, _range);
      conf->eb *= _range;
      conf->logging_max = _max_val;
      conf->logging_min = _min_val;

      metrics->max = _max_val;
      metrics->min = _min_val;
      metrics->range = _range;

      // std::cout << "Max: " << conf->logging_max << std::endl;
      // std::cout << "Min: " << conf->logging_min << std::endl;
      // std::cout << "Range: " << _range << std::endl;
      // std::cout << "Relative eb: " << conf->eb << std::endl;
      // std::cout << std::endl;
    }

    STOP_CPU_TIMER;
    TIME_ELAPSED_CPU_TIMER(ms);
    metrics->preprocessing_time = ms;

    //~~~~~~~~~~~~~~~~~~~~~~~~~ Predictor ~~~~~~~~~~~~~~~~~~~~~~~~~~ //

    START_CPU_TIMER;

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

    // print the first 100 quant codes
    // auto q_codes_h = MAKE_UNIQUE_HOST(uint16_t, conf->len);
    // cudaMemcpy(q_codes_h.get(), ibuffer->ectrl(), conf->len * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    // // printf("Quantization Codes:\n");
    // for (int i = 0; i < conf->len; i++) {
    //   // if (q_codes_h.get()[i] > 900 && q_codes_h.get()[i] != 0)
    //   printf("%u ", q_codes_h.get()[i]);
    // }
    // printf("\n");

    STOP_CPU_TIMER;
    TIME_ELAPSED_CPU_TIMER(ms);
    metrics->prediction_time = ms;

    //~~~~~~~~~~~~~~~~~~~~~~~~~ Lossless Encoder 1 ~~~~~~~~~~~~~~~~~~~~~~~~~~ //

    // STATISTICS
    if (conf->codec == CODEC_HUFFMAN) {
      
      START_CPU_TIMER;

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
      
      STOP_CPU_TIMER;
      TIME_ELAPSED_CPU_TIMER(ms);
      metrics->hist_time = ms;

      // std::cout << "Statistics Finished..." << std::endl;
    } // end if not huffman

    START_CPU_TIMER;

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

    STOP_CPU_TIMER;
    TIME_ELAPSED_CPU_TIMER(ms);
    metrics->encoder_time = ms;

    // std::cout << "Encoding Finished...\n" << std::endl;

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    // HEADER STUFF

    uint32_t nbyte[END];
    nbyte[HEADER] = HEADER_SIZE;
    nbyte[ENCODED] = ibuffer->codec_comp_output_len * sizeof(uint8_t);
    nbyte[ANCHOR] = conf->algo == ALGO::ALGO_SPLINE ? conf->anchor512_len * sizeof(float) : 0;
    nbyte[SPFMT] = conf->splen * (sizeof(uint32_t) + sizeof(float));

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

    // move outliers to host and print them
    if (conf->splen != 0) {
      auto outlier_vals_h = MAKE_UNIQUE_HOST(float, conf->splen);
      auto outlier_idx_h = MAKE_UNIQUE_HOST(uint32_t, conf->splen);
      cudaMemcpy(outlier_vals_h.get(), ibuffer->compact_val(), conf->splen * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(outlier_idx_h.get(), ibuffer->compact_idx(), conf->splen * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    }

    int END = sizeof(entry) / sizeof(entry[0]);
    conf->comp_size = entry[END - 1];
    // print out compression size and entry
    std::cout << "Compression Size: " << conf->comp_size << std::endl;
    std::cout << "Entry Info: " << std::endl;
    for (auto i = 0; i < END + 1; i++) std::cout << entry[i] << " ";
    std::cout << std::endl;

    //end total time
    total_end = std::chrono::steady_clock::now();
    float total_time = std::chrono::duration<float, std::milli>(total_end - total_start).count();
    metrics->end_to_end_comp_time = total_time;

    if (conf->toFile) {
      START_CPU_TIMER;

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

      STOP_CPU_TIMER;
      TIME_ELAPSED_CPU_TIMER(ms);
      metrics->file_io_time = ms;

      // std::cout << "Compressed file written to: " << compressed_fname << std::endl;
    }

    if (conf->report) {
      print_metrics();
    }

    
  } // end compress

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

  void decompress(uint8_t* in_data, T* out_data, cudaStream_t stream, T* orig_data = nullptr) {

    if (orig_data != nullptr) {
      conf->compare = true;
    }
    
    float ms;
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

    // std::cout << "Decompressing..." << std::endl;
    // std::cout << "Decoding..." << std::endl;

    std::chrono::time_point<std::chrono::steady_clock> total_start, total_end;
    total_start = std::chrono::steady_clock::now();

    STOP_CPU_TIMER;
    TIME_ELAPSED_CPU_TIMER(ms);
    metrics->decoding_time = ms;

    START_CPU_TIMER;

    if (conf->codec == CODEC::CODEC_HUFFMAN) {
      codec_hf->decode((uint8_t*)access(ENCODED), ibuffer->ectrl(), stream);
    } else if (conf->codec == CODEC::CODEC_FZG) {
      codec_fzg->decode(
        (uint8_t*)access(ENCODED), 
        ibuffer->ectrl(), 
        conf->len, 
        stream
      );
    }

    STOP_CPU_TIMER;
    TIME_ELAPSED_CPU_TIMER(ms);
    metrics->decoding_time = ms;

    START_CPU_TIMER;

    // std::cout << "Prediction Reversing..." << std::endl;

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
    TIME_ELAPSED_CPU_TIMER(ms);
    metrics->prediction_reversing_time = ms;

    // printf("Decompression time: %f ms\n", ms);


    total_end = std::chrono::steady_clock::now();
    float total_time = std::chrono::duration<float, std::milli>(total_end - total_start).count();
    metrics->end_to_end_decomp_time = total_time;

    if (conf->toFile) {
      START_CPU_TIMER;

      auto decompressed_fname = conf->fname + ".fzmodx";
      auto decompressed_file_data_host = MAKE_UNIQUE_HOST(T, conf->len);
      cudaMemcpy(decompressed_file_data_host.get(), d_xdata, conf->orig_size, cudaMemcpyDeviceToHost);

      // // print first 10 values of decompressed data
      // printf("Decompressed Data (first 10 values):\n");
      // for (size_t i = 0; i < 100 && i < conf->len; ++i) {
      //   printf("%f ", decompressed_file_data_host.get()[i]);
      // }
      // printf("\n");

      utils::tofile(decompressed_fname.c_str(), decompressed_file_data_host.get(), conf->len);
      // std::cout << "Decompressed file written to: " << decompressed_fname << std::endl;

      STOP_CPU_TIMER;
      TIME_ELAPSED_CPU_TIMER(ms);
      metrics->file_io_time = ms;
    }

    // if comparison is on run data comparison
    if (conf->compare) {
      START_CPU_TIMER;

      compare(d_xdata, orig_data, stream);

      STOP_CPU_TIMER;
      TIME_ELAPSED_CPU_TIMER(ms);
      metrics->comparison_time = ms;
    }

    // std::cout << "Decompression Finished...\n\n\n" << std::endl;

    if (conf->report) {
      print_metrics();
    }

  }

  void decompress(std::string fname, T* out_data, cudaStream_t stream, T* orig_data = nullptr) {

    uint8_t* compressed_data, * compressed_data_device;
    cudaMallocHost(&compressed_data, conf->comp_size);
    utils::fromfile(fname, compressed_data, conf->comp_size);
    cudaMalloc(&compressed_data_device, conf->comp_size);
    cudaMemcpy(compressed_data_device, compressed_data, conf->comp_size, cudaMemcpyHostToDevice);

    T* orig_data_device = nullptr;
    if (orig_data != nullptr) {
      conf->compare = true;

      // copy original data to device
      cudaMalloc(&orig_data_device, conf->orig_size);
      cudaMemcpy(orig_data_device, orig_data, conf->orig_size, cudaMemcpyHostToDevice);
    }

    decompress(compressed_data_device, out_data, stream, orig_data_device);

    cudaFree(compressed_data_device);
    cudaFreeHost(compressed_data);
  }

  void compare(T* decomp_data, T* orig_data, cudaStream_t stream) {
    constexpr auto MINVAL = 0;
    constexpr auto MAXVAL = 1;
    constexpr auto AVGVAL = 2;

    constexpr auto SUM_CORR = 0;
    constexpr auto SUM_ERR_SQ = 1;
    constexpr auto SUM_VAR_ODATA = 2;
    constexpr auto SUM_VAR_XDATA = 3;

    T orig_data_res[4], decomp_data_res[4];

    fz::module::GPU_extrema(orig_data, conf->len, orig_data_res);
    fz::module::GPU_extrema(decomp_data, conf->len, decomp_data_res);

    T h_err[4];

    fz::module::GPU_calculate_errors<T>(
      orig_data, orig_data_res[AVGVAL], decomp_data, decomp_data_res[AVGVAL], conf->len, h_err);

    double std_orig_data = sqrt(h_err[SUM_VAR_ODATA] / conf->len);
    double std_decomp_data = sqrt(h_err[SUM_VAR_XDATA] / conf->len);
    double ee = h_err[SUM_CORR] / conf->len;

    T max_abserr{0};
    size_t max_abserr_index{0};
    fz::module::GPU_find_max_error<T>(
      decomp_data, orig_data, conf->len, max_abserr, max_abserr_index, stream);

    metrics->min = orig_data_res[MINVAL];
    metrics->max = orig_data_res[MAXVAL];
    metrics->range = orig_data_res[MAXVAL] - orig_data_res[MINVAL];
    metrics->mean = orig_data_res[AVGVAL];
    metrics->stddev = std_orig_data;

    metrics->decomp_min = decomp_data_res[MINVAL];
    metrics->decomp_max = decomp_data_res[MAXVAL];
    metrics->decomp_range = decomp_data_res[MAXVAL] - decomp_data_res[MINVAL];
    metrics->decomp_mean = decomp_data_res[AVGVAL];
    metrics->decomp_stddev = std_decomp_data;

    metrics->max_err_idx = max_abserr_index;
    metrics->max_err = max_abserr;
    metrics->max_abserr = max_abserr / metrics->range;

    metrics->coeff = ee / std_orig_data / std_decomp_data;
    double mse = h_err[SUM_ERR_SQ] / conf->len;
    metrics->nrmse = sqrt(mse) / metrics->range;
    metrics->psnr = 20 * log10(metrics->range) - 10 * log10(mse);

    double bytes = 1.0 * sizeof(T) * conf->len;
    metrics->bitrate = 32.0 / (bytes / conf->comp_size);
    metrics->compression_ratio = (float)conf->orig_size / (float)conf->comp_size;
  }

  void print_metrics() {

    auto throughput = [](double n_bytes, double time_ms) {
      return n_bytes / (1.0 * 1024  * 1024 * 1024) / (time_ms * 1e-3);
    };

    printf("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
    printf("FZMod GPU Compression Library\n");
    printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n");

    if (conf->comp) {
      printf("~~COMPRESSION~~\n");
      
      printf("Timing Metrics\n");
      printf("Preprocessing stage:\t %f ms \t %f Gib/s\n", 
        metrics->preprocessing_time, 
        throughput(conf->orig_size, metrics->preprocessing_time));
      printf("Prediction stage:\t %f ms \t %f Gib/s\n",
             metrics->prediction_time,
             throughput(conf->orig_size, metrics->prediction_time));
      printf("Histogram stage:\t %f ms \t %f Gib/s\n", metrics->hist_time,
             throughput(conf->orig_size, metrics->hist_time));
      printf("Encoding stage:\t\t %f ms \t %f Gib/s\n", metrics->encoder_time,
             throughput(conf->orig_size, metrics->encoder_time));
      printf("End-end compression:\t %f ms \t %f Gib/s\n",
             metrics->end_to_end_comp_time,
             throughput(conf->orig_size, metrics->end_to_end_comp_time));
      printf("File IO stage:\t\t %f ms \t %f Gib/s\n", metrics->file_io_time,
             throughput(conf->orig_size, metrics->file_io_time));

      printf("\n");

      printf("Compression Metrics\n");
      printf("Original size:\t\t %zu bytes\n", conf->orig_size);
      printf("Compressed size:\t %zu bytes\n", conf->comp_size);
      printf("Compression ratio:\t %f\n", (float)conf->orig_size / (float)conf->comp_size);
      printf("Num outliers:\t\t %zu\n", conf->splen);
      printf("Data Length:\t\t %zu\n", conf->len);
      printf("Data Type:\t\t %s\n", conf->precision == PRECISION::PRECISION_FLOAT ? "fp32" : "fp64");

    } else if (!conf->comp) {
      printf("~~DECOMPRESSION~~\n");

      printf("Timing Metrics\n");
      printf("Decoding Stage:\t\t %f ms %f GiB/s\n", 
        metrics->decoding_time,
        throughput(conf->orig_size, metrics->decoding_time));
      printf("Pred-Reversing Stage:\t %f ms %f GiB/s\n",
             metrics->prediction_reversing_time,
             throughput(conf->orig_size, metrics->prediction_reversing_time));
      printf("End-end decompression:\t %f ms %f GiB/s\n",
             metrics->end_to_end_decomp_time,
             throughput(conf->orig_size, metrics->end_to_end_decomp_time));
      printf("File IO time:\t\t %f ms %f GiB/s\n", metrics->file_io_time,
             throughput(conf->orig_size, metrics->file_io_time));

      printf("\n");

      //////
      
      if (conf->compare) {
        printf("~~COMPARISON~~\n");

        printf("Comparison Stage:\t\t %f ms %f GiB/s\n",
               metrics->comparison_time,
               throughput(conf->orig_size, metrics->comparison_time));
          
        printf("Data Original Min:\t\t %f\n", metrics->min);
        printf("Data Original Max:\t\t %f\n", metrics->max);
        printf("Data Original Range:\t\t %f\n", metrics->range);
        printf("Data Original Mean:\t\t %f\n", metrics->mean);
        printf("Data Original Stddev:\t\t %f\n", metrics->stddev);
        printf("\n");
        printf("Data Decompressed Min:\t\t %f\n", metrics->decomp_min);
        printf("Data Decompressed Max:\t\t %f\n", metrics->decomp_max);
        printf("Data Decompressed Range:\t %f\n", metrics->decomp_range);
        printf("Data Decompressed Mean:\t\t %f\n", metrics->decomp_mean);
        printf("Data Decompressed Stddev:\t %f\n", metrics->decomp_stddev);
        printf("\n");
        printf("Compression Ratio:\t\t %f\n", metrics->compression_ratio);
        printf("Bitrate:\t\t\t %f\n", metrics->bitrate);
        printf("NRMSE:\t\t\t\t %f\n", metrics->nrmse);
        printf("PSNR:\t\t\t\t %f\n", metrics->psnr);
        printf("coeff:\t\t\t\t %f\n", metrics->coeff);
        printf("\n");
        printf("Max Error Index:\t\t %zu\n", metrics->max_err_idx);
        printf("Max Error Value:\t\t %f\n", metrics->max_err);
        printf("Max Abs Error:\t\t\t %f\n", metrics->max_abserr);
        printf("\n");
      }
    }

    printf("\n");
  }

};


} // namespace fz

// int compress(Config conf, float* input_data, uint8_t* compressed);


#endif /* FZMOD_API_HH */