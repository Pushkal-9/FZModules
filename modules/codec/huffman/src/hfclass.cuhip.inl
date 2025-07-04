/**
 * @file hfclass.cuhip.inl
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2023-06-13
 * (created) 2020-04-24
 *
 * @copyright (C) 2020 by Washington State University, The University of
 * Alabama, Argonne National Laboratory
 * @copyright (C) 2021 by Washington State University, Argonne National
 * Laboratory See LICENSE in top-level directory
 *
 */

#ifndef ABBC78E4_3E65_4633_9BEA_27823AB7C398
#define ABBC78E4_3E65_4633_9BEA_27823AB7C398

#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <cuda_runtime.h>

#include "cxx_hfbk.h"
#include "hf_type.h"
#include "hfbuf.inl"
#include "hfclass.hh"
#include "hfcxx_module.hh"

// #include "mem/cxx_memobj.h"
#include "mem/cxx_sp_gpu.h"

#include "io.hh"
#include "timer.hh"

#define PHF_TPL template <typename E>
#define PHF_CLASS HuffmanCodec<E>

using _portable::utils::tofile;

namespace phf {

template <typename E>
struct HuffmanCodec<E>::impl {
  static const bool TimeBreakdown{false};

  using phf_module = phf::cuhip::modules<E, H>;
  using Header = phf_header;

  phf_header header;  // TODO combine psz and pszhf headers

  hires::time_point f, e, d, c, b, a;
  float _time_book{0.0}, _time_lossless{0.0};
  float time_book() const { return _time_book; }
  float time_codec() const { return _time_lossless; }
  size_t inlen() const { return len; };

  cudaEvent_t event_start, event_end;

  size_t pardeg, sublen;
  int numSMs;
  size_t len;
  static constexpr uint16_t max_bklen = 1024;
  uint16_t rt_bklen;

  GPU_unique_hptr<uint32_t[]> h_hist;

  Buf* buf;

  impl() 
    : header(),  // Initialize header
      // Initialize the timepoints
      f(),
      e(),
      d(),
      c(),
      b(),
      a(),
      _time_book(0.0f), 
      _time_lossless(0.0f),
      event_start(nullptr),  // Initialize CUDA event pointers
      event_end(nullptr),
      pardeg(0),
      sublen(0),
      numSMs(0),
      len(0),
      rt_bklen(0),
      h_hist(nullptr),  // Initialize unique pointers
      buf(nullptr)
  {}

  ~impl()
  {
    delete buf;
    event_destroy_pair(event_start, event_end);
  }

  impl(const impl&) = delete;
  impl& operator=(const impl&) = delete;

  impl(impl&&) noexcept = delete;
  impl& operator=(impl&&) noexcept = delete;

  // keep ctor short
  void init(size_t const inlen, int const _pardeg, bool debug)
  {
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

    pardeg = _pardeg;
    len = inlen;
    sublen = (inlen - 1) / pardeg + 1;

    // TODO make unique_ptr; modify ctor
    buf = new Buf(inlen, max_bklen, _pardeg, true, debug);
    h_hist = MAKE_UNIQUE_HOST(uint32_t, max_bklen);

    std::tie(event_start, event_end) = event_create_pair();
  }

#ifdef ENABLE_HUFFBK_GPU
  // build Huffman tree on GPU
  void buildbook(uint32_t* freq, int const bklen, phf_stream_t stream)
  {
    psz::hf_buildbook<CUDA, E, H4>(
        freq, bklen, bk4->dptr(), revbk4->dptr(), revbk4_bytes(bklen), &_time_book,
        (cudaStream_t)stream);
  }
#else

  // build Huffman tree on CPU
  void buildbook(const uint32_t* freq, uint16_t const _rt_bklen, phf_stream_t stream)
  {
    rt_bklen = _rt_bklen;
    // memcpy_allkinds<D2H>(h_hist.get(), freq, bklen);
    cudaMemcpy(h_hist.get(), freq, rt_bklen * sizeof(uint32_t), cudaMemcpyDeviceToHost);



    phf_CPU_build_canonized_codebook_v2<E, H4>(
        h_hist.get(), rt_bklen, buf->h_bk4.get(), buf->h_revbk4.get(),
        &_time_book);

    // memcpy_allkinds_async<H2D>(buf->d_bk4.get(), buf->h_bk4.get(), rt_bklen, (cudaStream_t)stream);
    // memcpy_allkinds_async<H2D>(buf->d_revbk4.get(), buf->h_revbk4.get(), buf->revbk4_bytes, (cudaStream_t)stream);

    cudaMemcpyAsync(buf->d_bk4.get(), buf->h_bk4.get(), rt_bklen * sizeof(H4), cudaMemcpyHostToDevice, (cudaStream_t)stream);

    cudaMemcpyAsync(buf->d_revbk4.get(), buf->h_revbk4.get(), buf->revbk4_bytes, cudaMemcpyHostToDevice, (cudaStream_t)stream);

    cudaStreamSynchronize((cudaStream_t)stream);
  }
#endif

  void encode(E* in, size_t const len, uint8_t** out, size_t* outlen, phf_stream_t stream)
  {
    _time_lossless = 0;

    phf::par_config hfpar{sublen, pardeg};

    event_recording_start(event_start, stream);

    phf_module::GPU_coarse_encode_phase1(
        {in, len}, {buf->d_bk4.get(), rt_bklen}, numSMs, {buf->d_scratch4.get(), len}, stream);

    phf_module::GPU_coarse_encode_phase2(
        {buf->d_scratch4.get(), len}, hfpar, {buf->d_scratch4.get(), len} /* placeholder */,
        {buf->d_par_nbit.get(), buf->pardeg}, {buf->d_par_ncell.get(), buf->pardeg}, stream);

    // sync_by_stream(stream);

    phf_module::GPU_coarse_encode_phase3_sync(
        {buf->d_par_nbit.get(), buf->pardeg}, {buf->d_par_ncell.get(), buf->pardeg},
        {buf->d_par_entry.get(), buf->pardeg}, hfpar, {buf->h_par_nbit.get(), buf->pardeg},
        {buf->h_par_ncell.get(), buf->pardeg}, {buf->h_par_entry.get(), buf->pardeg},
        &header.total_nbit, &header.total_ncell, nullptr, stream);

    // sync_by_stream(stream);

    phf_module::GPU_coarse_encode_phase4(
        {buf->d_scratch4.get(), len}, {buf->d_par_entry.get(), buf->pardeg},
        {buf->d_par_ncell.get(), buf->pardeg}, hfpar,
        {buf->d_bitstream4.get(), buf->bitstream_max_len}, stream);

    event_recording_stop(event_end, stream);
    event_time_elapsed(event_start, event_end, &_time_lossless);

    cudaStreamSynchronize((cudaStream_t) stream);

    // move rebk4 (first 10 entries) to host and print
    // uint32_t revbk4[10];
    // cudaMemcpyAsync(revbk4, buf->d_revbk4.get(), 10 * sizeof(uint32_t), cudaMemcpyDeviceToHost, (cudaStream_t)stream);
    // cudaStreamSynchronize((cudaStream_t) stream);
    // printf("revbk4: [0]=%u [1]=%u [2]=%u [3]=%u [4]=%u [5]=%u\n",
    //        revbk4[0], revbk4[1], revbk4[2], revbk4[3], revbk4[4], revbk4[5]);

    make_metadata();
    buf->memcpy_merge(header, stream);  // TODO externalize/make explicit

    if constexpr (TimeBreakdown) {
      std::cout << "phase1: " << static_cast<duration_t>(b - a).count() * 1e6 << std::endl;
      std::cout << "phase2: " << static_cast<duration_t>(c - a).count() * 1e6 << std::endl;
      std::cout << "phase3: " << static_cast<duration_t>(d - a).count() * 1e6 << std::endl;
      std::cout << "phase4: " << static_cast<duration_t>(e - a).count() * 1e6 << std::endl;
      std::cout << "wrapup: " << static_cast<duration_t>(f - a).count() * 1e6 << std::endl;
    }

    // TODO may cooperate with upper-level; output
    *out = buf->d_encoded;
    *outlen = phf_encoded_bytes(&header);
  }




  void decode(uint8_t* in_encoded, E* out_decoded, phf_stream_t stream, bool header_on_device)
  {
    Header header;
    if (header_on_device)
      CHECK_GPU(cudaMemcpyAsync(
          &header, in_encoded, sizeof(header), cudaMemcpyDeviceToHost, (cudaStream_t)stream));

#define PHF_ACCESSOR(SYM, TYPE) reinterpret_cast<TYPE*>(in_encoded + header.entry[PHFHEADER_##SYM])

    event_recording_start(event_start, stream);

    phf_module::GPU_coarse_decode(
        PHF_ACCESSOR(BITSTREAM, H4), PHF_ACCESSOR(REVBK, PHF_BYTE), buf->revbk4_bytes,
        PHF_ACCESSOR(PAR_NBIT, M), PHF_ACCESSOR(PAR_ENTRY, M), header.sublen, header.pardeg,
        out_decoded, stream);

    event_recording_stop(event_end, stream);
    event_time_elapsed(event_start, event_end, &_time_lossless);

#undef PHF_ACCESSOR
  }


  

  void make_metadata()
  {
    // header.self_bytes = sizeof(Header);
    header.bklen = rt_bklen;
    header.sublen = sublen;
    header.pardeg = pardeg;
    header.original_len = len;

    M nbyte[PHFHEADER_END];
    nbyte[PHFHEADER_HEADER] = PHFHEADER_FORCED_ALIGN;
    nbyte[PHFHEADER_REVBK] = buf->revbk4_bytes;
    nbyte[PHFHEADER_PAR_NBIT] = buf->pardeg * sizeof(M);
    nbyte[PHFHEADER_PAR_ENTRY] = buf->pardeg * sizeof(M);
    nbyte[PHFHEADER_BITSTREAM] = 4 * header.total_ncell;

    header.entry[0] = 0;
    // *.END + 1: need to know the ending position
    for (auto i = 1; i < PHFHEADER_END + 1; i++) header.entry[i] = nbyte[i - 1];
    for (auto i = 1; i < PHFHEADER_END + 1; i++) header.entry[i] += header.entry[i - 1];

    // print header entries
    printf("HF CODEC - Header entries: [0]=%u [1]=%u [2]=%u [3]=%u [4]=%u [5]=%u\n",
           header.entry[0], header.entry[1], header.entry[2], header.entry[3],
           header.entry[4], header.entry[5]);
  }

  void clear_buffer() { buf->clear_buffer(); }

};  // end of pimpl class

PHF_TPL PHF_CLASS::HuffmanCodec(size_t const inlen, int const pardeg, bool debug) :
    pimpl{std::make_unique<impl>()},
    in_dtype{
        std::is_same_v<E, uint8_t>   ? HF_U1
        : std::is_same_v<E, uint16_t> ? HF_U2
        : std::is_same_v<E, uint32_t> ? HF_U4
                                : HF_INVALID}
{
  pimpl->init(inlen, pardeg, debug);
};

PHF_TPL PHF_CLASS::~HuffmanCodec(){};

// using CPU huffman
PHF_TPL PHF_CLASS* PHF_CLASS::buildbook(const uint32_t* freq, uint16_t const rt_bklen, phf_stream_t stream)
{
  pimpl->buildbook(freq, rt_bklen, stream);
  return this;
}

PHF_TPL
PHF_CLASS* PHF_CLASS::encode(
    E* in, size_t const len, uint8_t** out, size_t* outlen, phf_stream_t stream)
{
  pimpl->encode(in, len, out, outlen, stream);
  return this;
}

PHF_TPL PHF_CLASS* PHF_CLASS::decode(
    uint8_t* in_encoded, E* out_decoded, phf_stream_t stream, bool header_on_device)
{
  pimpl->decode(in_encoded, out_decoded, stream, header_on_device);
  return this;
}

PHF_TPL PHF_CLASS* PHF_CLASS::clear_buffer()
{
  pimpl->clear_buffer();
  return this;
}

PHF_TPL PHF_CLASS* PHF_CLASS::dump_internal_data(std::string field, std::string fname)
{
  auto ofname = fname + ".book_u4";
  if (field == "book") _portable::utils::tofile(ofname, pimpl->buf->h_bk4.get(), pimpl->rt_bklen);
  return this;
}

PHF_TPL float PHF_CLASS::time_book() const { return pimpl->time_book(); }
PHF_TPL float PHF_CLASS::time_lossless() const { return pimpl->time_codec(); }
PHF_TPL size_t PHF_CLASS::inlen() const { return pimpl->inlen(); }

}  // namespace phf

#undef PHF_ACCESSOR
#undef PHF_TPL
#undef PHF_CLASS

#endif /* ABBC78E4_3E65_4633_9BEA_27823AB7C398 */
