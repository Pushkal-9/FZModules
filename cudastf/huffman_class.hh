#pragma once
#include <memory>

#include "huffman/include/hfcanon.hh"
#include "huffman/include/hfword.hh"
#include "huffman/include/hfbk_impl.hh"
#include "huffman/src/hfbk_impl1.seq.cc"

#include "huffman_kernels.cu"

using namespace cuda::experimental::stf;

#define HF_HEADER_FORCED_ALIGN 128
#define HF_HEADER_HEADER 0
#define HF_HEADER_REVBK 1
#define HF_HEADER_PAR_NBIT 2
#define HF_HEADER_PAR_ENTRY 3
#define HF_HEADER_BITSTREAM 4
#define HF_HEADER_END 5

#define COMPRESSION_PIPELINE_HEADER_SIZE 128

typedef struct {
  int bklen : 16;
  int sublen, pardeg;
  size_t original_len;
  size_t total_nbit, total_ncell;
  uint32_t entry[HF_HEADER_END + 1];
} hf_header;

struct HuffmanHelper {
  static const int BLOCK_DIM_ENCODE = 256;
  static const int BLOCK_DIM_DEFLATE = 256;

  static const int ENC_SEQUENTIALITY = 4;  // empirical
  static const int DEFLATE_CONSTANT = 4;   // deflate_chunk_constant
};

size_t capi_phf_coarse_tune_sublen(size_t len)
{
  auto div = [](auto _l, auto _subl) { return (_l - 1) / _subl + 1; };

  // TODO ROCm GPUs should use different constants.
  int current_dev = 0;
  cudaSetDevice(current_dev);
  cudaDeviceProp dev_prop{};
  cudaGetDeviceProperties(&dev_prop, current_dev);

  auto nSM = dev_prop.multiProcessorCount;
  auto allowed_block_dim = dev_prop.maxThreadsPerBlock;
  auto deflate_nthread = allowed_block_dim * nSM / HuffmanHelper::DEFLATE_CONSTANT;

  auto optimal_sublen = div(len, deflate_nthread);
  // round up
  optimal_sublen =
      div(optimal_sublen, HuffmanHelper::BLOCK_DIM_DEFLATE) * HuffmanHelper::BLOCK_DIM_DEFLATE;

  return optimal_sublen;
};

void capi_phf_coarse_tune(size_t len, int* sublen, int* pardeg) {
  auto div = [](auto l, auto subl) { return (l - 1) / subl + 1; };
  *sublen = capi_phf_coarse_tune_sublen(len);
  *pardeg = div(len, *sublen);
}

class HuffmanCodecSTF {

  public:

  int num_SMs;
  size_t pardeg, len, sublen;
  static constexpr uint16_t max_bklen  = 1024;
  size_t revbk4_bytes;
  uint16_t rt_bklen;
  size_t total_nbit, total_ncell = 0;
  hf_header header;

  HuffmanCodecSTF(size_t const inlen, int const _pardeg, context& ctx) {
    cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, 0);
    pardeg = _pardeg;
    len = inlen;
    sublen = (len - 1) / pardeg + 1;
    revbk4_bytes = 4 * (2 * 4 * 8) + sizeof(uint16_t) * max_bklen;
  }

  void buildbook(uint16_t const _rt_bklen, stf_internal_buffers& ibuffer, context& ctx, cudaStream_t stream) {
    
    using PW4 = HuffmanWord<4>;
    using PW8 = HuffmanWord<8>;

    rt_bklen = _rt_bklen;
    auto reverse_book_len = revbk4_bytes;

    auto bk_bytes = sizeof(uint32_t) * _rt_bklen;
    constexpr auto TYPE_BITS = sizeof(uint32_t) * 8;

    // Create shared pointers for task usage
    auto space_ptr = std::make_shared<hf_canon_reference<uint16_t, uint32_t>>(_rt_bklen);
    auto freq_ptr = std::shared_ptr<uint32_t[]>(new uint32_t[4 * _rt_bklen]);
    auto host_bk4_ptr = std::shared_ptr<uint32_t[]>(new uint32_t[_rt_bklen]);

    // Get raw pointers for local usage
    auto space = space_ptr.get();
    auto freq = freq_ptr.get();
    auto host_bk4 = host_bk4_ptr.get();

    // Create logical data
    auto l_freq = ctx.logical_data(freq, {size_t(4 * _rt_bklen)}).set_symbol("l_freq");

    // First task
    ctx.task(ibuffer.l_hist.read(), l_freq.write()).set_symbol("hf_bb_copy_hist")->*[_rt_bklen](cudaStream_t s, auto l_h, auto l_freq) {
      cuda_safe_call(cudaMemcpyAsync(l_freq.data_handle(), l_h.data_handle(),
        sizeof(uint32_t) * _rt_bklen, cudaMemcpyDeviceToHost, s));
    };

    ctx.task(ibuffer.l_bk4.write()).set_symbol("hf_bb_bk4_init")->*[bk_bytes](cudaStream_t s, auto bk4) {
      cuda_safe_call(cudaMemsetAsync(bk4.data_handle(), 0xff, bk_bytes, s));
    };

    cuda_safe_call(cudaStreamSynchronize(ctx.task_fence()));

    ctx.host_launch(ibuffer.l_hist.read(), l_freq.read()).set_symbol("hf_bb_cpu_build_tree")->*
      [space_ptr, host_bk4_ptr, bk_bytes, _rt_bklen, stream](auto l_h, auto l_freq) {

      auto space = space_ptr.get();
      auto host_bk4 = host_bk4_ptr.get();

      printf("Building Huffman book...\n");

      auto bk8 = new uint64_t[_rt_bklen];
      memset(bk8, 0xff, sizeof(uint64_t) * _rt_bklen);

      {
        auto state_num = 2 * _rt_bklen;

        auto tree = create_tree_serial(state_num);

        for (size_t i = 0; i < _rt_bklen; i++)
          if (l_freq(i)) qinsert(tree, new_node(tree, l_freq(i), i, 0, 0));
        while (tree->qend > 2) qinsert(tree, new_node(tree, 0, 0, qremove(tree), qremove(tree)));
        phf_stack<node_t, sizeof(uint64_t)>::template inorder_traverse<uint64_t>(tree->qq[1], bk8);

        destroy_tree(tree);
      }

      // Process the generated codebook
      memset(host_bk4, 0xff, bk_bytes);

      // Convert 64-bit to 32-bit codebook
      for (auto i = 0; i < _rt_bklen; i++) {
        if (*(bk8 + i) != ~((uint64_t)0x0)) {
          auto pw8 = reinterpret_cast<PW8*>(bk8 + i);
          auto pw4 = reinterpret_cast<PW4*>(host_bk4 + i);

          if (pw8->bitcount <= pw4->FIELD_CODE) {
            pw4->bitcount = pw8->bitcount;
            pw4->prefix_code = pw8->prefix_code;
          }
        }
      }

      delete[] bk8;

      space->input_bk() = host_bk4;  // external
      space->canonize();

      cudaMemcpyAsync(host_bk4, space->output_bk(), bk_bytes, cudaMemcpyDeviceToHost, stream);
    };

    cuda_safe_call(cudaStreamSynchronize(ctx.task_fence()));
    auto l_temp_bk4 = ctx.logical_data(host_bk4, {size_t(_rt_bklen)}).set_symbol("l_temp_bk4");

    // Task for copy to device
    ctx.task(ibuffer.l_bk4.write(), l_temp_bk4.read()).set_symbol("hf_bb_copyto_device")->*[bk_bytes]
      (cudaStream_t s, auto bk4, auto temp_bk4) {
        cuda_safe_call(
          cudaMemcpyAsync(bk4.data_handle(), temp_bk4.data_handle(), bk_bytes, cudaMemcpyDeviceToDevice, s));
    };

    // Create logical data for the canonized data
    //? freeze it?
    auto l_first = ctx.logical_data(space_ptr->first(), {size_t(TYPE_BITS)}).set_symbol("l_first");
    auto l_entry = ctx.logical_data(space_ptr->entry(), {size_t(TYPE_BITS)}).set_symbol("l_entry");
    auto l_keys = ctx.logical_data(space_ptr->keys(), {size_t(_rt_bklen)}).set_symbol("l_keys");

    cuda_safe_call(cudaStreamSynchronize(ctx.task_fence()));

    // initialize the logical data for the reverse book
    ctx.task(ibuffer.l_revbk4.write()).set_symbol("hf_bb_init_revbook")->*[reverse_book_len](cudaStream_t s, auto rev_book) {
      cuda_safe_call(cudaMemsetAsync(rev_book.data_handle(), 0, reverse_book_len, s));
    };

    cuda_safe_call(cudaStreamSynchronize(ctx.task_fence()));

    ctx.task(l_first.read(), l_entry.read(), l_keys.read(), ibuffer.l_revbk4.rw()).set_symbol("hf_bb_revbook_concat")->*
        [TYPE_BITS, _rt_bklen](cudaStream_t s, auto first, auto entry, auto keys, auto rev_book) {
        auto offset = 0;
        
        // Copy from logical data to device memory
        cuda_safe_call(cudaMemcpyAsync(
            rev_book.data_handle(), first.data_handle(),
            sizeof(int) * TYPE_BITS, cudaMemcpyHostToDevice, s));
        offset += sizeof(int) * TYPE_BITS;

        cuda_safe_call(cudaMemcpyAsync(
            rev_book.data_handle() + offset, entry.data_handle(),
            sizeof(int) * TYPE_BITS, cudaMemcpyHostToDevice, s));
        offset += sizeof(int) * TYPE_BITS;

        cuda_safe_call(cudaMemcpyAsync(
            rev_book.data_handle() + offset, keys.data_handle(),
            sizeof(uint16_t) * _rt_bklen, cudaMemcpyHostToDevice, s));
    };
  }

  void encode(int pardeg, stf_internal_buffers& ibuffer, context& ctx, cudaStream_t stream) {

    auto div = [](auto whole, auto part) -> uint32_t {
      if (whole == 0) throw std::runtime_error("Dividend is zero.");
      if (part == 0) throw std::runtime_error("Divisor is zero.");
      return (whole - 1) / part + 1;
    };

    auto data_len = len;
    auto numSMs = num_SMs;
    auto blen = rt_bklen;
    // auto revbk4_b = revbk4_bytes;
    auto t_sublen = sublen;

    //! GPU coarse encode phase 1
    ctx.task(ibuffer.l_q_codes.read(), ibuffer.l_bk4.read(), ibuffer.l_scratch.write()).set_symbol("hf_en_phase1")->*[div, data_len, blen, numSMs](cudaStream_t s, auto q_codes, auto book, auto scratch) {
      auto block_dim = 256;
      auto grid_dim = div(data_len, block_dim);
      hf_encode_phase1_fill<<<8 * numSMs, 256, sizeof(uint32_t) * blen, s>>>(
        q_codes, data_len, book, blen, scratch);
    };

    //! GPU_coarse_encode_phase2
    ctx.task(ibuffer.l_scratch.rw(), ibuffer.l_par_nbit.write(), ibuffer.l_par_ncell.write()).set_symbol("hf_en_phase2")->*[div, pardeg, data_len, t_sublen](cudaStream_t s, auto scratch, auto par_nbit, auto par_ncell) {
      auto block_dim = 256;
      auto grid_dim = div(pardeg, block_dim);
      hf_encode_phase2_deflate<<<grid_dim, block_dim, 0, s>>>(scratch, data_len, par_nbit, par_ncell, t_sublen, pardeg);
    };

    ctx.task(ibuffer.l_par_entry.write()).set_symbol("hf_en_parentry_set")->*[pardeg](cudaStream_t s, auto par_entry) {
      cuda_safe_call(cudaMemsetAsync(par_entry.data_handle(), 0, pardeg * sizeof(uint32_t), s));
    };

    ctx.task(ibuffer.l_par_entry.rw(), ibuffer.l_par_ncell.read()).set_symbol("hf_en_par_entry_fill")->*[pardeg](cudaStream_t s, auto par_entry, auto par_ncell) {
      cuda_safe_call(cudaMemcpyAsync(par_entry.data_handle() + 1, par_ncell.data_handle(),
        (pardeg - 1) * sizeof(uint32_t), cudaMemcpyDeviceToDevice, s));
    };

    cuda_safe_call(cudaStreamSynchronize(ctx.task_fence()));

    //! GPU_coarse_encode_phase3_sync
    ctx.host_launch(ibuffer.l_par_nbit.read(), ibuffer.l_par_ncell.read(), ibuffer.l_par_entry.rw()).set_symbol("hf_en_phase3")->*[pardeg, this, stream](auto par_nbit, auto par_ncell, auto par_entry) {
      for (auto i = 1; i < pardeg; i++) {
        par_entry(i) += par_entry(i - 1);
      }
      total_nbit = std::accumulate(par_nbit.data_handle(), par_nbit.data_handle() + pardeg, size_t(0));
      total_ncell = std::accumulate(par_ncell.data_handle(), par_ncell.data_handle() + pardeg, size_t(0));
    };

    // //! GPU_coarse_encode_phase4
    ctx.task(ibuffer.l_scratch.read(), ibuffer.l_par_entry.read(), ibuffer.l_par_ncell.read(), ibuffer.l_bitstream.write()).set_symbol("hf_en_phase4")->*[pardeg, t_sublen](cudaStream_t s, auto scratch, auto par_entry, auto par_ncell, auto bitstream) {
      hf_encode_phase4_concatenate<<<pardeg, 128, 0, s>>>(scratch, par_entry, par_ncell, t_sublen, bitstream);
    };

    cuda_safe_call(cudaStreamSynchronize(ctx.task_fence()));

    ctx.host_launch(ibuffer.hf_header_entry.write()).set_symbol("hf_en_bitstream_setup")->*[&](auto hf_h_entry) {
      // Reset header values
      header.bklen = rt_bklen;
      header.sublen = sublen;
      header.pardeg = pardeg;
      header.original_len = len;
      header.total_nbit = total_nbit;
      header.total_ncell = total_ncell;

      // Calculate section sizes
      uint32_t sizes[HF_HEADER_END + 1];
      sizes[HF_HEADER_HEADER] = HF_HEADER_FORCED_ALIGN;
      sizes[HF_HEADER_REVBK] = revbk4_bytes;
      sizes[HF_HEADER_PAR_NBIT] = pardeg * sizeof(uint32_t);
      sizes[HF_HEADER_PAR_ENTRY] = pardeg * sizeof(uint32_t);
      sizes[HF_HEADER_BITSTREAM] = 4 * total_ncell;

      // Calculate offsets properly (only once)
      header.entry[0] = 0;
      hf_h_entry(0) = 0;
      for (auto i = 1; i < HF_HEADER_END + 1; i++) {
        header.entry[i] = header.entry[i-1] + sizes[i-1];
        hf_h_entry(i) = header.entry[i];
      }

      ibuffer.codec_comp_output_len = header.entry[HF_HEADER_END];

      // print total nbit and total ncell
      printf("total_nbit: %zu\n", total_nbit);
      printf("total_ncell: %zu\n", total_ncell);

      // print header entries
      printf("Header entries: [0]=%u [1]=%u [2]=%u [3]=%u [4]=%u [5]=%u\n",
             header.entry[0], header.entry[1], header.entry[2], header.entry[3],
             header.entry[4], header.entry[5]);
    };

    cuda_safe_call(cudaStreamSynchronize(ctx.task_fence()));

    ctx.task(ibuffer.l_compressed.write()).set_symbol("hf_en_comp_set")->*[data_len](cudaStream_t s, auto compressed) {
      cuda_safe_call(cudaMemsetAsync(compressed.data_handle(), 0, data_len*4/2, s));
    };

    ctx.task(ibuffer.l_compressed.rw()).set_symbol("hf_en_comp_hfheader")->*[this](cudaStream_t s, auto compressed) {
      cuda_safe_call(cudaMemcpyAsync(
        compressed.data_handle() + COMPRESSION_PIPELINE_HEADER_SIZE, &header, sizeof(hf_header), cudaMemcpyHostToDevice, s));
    };

    ctx.task(ibuffer.l_compressed.rw(), ibuffer.l_revbk4.read()).set_symbol("hf_en_comp_revbkheader")->*[this](cudaStream_t s, auto compressed, auto revbook) {
      cuda_safe_call(cudaMemcpyAsync(
        compressed.data_handle() + COMPRESSION_PIPELINE_HEADER_SIZE + header.entry[HF_HEADER_REVBK], revbook.data_handle(),
        revbk4_bytes, cudaMemcpyDeviceToDevice, s));
    };

    ctx.task(ibuffer.l_compressed.rw(), ibuffer.l_par_nbit.read()).set_symbol("hf_en_comp_nbit")->*[this, pardeg](cudaStream_t s, auto compressed, auto par_nbit) {
      cuda_safe_call(cudaMemcpyAsync(
        compressed.data_handle() + COMPRESSION_PIPELINE_HEADER_SIZE + header.entry[HF_HEADER_PAR_NBIT], par_nbit.data_handle(),
        pardeg * sizeof(uint32_t), cudaMemcpyDeviceToDevice, s));
    };

    ctx.task(ibuffer.l_compressed.rw(), ibuffer.l_par_entry.read()).set_symbol("hf_en_comp_entry")->*[this, pardeg](cudaStream_t s, auto compressed, auto par_entry) {
      cuda_safe_call(cudaMemcpyAsync(
        compressed.data_handle() + COMPRESSION_PIPELINE_HEADER_SIZE + header.entry[HF_HEADER_PAR_ENTRY], par_entry.data_handle(),
        pardeg * sizeof(uint32_t), cudaMemcpyDeviceToDevice, s));
    };

    ctx.task(ibuffer.l_compressed.rw(), ibuffer.l_bitstream.read()).set_symbol("hf_en_comp_bitstream")->*[this, data_len](cudaStream_t s, auto compressed, auto bitstream) {
      cuda_safe_call(cudaMemcpyAsync(
        compressed.data_handle() + COMPRESSION_PIPELINE_HEADER_SIZE + header.entry[HF_HEADER_BITSTREAM], bitstream.data_handle(),
        4 * total_ncell, cudaMemcpyDeviceToDevice, s));
    };

  }


};