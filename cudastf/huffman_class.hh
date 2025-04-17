#include <memory>

#include "huffman/include/hfcanon.hh"
#include "huffman/include/hfword.hh"
#include "huffman/include/hfbk_impl.hh"

using namespace cuda::experimental::stf;

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

struct stf_huff_buff {
  stf_huff_buff(
    const context& _ctx,
    size_t revbk4_bytes,
    size_t bk_bytes,
    
  ) : ctx(_ctx) 
  {

  }

  logical_data<slice<uint32_t>> l_bk4;
  logical_data<slice<uint8_t>> revbk4;
  context ctx;
};

class HuffmanCodecSTF {

  public:

  int num_SMs;
  size_t pardeg, len, sublen;
  static constexpr uint16_t max_bklen  = 1024;

  size_t revbk4_bytes;


  HuffmanCodecSTF(size_t const inlen, int const _pardeg, context& ctx) {

    cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, 0);

    pardeg = _pardeg;
    len = inlen;
    sublen = (len - 1) / pardeg + 1;

    revbk4_bytes = 4 * (2 * 4 * 8) + sizeof(uint16_t) * max_bklen;

    // allocate memory
    auto l_bk4 = ctx.logical_data<uint32_t>(max_bklen);
    auto revbk4 = ctx.logical_data<uint8_t>(revbk4_bytes);

  }

  void buildbook(logical_data<slice<float>>& l_hist, uint16_t const _rt_bklen, context& ctx) {
    
    constexpr auto TYPE_BITS = sizeof(uint32_t) * 8;
    auto bk_bytes = sizeof(uint32_t) * _rt_bklen;
    auto space_bytes = sizeof(uint32_t) * (3 * _rt_bklen) + sizeof(uint32_t) * (4 * (sizeof(uint32_t) * 8)) + sizeof(uint16_t) * _rt_bklen;
    auto revbok_ofst = sizeof(uint32_t) * (3 * _rt_bklen) + sizeof(uint32_t) * (2 * TYPE_BITS);
    auto space = new hf_canon_reference<uint16_t, uint32_t>(_rt_bklen);

    // ctx.host_launch()->*[&]() {

    // };

    // memset(l_bk4.data_handle(), 0xff, bk_bytes);

    // // part 1
    // {
    //   auto state_num = 2 * _rt_bklen;
    //   auto all_nodes = 2 * state_num;

    //   auto freq = ctx.logical_data(shape_of<slice<uint32_t>>(all_nodes));
    //   memset(freq.data_handle(), 0, sizeof(uint32_t) * all_nodes);
    //   memcpy(freq.data_handle(), l_hist.data_handle(), sizeof(uint32_t) * _rt_bklen);

    //   auto tree = create_tree_serial(state_num);

    //   {
    //     for (size_t i = 0; i < tree->all_nodes; i++) {
    //       if (freq(i)) qinsert(tree, new_node(tree, freq(i), i, 0, 0));
    //     }
    //     while (tree->qend > 2) qinsert(tree, new_node(tree, 0, 0, qremove(tree), qremove(tree)));
    //     phf_stack<node_t, sizeof(uint32_t)>::template inorder_traverse<uint32_t>(tree->qq[1], l_bk4.data_handle());
    //   }

    //   destroy_tree(tree);

    // }

    // space->input_bk() = l_bk4.data_handle();

    // {  // part 2
    //   space->canonize();
    // }

    // // copy to output1
    // memcpy(l_bk4.data_handle(), space->output_bk(), bk_bytes);

    // // copy to output2
    // auto offset = 0;
    // memcpy(revbk4.data_handle(), space->first(), sizeof(int) * TYPE_BITS);
    // offset += sizeof(int) * TYPE_BITS;
    // memcpy(revbk4.data_handle() + offset, space->entry(), sizeof(int) * TYPE_BITS);
    // offset += sizeof(int) * TYPE_BITS;
    // memcpy(revbk4.data_handle() + offset, space->keys(), sizeof(uint16_t) * _rt_bklen);

    // delete space;

  }

};