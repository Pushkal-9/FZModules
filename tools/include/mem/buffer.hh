#ifndef FZMOD_BUFFER_HH
#define FZMOD_BUFFER_HH

#include "mem/cxx_sp_gpu.h"
#include "config.hh"

#define MAKE_STDLEN3(X, Y, Z) \
  std::array<size_t, 3> { X, Y, Z }

#define ALIGN_4Ki(len) (((len) + 4095) & ~4095)

namespace fz {

template <typename DType>
class InternalBuffer {
  public:
    using Compact = _portable::compact_gpu<DType>;

    Config* conf;

    uint8_t* codec_comp_output{nullptr};
    size_t codec_comp_output_len{0};

    GPU_unique_dptr<uint16_t[]> d_ectrl;
    GPU_unique_dptr<DType[]> d_anchor;
    GPU_unique_dptr<uint8_t[]> d_compressed;
    GPU_unique_hptr<uint8_t[]> h_compressed;
    GPU_unique_dptr<uint32_t[]> d_hist;
    GPU_unique_dptr<uint32_t[]> d_top1;
    GPU_unique_hptr<uint32_t[]> h_top1;
    Compact* compact;
    uint32_t x, y, z;
    size_t const len;

    InternalBuffer(Config* config, uint32_t x, uint32_t y = 1, uint32_t z = 1) :
      x(x), y(y), z(z), len(x * y * z), conf(config) {
      
      d_ectrl = MAKE_UNIQUE_DEVICE(uint16_t, ALIGN_4Ki(len)); // for FZGPU

      compact = new Compact(len/ 5);
      d_anchor = MAKE_UNIQUE_DEVICE(DType, conf->anchor512_len);
      d_hist = MAKE_UNIQUE_DEVICE(uint32_t, conf->radius * 2);
      d_compressed = MAKE_UNIQUE_DEVICE(uint8_t, len * 4 / 2);
      h_compressed = MAKE_UNIQUE_HOST(uint8_t, len * 4 / 2);
      d_top1 = MAKE_UNIQUE_DEVICE(uint32_t, 1);
      h_top1 = MAKE_UNIQUE_HOST(uint32_t, 1);
      
    }

    ~InternalBuffer() {
      delete compact;
    }

    // Getters
    uint16_t* ectrl() const { return d_ectrl.get(); }

    uint32_t* hist() const { return d_hist.get(); }
    uint32_t* top1() const { return d_top1.get(); }
    uint32_t* top1_h() const
    {
      cudaMemcpy(h_top1.get(), d_top1.get(), sizeof(uint32_t), cudaMemcpyDeviceToHost);
      return h_top1.get();
    }

    void clear_top1() { cudaMemset(d_top1.get(), 1, sizeof(uint32_t)); }
    std::array<size_t, 3> ectrl_len3() const { return std::array<size_t, 3>{x, y, z}; }
    uint8_t* compressed() { return d_compressed.get(); }
    uint8_t* compressed_h() { return h_compressed.get(); }
    DType* anchor() const { return d_anchor.get(); }
    DType* compact_val() const { return compact->val(); }
    uint32_t* compact_idx() const { return compact->idx(); }
    uint32_t* compact_num_outliers() const { return compact->num_outliers(); }
    Compact* outlier() { return compact; }

}; // InternalBuffer

} // namespace fz


#endif /* FZMOD_BUFFER_HH */