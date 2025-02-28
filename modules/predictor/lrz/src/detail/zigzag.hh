/**
 * @file suint.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-01-05
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef A2F829BC_283B_488E_A52B_CD597670EC52
#define A2F829BC_283B_488E_A52B_CD597670EC52

#include <cstdint>
#include <cstdlib>
#include <sstream>

// #include "typing.hh"

namespace fz {

template <int ByteWidth> struct SInt;
template <> struct SInt<1> { using T =  int8_t; }; 
template <> struct SInt<2> { using T = int16_t; }; 
template <> struct SInt<4> { using T = int32_t; }; 
template <> struct SInt<8> { using T = int64_t; };
template <int ByteWidth> using SInt_t = typename SInt<ByteWidth>::T;

template <int ByteWidth> struct UInt;
template <> struct UInt<1> { using T =  uint8_t; }; 
template <> struct UInt<2> { using T = uint16_t; }; 
template <> struct UInt<4> { using T = uint32_t; }; 
template <> struct UInt<8> { using T = uint64_t; };
template <int ByteWidth> using UInt_t = typename UInt<ByteWidth>::T;

// ZigZag encoding
// reference:
// https://lemire.me/blog/2022/11/25/making-all-your-integers-positive-with-zigzag-encoding/
template <typename T>
struct ZigZag {
 public:
  static constexpr int ByteWidth = sizeof(T);
  using UInt = UInt_t<ByteWidth>;
  using SInt = SInt_t<ByteWidth>;

 private:
  static constexpr int BitWidth = ByteWidth * 8;

 public:
  // force type checking to unsure the bitwidth
  template <typename _SUPPOSED_SINT>
  [[nodiscard]] static constexpr
      typename std::enable_if_t<std::is_same_v<_SUPPOSED_SINT, SInt>, UInt>
      encode(_SUPPOSED_SINT const x) {
    static_assert(
        std::is_same_v<_SUPPOSED_SINT, SInt>,
        "[ZigZag] encode() input must be a SIGNED integer, whose bitwidth is "
        "the same as T in ZigZag<T>.");
    return (x << 1) ^ (x >> (BitWidth - 1));
  }

  // force type checking to unsure the bitwidth
  template <typename _SUPPOSED_UINT>
  [[nodiscard]] static constexpr
      typename std::enable_if_t<std::is_same_v<_SUPPOSED_UINT, UInt>, SInt>
      decode(_SUPPOSED_UINT const x) {
    static_assert(
        std::is_same_v<_SUPPOSED_UINT, UInt>,
        "[ZigZag] decode() input must be an UNSIGNED integer, whose bitwidth "
        "is the same as T in ZigZag<T>.");
    return (x >> 1) ^ (-(x & 1));
  }
};

}  // namespace fz

#endif /* A2F829BC_283B_488E_A52B_CD597670EC52 */
