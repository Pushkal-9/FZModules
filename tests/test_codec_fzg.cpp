#include <gtest/gtest.h>
#include "fzg_class.hh"
#include <vector>

TEST(CodecTest, FzgCodecMultipleUse) {
    size_t len = 20;
    fz::FzgCodec codec(len);

    std::vector<uint16_t> data1(len, 0);
    uint8_t* compData1 = nullptr;
    size_t compSize1 = 0;
    size_t paddedLen1 = codec.expose_padded_input_len();

    // Encode first input
    ASSERT_NO_THROW(codec.encode(data1.data(), paddedLen1, &compData1, &compSize1, nullptr));
    ASSERT_NE(compData1, nullptr);
    EXPECT_GT(compSize1, 0u);

    std::vector<uint16_t> outData1(paddedLen1);
    codec.decode(compData1, compSize1, outData1.data(), paddedLen1, nullptr);
    EXPECT_EQ(std::vector<uint16_t>(outData1.begin(), outData1.begin() + len), data1);
    cudaFreeHost(compData1);
    codec.clear_buffer();

    // Second encode/decode
    std::vector<uint16_t> data2(len);
    for (size_t i = 0; i < len; ++i) data2[i] = i % 5;

    size_t paddedLen2 = codec.expose_padded_input_len();  // 💡 Important
    data2.resize(paddedLen2, 0);  // pad with 0s

    uint8_t* compData2 = nullptr;
    size_t compSize2 = 0;

    ASSERT_NO_THROW(codec.encode(data2.data(), paddedLen2, &compData2, &compSize2, nullptr));
    ASSERT_NE(compData2, nullptr);
    EXPECT_GT(compSize2, 0u);

    std::vector<uint16_t> outData2(paddedLen2);  // 💡 match padded size
    ASSERT_NO_THROW(codec.decode(compData2, compSize2, outData2.data(), paddedLen2, nullptr));

    for (size_t i = 0; i < len; ++i) {
        EXPECT_EQ(outData2[i], data2[i]);
    }

    cudaFreeHost(compData2);
}
