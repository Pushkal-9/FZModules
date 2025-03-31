#include <gtest/gtest.h>
#include "hfclass.hh"
#include <cuda_runtime.h> 

// Huffman Codec Tests
TEST(CodecTest, HuffmanCodecBasicProperties) {
    size_t len = 10;
    int pardeg = 1;
    phf::HuffmanCodec<uint16_t> codec(len, pardeg);
    EXPECT_EQ(codec.inlen(), len);
    EXPECT_EQ(codec.time_book(), 0.0f);
    EXPECT_EQ(codec.time_lossless(), 0.0f);
}

TEST(CodecTest, HuffmanCodecRoundTripMultiple) {
    size_t len = 20;
    int pardeg = 1;
    phf::HuffmanCodec<uint16_t> codec(len, pardeg);

    std::vector<uint16_t> data1(len, 42);
    PHF_BYTE* compData1 = nullptr;
    size_t compSize1 = 0;
    codec.encode(data1.data(), len, &compData1, &compSize1, nullptr);
    ASSERT_NE(compData1, nullptr);
    EXPECT_GT(compSize1, 0u);
    std::vector<uint16_t> outData1(len);
    codec.decode(compData1, outData1.data(), nullptr);
    for (size_t i = 0; i < len; ++i) {
        EXPECT_EQ(outData1[i], data1[i]);
    }
    cudaFreeHost(compData1);
    phf::HuffmanCodec<uint16_t>* retPtr = codec.clear_buffer();
    EXPECT_EQ(retPtr, &codec);

    std::vector<uint16_t> data2(len);
    for (size_t i = 0; i < len; ++i) {
        data2[i] = i % 2;
    }
    PHF_BYTE* compData2 = nullptr;
    size_t compSize2 = 0;
    codec.encode(data2.data(), len, &compData2, &compSize2, nullptr);
    ASSERT_NE(compData2, nullptr);
    EXPECT_GT(compSize2, 0u);
    std::vector<uint16_t> outData2(len);
    codec.decode(compData2, outData2.data(), nullptr);
    for (size_t i = 0; i < len; ++i) {
        EXPECT_EQ(outData2[i], data2[i]);
    }
    cudaFreeHost(compData2);
}