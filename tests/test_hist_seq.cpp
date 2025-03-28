#include <gtest/gtest.h>
#include "hist.hh"
#include <vector>

// Histogram Tests
TEST(StatTest, HistogramGenericSEQ_Counting) {
    uint8_t input[] = {0, 2, 2, 7, 7, 7};
    size_t len = sizeof(input) / sizeof(input[0]);
    std::vector<uint32_t> hist(10, 0);
    float ms = 0.0f;

    int res = fz::module::SEQ_histogram_generic<uint8_t>(input, len, hist.data(), 10, &ms);
    EXPECT_EQ(res, 0);
    EXPECT_EQ(hist[0], 1u);
    EXPECT_EQ(hist[2], 2u);
    EXPECT_EQ(hist[7], 3u);
    EXPECT_GE(ms, 0.0f);
}

TEST(StatTest, HistogramCauchySEQ_Counting) {
    uint8_t input[] = {1, 1, 2, 3};
    std::vector<uint32_t> hist(5, 0);
    float ms = 0.0f;
    int res = fz::module::SEQ_histogram_Cauchy_v2<uint8_t>(input, 4, hist.data(), 5, &ms);
    EXPECT_EQ(res, 0);
    EXPECT_EQ(hist[1], 2u);
    EXPECT_EQ(hist[2], 1u);
    EXPECT_EQ(hist[3], 1u);
}

TEST(StatTest, HistogramEmptyInput) {
    uint16_t hist_len = 5;
    std::vector<uint32_t> histogram(hist_len, 123);
    float milliseconds = 0.0f;
    int result = fz::module::SEQ_histogram_generic<uint8_t>(
        nullptr, 0, histogram.data(), hist_len, &milliseconds
    );
    EXPECT_EQ(result, 0);
    for (auto& val : histogram) {
        EXPECT_EQ(val, 123u);
    }
    EXPECT_GE(milliseconds, 0.0f);
}
