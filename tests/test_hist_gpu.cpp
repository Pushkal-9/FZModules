#include <gtest/gtest.h>
#include "hist.hh"
#include <vector>
#include <cuda_runtime.h>

TEST(StatTest, HistogramGenericGPU_Counting) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    uint8_t h_input[] = {0, 1, 2, 3};
    size_t len = 4;
    uint8_t *d_input;
    uint32_t *d_hist;
    std::vector<uint32_t> h_hist(5, 0);

    cudaMalloc(&d_input, len * sizeof(uint8_t));
    cudaMalloc(&d_hist, 5 * sizeof(uint32_t));
    cudaMemcpy(d_input, h_input, len * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemset(d_hist, 0, 5 * sizeof(uint32_t));

    int res = fz::module::GPU_histogram_generic<uint8_t>(d_input, len, d_hist, 5, 1, 128, 0, 1, stream);
    cudaMemcpy(h_hist.data(), d_hist, 5 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(stream);

    EXPECT_EQ(res, 0);
    EXPECT_EQ(h_hist[0], 1u);
    EXPECT_EQ(h_hist[1], 1u);
    EXPECT_EQ(h_hist[2], 1u);
    EXPECT_EQ(h_hist[3], 1u);

    cudaFree(d_input);
    cudaFree(d_hist);
    cudaStreamDestroy(stream);
}