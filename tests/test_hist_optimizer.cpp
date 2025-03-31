#include <gtest/gtest.h>
#include "hist.hh"

TEST(StatTest, HistogramOptimizerInitialization) {
    int grid = 0, block = 0, shmem = 0, rpb = 0;
    fz::module::GPU_histogram_generic_optimizer_on_initialization<uint8_t>(
        1000, 10, grid, block, shmem, rpb
    );
    EXPECT_GT(grid, 0);
    EXPECT_GT(block, 0);
    EXPECT_GE(shmem, 0);
    EXPECT_GT(rpb, 0);
}