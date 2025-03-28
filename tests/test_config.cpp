#include <gtest/gtest.h>
#include "config.hh"
#include "fzg_class.hh"

TEST(ConfigTest, DimensionAndAnchorCalculation) {
    fz::Config conf3d(4, 5, 6);
    EXPECT_EQ(conf3d.len, 4u * 5u * 6u);
    EXPECT_EQ(conf3d.orig_size, conf3d.len * sizeof(float));
    std::array<size_t, 3> anchorLens = conf3d.anchor_len3();
    EXPECT_EQ(anchorLens[0], (4 - 1) / 8 + 1);
    EXPECT_EQ(anchorLens[1], (5 - 1) / 8 + 1);
    EXPECT_EQ(anchorLens[2], (6 - 1) / 8 + 1);

    fz::Config conf1d(10, 1, 1);
    EXPECT_EQ(conf1d.len, 10u);
    EXPECT_GT(conf1d.orig_size, 0u);
    std::array<size_t, 3> anchorLens1d = conf1d.anchor_len3();
    EXPECT_EQ(anchorLens1d[0], (10 - 1) / 8 + 1);
    EXPECT_EQ(anchorLens1d[1], (1 - 1) / 8 + 1);
    EXPECT_EQ(anchorLens1d[2], (1 - 1) / 8 + 1);
}