#include <gtest/gtest.h>
#include "io.hh"
#include <vector>
#include <string>
#include <fstream>
#include <cstdio>

TEST(FileIOTest, ReadWriteSuccess) {
    std::vector<int> outData = {1, 2, 3, 4, 5};
    const std::string fname = "test_io.bin";
    int writeResult = _portable::utils::tofile(fname, outData.data(), outData.size());
    ASSERT_EQ(writeResult, 0);
    std::vector<int> inData(outData.size());
    int readResult = _portable::utils::fromfile(fname, inData.data(), inData.size());
    ASSERT_EQ(readResult, 0);
    EXPECT_EQ(inData, outData);
    std::remove(fname.c_str());
}

TEST(FileIOTest, ReadNonexistentFile) {
    std::vector<char> buffer(10);
    int result = _portable::utils::fromfile("nonexistent_file.bin", buffer.data(), buffer.size());
    EXPECT_EQ(result, -1);
}

TEST(FileIOTest, ReadNullPointer) {
    const std::string fname = "dummy.bin";
    {
        std::ofstream ofs(fname, std::ios::binary);
        ofs << "test";
    }
    int result = _portable::utils::fromfile(fname, (char*)nullptr, 4);
    EXPECT_EQ(result, 1);
    std::remove(fname.c_str());
}

TEST(FileIOTest, WriteFail) {
    std::vector<int> data = {10, 20, 30};
    std::string badPath = "/no_such_dir/test.bin";
    int result = _portable::utils::tofile(badPath, data.data(), data.size());
    EXPECT_EQ(result, -2);
}
