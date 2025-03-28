# Add the test executable with all modular test files
add_executable(fzmodules_tests
  main.cpp
  test_config.cpp
  test_fileio.cpp
  test_hist_seq.cpp
  test_hist_gpu.cpp
  test_hist_optimizer.cpp
  test_codec_fzg.cpp
  test_codec_huffman.cpp
)

# Link with FZModules library and GoogleTest
target_link_libraries(fzmodules_tests PRIVATE
  fzmod
  gtest
  gtest_main
  pthread
  cuda
)

# Include necessary headers
target_include_directories(fzmodules_tests PRIVATE
  ${PROJECT_SOURCE_DIR}/modules/stat/histogram/include
  ${PROJECT_SOURCE_DIR}/tools/include
  ${PROJECT_SOURCE_DIR}/modules/codec/huffman/include
  ${PROJECT_SOURCE_DIR}/modules/codec/fzg/include
)

# Enable test discovery
enable_testing()
add_test(NAME FZModulesTests COMMAND fzmodules_tests)