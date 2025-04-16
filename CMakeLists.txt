cmake_minimum_required(VERSION 3.18)

# needed for WSL cuda
link_directories(/usr/lib/wsl/lib)

project(FZModules LANGUAGES CUDA CXX C VERSION 0.1)

include(GNUInstallDirs)

option(ENABLE_COVERAGE "Enable coverage reporting" OFF)

if(ENABLE_COVERAGE)
  message(STATUS "Compiling with coverage flags")
  add_compile_options(--coverage -O0 -g)
  add_link_options(--coverage)
  # Include GoogleTest
  add_subdirectory(third_party/googletest)
  # Add modularized test suite
  add_subdirectory(tests)
endif()

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# general cmake setup
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(BUILD_SHARED_LIBS "prefer shared libaries" ON)

set(FZMOD_BACKEND "CUDA" CACHE STRING "Select the backend for the FZ Compression Modules library")

# define PSZ_USE_CUDA
if(FZMOD_BACKEND STREQUAL "CUDA")
  add_compile_definitions(FZMOD_USE_CUDA)
  add_compile_definitions(_PORTABLE_USE_CUDA)
endif()

find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

# Add CPM.cmake for fetching CCCL
include(cmake/CPM.cmake)

# Add CCCL package for CUDASTF support
CPMAddPackage(
  NAME CCCL
  GITHUB_REPOSITORY "nvidia/cccl"
  GIT_TAG "main"
  OPTIONS "CCCL_ENABLE_UNSTABLE ON"
)

# Compile settings interface
add_library(fzmod_compile_settings INTERFACE)
target_compile_options(
  fzmod_compile_settings 
  INTERFACE 
    $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda>
    $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Wno-deprecated-declarations>
)
target_compile_features(fzmod_compile_settings INTERFACE cxx_std_17 cuda_std_17)
target_link_libraries(fzmod_compile_settings INTERFACE CUDA::cudart)

# include directories
target_include_directories(fzmod_compile_settings INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/tools/include/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/tools/src/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/modules/codec/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/modules/stat/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/modules/predictor/>
  # fzg codec module:
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/modules/codec/fzg/include/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/modules/codec/fzg/src/>
  # histogram module
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/modules/stat/histogram/include/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/modules/stat/histogram/src/>
  # huffman module
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/modules/codec/huffman/include/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/modules/codec/huffman/src/>
  # lorenzo module
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/modules/predictor/lrz/include/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/modules/predictor/lrz/src/>
  # spline module
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/modules/predictor/spline/include/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/modules/predictor/spline/src/>
  # extrema module
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/modules/stat/extrema/include/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/modules/stat/extrema/src/>
  # spvn module
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/modules/stat/spvn/include/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/modules/stat/spvn/src/>
  # misc
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# LIBRARIES

# FZ utils
add_library(fz_utils
  modules/stat/extrema/src/extrema.cu
  modules/stat/spvn/src/spvn.cu
  modules/stat/extrema/src/calc_err.cu
  modules/stat/extrema/src/max_err.cu
)
target_link_libraries(fz_utils
  PUBLIC
  fzmod_compile_settings
  CUDA::cudart
)

# Individiual Modules

## FZG Codec Module
add_library(fzmod_cu_fzg
  modules/codec/fzg/src/fzg_kernel.cu
  modules/codec/fzg/src/fzg_class.cc
)
target_link_libraries(fzmod_cu_fzg
  PUBLIC
  fzmod_compile_settings
)
add_library(FZMOD::CUDA::fzg ALIAS fzmod_cu_fzg)
add_library(FZMOD::fzg ALIAS fzmod_cu_fzg)

## Histogram Module
add_library(fzmod_cu_histogram
  modules/stat/histogram/src/hist_generic_seq.cc
  modules/stat/histogram/src/hist_generic.cu
  modules/stat/histogram/src/histsp.seq.cc
  modules/stat/histogram/src/histsp.cu
)
target_link_libraries(fzmod_cu_histogram
  PUBLIC
  fzmod_compile_settings
  CUDA::cudart
)
add_library(FZMOD::CUDA::histogram ALIAS fzmod_cu_histogram)
add_library(FZMOD::histogram ALIAS fzmod_cu_histogram)

## Huffman Module
add_library(fzmod_cu_huffman
  modules/codec/huffman/src/hfclass.cc
  # modules/codec/huffman/src/hf_est.cc
  modules/codec/huffman/src/hfbk_impl1.seq.cc
  modules/codec/huffman/src/hfbk_impl2.seq.cc
  modules/codec/huffman/src/hfbk_internal.seq.cc
  modules/codec/huffman/src/hfbk.seq.cc
  modules/codec/huffman/src/hfcanon.seq.cc
  modules/codec/huffman/src/hfcxx_module.cu
  modules/codec/huffman/src/libphf.cc
)
target_link_libraries(fzmod_cu_huffman
  PUBLIC
  fzmod_compile_settings
  CUDA::cuda_driver
)
add_library(FZMOD::CUDA::huffman ALIAS fzmod_cu_huffman)
add_library(FZMOD::huffman ALIAS fzmod_cu_huffman)

# Lorenzo Module
add_library(fzmod_cu_lrz
  modules/predictor/lrz/src/lrz_c.cu
  modules/predictor/lrz/src/lrz_x.cu
)
target_link_libraries(fzmod_cu_lrz
  PUBLIC
  fzmod_compile_settings
  CUDA::cudart  
)
add_library(FZMOD::CUDA::lrz ALIAS fzmod_cu_lrz)
add_library(FZMOD::lrz ALIAS fzmod_cu_lrz)

## Spline Module
add_library(fzmod_cu_spline
  modules/predictor/spline/src/spline3.cu
)
target_link_libraries(fzmod_cu_spline
  PUBLIC
  fzmod_compile_settings
  CUDA::cudart
)
add_library(FZMOD::CUDA::spline ALIAS fzmod_cu_spline)
add_library(FZMOD::spline ALIAS fzmod_cu_spline)

# combine all modules into a single library
add_library(fzmod
  demo.cc
  # pipeline/src/libfzmod.cc
)
target_link_libraries(fzmod
  PUBLIC
  fzmod_compile_settings
  fz_utils
  fzmod_cu_fzg
  fzmod_cu_histogram
  fzmod_cu_huffman
  fzmod_cu_lrz
  fzmod_cu_spline
  CUDA::cudart
)
add_library(FZMOD::CUDA ALIAS fzmod)
add_library(FZMOD::all ALIAS fzmod)

############################

# export binary fzmod
add_executable(fzmod-bin demo.cc)
set_source_files_properties(demo.cc PROPERTIES LANGUAGE CUDA)
target_link_libraries(fzmod-bin PRIVATE fzmod)
set_target_properties(fzmod-bin PROPERTIES OUTPUT_NAME fzmod)

# CUDASTF demo executable
add_executable(cudastf_demo cudastf/cudastf_demo.cc)
set_source_files_properties(cudastf/cudastf_demo.cc PROPERTIES LANGUAGE CUDA)
target_link_libraries(cudastf_demo 
  PRIVATE 
  fzmod_compile_settings
  CUDA::cudart
  CUDA::cuda_driver
  CCCL::CCCL 
  CCCL::cudax
)

## INSTALLATION

# install libs
install(TARGETS fzmod_compile_settings EXPORT FZModulesTargets)
install(TARGETS 
  fz_utils
  fzmod_cu_fzg 
  fzmod_cu_histogram 
  fzmod_cu_huffman 
  fzmod_cu_lrz 
  fzmod_cu_spline 
  fzmod 
  EXPORT FZModulesTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# install executable
install(TARGETS fzmod-bin
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# install executable
install(TARGETS fzmod-bin cudastf_demo
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# install the package
install(
  EXPORT FZModulesTargets
  NAMESPACE FZMOD::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FZModules)

include(CMakePackageConfigHelpers)

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FZModulesConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/FZModulesConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FZModules
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/FZModulesConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY AnyNewerVersion
)

install(FILES 
  "${CMAKE_CURRENT_BINARY_DIR}/FZModulesConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/FZModulesConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FZModules
)

# install headers
install(DIRECTORY
  tools/include/
  modules/codec/fzg/include/
  modules/stat/histogram/include/
  modules/codec/huffman/include/
  modules/predictor/lrz/include/
  modules/predictor/spline/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/fzmod
)