cmake_minimum_required(VERSION 3.18)

project(FZModules LANGUAGES CUDA CXX C VERSION 0.1)

include(GNUInstallDirs)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# general cmake setup
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(BUILD_SHARED_LIBS "prefer shared libaries" ON)

set(FZMOD_BACKEND "CUDA" CACHE STRING "Select the backend for the FZ Compression Modules library")

find_package(CUDAToolkit REQUIRED)

# define compile settings cmake interface library
add_library(fzmod_compile_settings INTERFACE)
target_compile_definitions(
  fzmod_compile_settings 
  INTERFACE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:
  --extended-lambda 
  --expt-relaxed-constexpr
  -Wno-deprecated-declarations
)
target_compile_features(fzmod_compile_settings INTERFACE cxx_std_17 cuda_std_17)

# include directories
target_include_directories(fzmod_compile_settings INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/portable/include/>
  # fzg codec module:
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/modules/fzg/include/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/modules/fzg/src/>
  # histogram module
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/modules/histogram/include/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/modules/histogram/src/>
  # huffman module
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/modules/huffman/include/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/modules/huffman/src/>
  # lorenzo module
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/modules/lrz/include/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/modules/lrz/src/>
  # spline module
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/modules/spline/include/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/modules/spline/src/>
  # pipeline
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/pipeline/include/>
  # misc
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# LIBRARIES

## FZ Portable MEM module
add_library(fz_mem
  portable/src/mem/memobj.f.cc
  portable/src/mem/memobj.i.cc
  portable/src/mem/memobj.u.cc
  portable/src/mem/memobj.misc.cc)
add_library(FZMOD::mem ALIAS fz_mem)
add_library(FZMOD::cu_mem ALIAS fz_mem)
target_link_libraries(fz_mem
  PUBLIC
  fzmod_compile_settings
  CUDA::cudart
)

# Individiual Modules

## FZG Codec Module
add_library(fzmod_cu_fzg
  codec/fzg/src/fzg_kernel.cu
  codec/fzg/src/fzg_class.cc
)
target_link_libraries(fzmod_cu_fzg
  PUBLIC
  fzmod_compile_settings
)
add_library(FZMOD::CUDA::fzg ALIAS fzmod_cu_fzg)
add_library(FZMOD::fzg ALIAS fzmod_cu_fzg)

## Histogram Module
add_library(fzmod_cu_histogram
  modules/histogram/src/hist_generic_seq.cc
  modules/histogram/src/hist_generic.cu
  modules/histogram/src/histsp.seq.cc
  modules/histogram/src/histsp.cu
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
  modules/huffman/src/hfclass.cc
  modules/huffman/src/hf_est.cc
  modules/huffman/src/hfbk_impl1.seq.cc
  modules/huffman/src/hfbk_impl2.seq.cc
  modules/huffman/src/hfbk_internal.seq.cc
  modules/huffman/src/hfbk.seq.cc
  modules/huffman/src/hfcanon.seq.cc
  modules/huffman/src/hfcxx_module.cu
  modules/huffman/src/libphf.cc
)
target_link_libraries(fzmod_cu_huffman
  PUBLIC
  fzmod_compile_settings
  CUDA::cuda_driver
)
add_library(FZMOD::CUDA::huffman ALIAS fzmod_cu_huffman)
add_library(FZMOD::huffman ALIAS fzmod_cu_huffman)

## Lorenzo Module
add_library(fzmod_cu_lrz
  modules/lrz/src/lrz_c.cu
  modules/lrz/src/lrz_x.cu
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
  modules/spline/src/spline3.cu
)
target_link_libraries(fzmod_cu_spline
  PUBLIC
  fzmod_compile_settings
  CUDA::cudart
)
add_library(FZMOD::CUDA::spline ALIAS fzmod_cu_spline)
add_library(FZMOD::spline ALIAS fzmod_cu_spline)

# combine all modules into a single library
add_library(cusz
  # TODO: API HERE
)
target_link_libraries(fzmod
  PUBLIC
  fzmod_compile_settings
  # modules
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
add_executable(fzmod-bin pipeline/src/driver.cc)
set_source_files_properties(pipeline/src/driver.cc PROPERTIES LANGUAGE CUDA)
target_link_libraries(fzmod-bin PRIVATE fzmod)
set_target_properties(fzmod-bin PROPERTIES OUTPUT_NAME fzmod)

## INSTALLATION

# install libs
install(TARGETS fzmod_compile_settings EXPORT FZModulesTargets)
install(TARGETS 
  fz_mem 
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
  portable/include/
  modules/fzg/include/
  modules/histogram/include/
  modules/huffman/include/
  modules/lrz/include/
  modules/spline/include/
  pipeline/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/fzmod
)
