cmake_minimum_required(VERSION 3.20)
project(tron_gpu_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

get_filename_component(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
file(GLOB NVIDIA_SITE_PACKAGES "${PROJECT_ROOT}/.venv/lib/python*/site-packages/nvidia")
list(GET NVIDIA_SITE_PACKAGES 0 NVIDIA_ROOT)

find_path(CUDA_DRIVER_INCLUDE_DIR cuda.h
  PATHS "${NVIDIA_ROOT}/cuda_runtime/include"
  NO_DEFAULT_PATH REQUIRED)
find_path(NVRTC_INCLUDE_DIR nvrtc.h
  PATHS "${NVIDIA_ROOT}/cuda_nvrtc/include"
  NO_DEFAULT_PATH REQUIRED)
find_library(CUDA_DRIVER_LIBRARY cuda REQUIRED)
find_library(NVRTC_LIBRARY
  NAMES nvrtc libnvrtc.so.12
  PATHS "${NVIDIA_ROOT}/cuda_nvrtc/lib"
  NO_DEFAULT_PATH REQUIRED)
find_package(OpenSSL 3 REQUIRED COMPONENTS Crypto)
find_package(Threads REQUIRED)

add_executable(tron_gpu_benchmark
  src/main.cpp
  src/pattern.cpp
  src/crypto.cpp
  src/gpu.cpp)

target_include_directories(tron_gpu_benchmark PRIVATE
  include
  "${CUDA_DRIVER_INCLUDE_DIR}"
  "${NVRTC_INCLUDE_DIR}")
target_compile_definitions(tron_gpu_benchmark PRIVATE
  TRON_DEFAULT_KERNEL_PATH="${PROJECT_ROOT}/vendor/tron_vanity_cuda/kernels.cu")
target_compile_options(tron_gpu_benchmark PRIVATE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
target_link_libraries(tron_gpu_benchmark PRIVATE
  OpenSSL::Crypto Threads::Threads ${CUDA_DRIVER_LIBRARY} ${NVRTC_LIBRARY} ${CMAKE_DL_LIBS})

# The project intentionally uses the pip-distributed NVRTC library so a full
# CUDA toolkit/nvcc install is not needed on the benchmark host.
get_filename_component(NVRTC_LIBRARY_DIR "${NVRTC_LIBRARY}" DIRECTORY)
set_target_properties(tron_gpu_benchmark PROPERTIES
  BUILD_RPATH "${NVRTC_LIBRARY_DIR}"
  INSTALL_RPATH "${NVRTC_LIBRARY_DIR}")

enable_testing()
add_executable(pattern_tests tests/pattern_tests.cpp src/pattern.cpp)
target_include_directories(pattern_tests PRIVATE include)
target_compile_options(pattern_tests PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME pattern_tests COMMAND pattern_tests)

add_executable(crypto_tests tests/crypto_tests.cpp src/crypto.cpp)
target_include_directories(crypto_tests PRIVATE include)
target_compile_options(crypto_tests PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(crypto_tests PRIVATE OpenSSL::Crypto Threads::Threads)
add_test(NAME crypto_tests COMMAND crypto_tests)
