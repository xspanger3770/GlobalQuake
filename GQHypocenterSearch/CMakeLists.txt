cmake_minimum_required(VERSION 3.10)
project(GQHypocenterSearch)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} $ENV{CUDA_NVCC_FLAGS} -lineinfo -g")

# Find CUDA package
find_package(CUDA REQUIRED)

# Set CUDA architecture (adjust it to your target GPU architecture)
set(CUDA_ARCH_LIST "35" CACHE STRING "List of CUDA architectures (e.g., 35, 61, 75)")

# Set source directory
set(SRC_DIR ./src)

# Collect source files
file(GLOB SRC_FILES ${SRC_DIR}/*.c*)

# Set output directory for binaries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Set output directory for the shared library
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# CUDA-specific settings
foreach(arch ${CUDA_ARCH_LIST})
    set(CUDA_NVCC_FLAGS_${arch} "-gencode arch=compute_${arch},code=sm_${arch}")
endforeach()

include_directories(/usr/lib/jvm/java-17-openjdk-amd64/include/)
include_directories(/usr/lib/jvm/java-17-openjdk-amd64/include/linux)

# Compile CUDA sources into a shared library
cuda_add_library(gq_hypocs SHARED ${SRC_FILES})

# Add an executable that links to the shared library
add_executable(gq_test src/main.cpp)

target_link_libraries(gq_test gq_hypocs)

# Set C++11 standard
set_property(TARGET gq_test PROPERTY CXX_STANDARD 11)
