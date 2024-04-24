cmake_minimum_required(VERSION 3.10)
project(GQHypocenterSearch)

# Define default values for BLOCK_HYPOCS and TILE
if(NOT DEFINED BLOCK_HYPOCS)
    set(BLOCK_HYPOCS_DEFAULT 512)
    set(BLOCK_HYPOCS ${BLOCK_HYPOCS_DEFAULT})
endif()

if(NOT DEFINED TILE)
    set(TILE_DEFAULT 5)
    set(TILE ${TILE_DEFAULT})
endif()

if(NOT DEFINED TESTS)
    set(TESTS_DEFAULT 16)
    set(TESTS ${TESTS_DEFAULT})
endif()

# Add definitions for CUDA_NVCC_FLAGS based on BLOCK_HYPOCS and TILE
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} $ENV{CUDA_NVCC_FLAGS} --use_fast_math -DTESTS=${TESTS} -DBLOCK_HYPOCS=${BLOCK_HYPOCS} -DTILE=${TILE}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBLOCK_HYPOCS=${BLOCK_HYPOCS} -DTESTS=${TESTS} -DTILE=${TILE}")

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

# Find JNI (Java Native Interface)
find_package(JNI REQUIRED)
include_directories(${JNI_INCLUDE_DIRS})

# Compile CUDA sources into a shared library
cuda_add_library(gq_hypocs SHARED ${SRC_FILES})

# Add an executable that links to the shared library
add_executable(gq_test src/main.cpp)

target_link_libraries(gq_test gq_hypocs)

# Set C++11 standard
set_property(TARGET gq_test PROPERTY CXX_STANDARD 11)
